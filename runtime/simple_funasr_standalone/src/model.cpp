#include "model.h"
#include "com-define.h"
#include <fstream>
#include <numeric>
#include <sstream>
#include <cmath>

namespace funasr {

Paraformer::Paraformer() {}
Paraformer::~Paraformer() {}

void Paraformer::LoadCmvn(const std::string& mvn_file) {
    std::ifstream in(mvn_file);
    if (!in.is_open()) {
        std::cerr << "Warning: Failed to load am.mvn from " << mvn_file << std::endl;
        return;
    }
    std::string line;
    while (getline(in, line)) {
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "<AddShift>") {
             int dim;
             ss >> dim; // read output dim (560)
             ss >> dim; // read input dim (560)
             
             // The next token is <LearnRateCoef> or just the vector start?
             // In the file provided: <AddShift> 560 560 \n <LearnRateCoef> 0 [ ... ]
             // So the vector is on the NEXT line or after <LearnRateCoef> tag.
             
             // Let's read the vector from the stream until we find '['
             // But based on the file content provided:
             // <AddShift> 560 560 
             // <LearnRateCoef> 0 [ -8.31... ]
             
             // So we need to read the NEXT line.
             if (!getline(in, line)) break;
             std::stringstream ss2(line);
             std::string tag2;
             ss2 >> tag2; // <LearnRateCoef>
             float lr;
             ss2 >> lr; // 0
             char bracket;
             ss2 >> bracket; // [
             
             means_.resize(dim);
             for(int i=0; i<dim; ++i) ss2 >> means_[i];
             // The last one might have ']' attached or separate?
             // usually kaldi prints space before ]
        }
        else if (tag == "<Rescale>") {
             int dim;
             ss >> dim; 
             ss >> dim;
             
             if (!getline(in, line)) break;
             std::stringstream ss2(line);
             std::string tag2;
             ss2 >> tag2; // <LearnRateCoef>
             float lr;
             ss2 >> lr; // 0
             char bracket;
             ss2 >> bracket; // [
             
             vars_.resize(dim);
             for(int i=0; i<dim; ++i) ss2 >> vars_[i];
        }
    }
    std::cout << "Loaded CMVN: dim=" << means_.size() << std::endl;
}

void Paraformer::ApplyCmvn(std::vector<float>& feats, int dim) {
    if (means_.empty() || vars_.empty()) return;
    if (means_.size() != dim || vars_.size() != dim) {
        std::cerr << "CMVN dim mismatch!" << std::endl;
        return;
    }
    
    int num_frames = feats.size() / dim;
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < dim; ++j) {
            float* val = &feats[i * dim + j];
            // f = (f + mean) * var
            // NOTE: In Kaldi <AddShift>, the values are usually negative means.
            // And <Rescale> values are 1/stddev.
            // So the formula is correct: val + add_shift * rescale
            *val = (*val + means_[j]) * vars_[j];
        }
    }
}

void Paraformer::Init(const std::string& model_dir, int thread_num) {
    std::string model_file = model_dir + "/" + QUANT_MODEL_NAME;
    std::ifstream f(model_file.c_str());
    if (!f.good()) {
        model_file = model_dir + "/" + MODEL_NAME;
    }
    
    // 1. Load Vocab
    vocab_ = std::make_unique<Vocab>(model_dir + "/" + TOKEN_PATH);
    if (!vocab_->IsLoaded()) {
        std::cerr << "Error: Failed to load tokens.json" << std::endl;
        exit(-1);
    }
    
    // 2. Init Feature Extractor
    InitFbank();
    
    // Load CMVN
    LoadCmvn(model_dir + "/" + AM_CMVN_NAME);
    
    // 3. Init Inference Engine (ONNX or NPU)
    InitOnnx(model_file, thread_num);
}

void Paraformer::InitFbank() {
    fbank_opts_ = std::make_unique<knf::FbankOptions>();
    fbank_opts_->frame_opts.dither = 0.0f;
    fbank_opts_->frame_opts.snip_edges = false;
    fbank_opts_->frame_opts.samp_freq = 16000.0f;
    fbank_opts_->mel_opts.num_bins = 80;
    fbank_computer_ = std::make_unique<knf::FbankComputer>(*fbank_opts_);
}

void Paraformer::InitOnnx(const std::string& model_file, int thread_num) {
    // ======================================================================
    // TODO: NPU Porting Point - Initialization
    // Replace the following ONNX Runtime code with NPU model loading code
    // ======================================================================
    
    try {
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Paraformer");
        session_options_ = std::make_shared<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(thread_num);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session_ = std::make_shared<Ort::Session>(*env_, model_file.c_str(), *session_options_);
        std::cout << "Successfully loaded model: " << model_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX session: " << e.what() << std::endl;
        exit(-1);
    }
}

std::string Paraformer::Forward(float* speech_data, int len) {
    // 1. Feature Extraction (Fbank)
    knf::OnlineFbank fbank(*fbank_opts_);
    fbank.AcceptWaveform(16000, speech_data, len);
    fbank.InputFinished();
    
    int32_t num_frames = fbank.NumFramesReady();
    int32_t dim = fbank_opts_->mel_opts.num_bins;
    
    std::vector<float> feats;
    feats.reserve(num_frames * dim);
    for (int i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        for (int j = 0; j < dim; ++j) {
            feats.push_back(frame[j]);
        }
    }
    
    // 2. LFR (Low Frame Rate) - Paraformer specific
    // Ref: runtime/onnxruntime/src/paraformer.cpp LfrCmvn
    // Input: [T, 80] -> Output: [ceil(T/lfr_n), 560] (concat 7 frames, stride 6)
    
    int lfr_m = lfr_m_; // 7
    int lfr_n = lfr_n_; // 6
    
    // Convert flat feats vector to vector of vectors for easier manipulation
    std::vector<std::vector<float>> asr_feats;
    int T = num_frames;
    for(int i=0; i<T; ++i) {
        std::vector<float> frame(feats.begin() + i*dim, feats.begin() + (i+1)*dim);
        asr_feats.push_back(frame);
    }
    
    int T_lfr = std::ceil(1.0 * T / lfr_n);
    
    // Pad frames at start (copy first frame)
    // For lfr_m=7, padding = (7-1)/2 = 3 frames
    for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        asr_feats.insert(asr_feats.begin(), asr_feats[0]);
    }
    
    // Update T after front padding
    T = asr_feats.size();
    
    std::vector<float> lfr_feats; 
    lfr_feats.reserve(T_lfr * lfr_m * dim);
    
    for (int i = 0; i < T_lfr; i++) {
        if (lfr_m <= T - i * lfr_n) {
            // Enough frames remaining
            for (int j = 0; j < lfr_m; j++) {
                int frame_idx = i * lfr_n + j;
                lfr_feats.insert(lfr_feats.end(), asr_feats[frame_idx].begin(), asr_feats[frame_idx].end());
            }
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames (copy last frame)
            int num_padding = lfr_m - (T - i * lfr_n);
            // Copy remaining existing frames
            for (int j = 0; j < (T - i * lfr_n); j++) {
                int frame_idx = i * lfr_n + j;
                lfr_feats.insert(lfr_feats.end(), asr_feats[frame_idx].begin(), asr_feats[frame_idx].end());
            }
            // Pad with last frame
            const auto& last_frame = asr_feats.back();
            for (int j = 0; j < num_padding; j++) {
                lfr_feats.insert(lfr_feats.end(), last_frame.begin(), last_frame.end());
            }
        }
    }

    // Apply CMVN (on LFR features, dim=560)
    ApplyCmvn(lfr_feats, lfr_m * dim);
    
    // 3. Inference
    // ======================================================================
    // TODO: NPU Porting Point - Inference
    // Replace ONNX Run() with NPU inference call
    // Input: lfr_feats [T_lfr, 560]
    // Output: logits [1, T_lfr, VocabSize] (or similar)
    // ======================================================================
    
    std::vector<int64_t> input_shape = {1, T_lfr, lfr_m * dim};
    std::vector<int32_t> speech_lengths = {T_lfr};
    std::vector<int64_t> speech_lengths_shape = {1};

    const char* input_names[] = {"speech", "speech_lengths"};
    const char* output_names[] = {"logits", "token_num"}; // Adjust based on actual model outputs

    // Create OrtTensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, lfr_feats.data(), lfr_feats.size(), input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
        memory_info, speech_lengths.data(), speech_lengths.size(), speech_lengths_shape.data(), speech_lengths_shape.size()));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 2, output_names, 2);

    // 4. Decoding (Greedy Search)
    float* float_data = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape(); // [1, T_out, VocabSize]
    
    int out_t = shape[1];
    int vocab_size = shape[2];
    
    std::vector<int> token_ids;
    for (int t = 0; t < out_t; ++t) {
        float max_val = -1e10;
        int max_id = -1;
        for (int v = 0; v < vocab_size; ++v) {
            float val = float_data[t * vocab_size + v];
            if (val > max_val) {
                max_val = val;
                max_id = v;
            }
        }
        if (max_id > 2) { // Skip <blank>(0), <s>(1), </s>(2)
             token_ids.push_back(max_id);
        }
    }
    
    return vocab_->Vector2String(token_ids);
}

} // namespace funasr
