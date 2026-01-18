#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "funasrruntime.h"
#include "com-define.h"
#include "audio.h"

/**
 * Usage Example:
 * 
 * ./simple-asr-demo-2pass \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
 *   /workspace/Python/FunASR/runtime/onnxruntime/demo_assets/asr_example.wav
 */

// Usage: ./simple_asr_demo_2pass <offline_model_dir> <online_model_dir> <vad_model_dir> <punc_model_dir> <wav_path>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <offline_model_dir> <online_model_dir> <vad_model_dir> <punc_model_dir> <wav_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " path/to/offline_model path/to/online_model path/to/vad_model path/to/punc_model path/to/audio.wav" << std::endl;
        return 1;
    }

    std::string offline_model_dir = argv[1];
    std::string online_model_dir = argv[2];
    std::string vad_model_dir = argv[3];
    std::string punc_model_dir = argv[4];
    std::string wav_path = argv[5];

    // 1. Config
    std::map<std::string, std::string> model_path;
    model_path[OFFLINE_MODEL_DIR] = offline_model_dir;
    model_path[ONLINE_MODEL_DIR] = online_model_dir;
    model_path[WAV_PATH] = wav_path;
    model_path[QUANTIZE] = "true";
    model_path[ASR_MODE] = "2pass";
    model_path[VAD_DIR] = vad_model_dir;
    model_path[VAD_QUANT] = "true";
    model_path[PUNC_DIR] = punc_model_dir;
    model_path[PUNC_QUANT] = "true";
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Initializing 2pass..." << std::endl;
    std::cout << "Offline Model: " << offline_model_dir << std::endl;
    std::cout << "Online Model:  " << online_model_dir << std::endl;
    std::cout << "VAD Model:     " << vad_model_dir << std::endl;
    std::cout << "Punc Model:    " << punc_model_dir << std::endl;
    
    // Init with 1 thread
    FUNASR_HANDLE tpass_handle = FunTpassInit(model_path, 1);
    if (!tpass_handle) {
        std::cerr << "Error: FunTpassInit failed" << std::endl;
        return -1;
    }

    // 2. Online Init
    // chunk_size = [5, 10, 5] corresponds to {5*60ms, 10*60ms, 5*60ms} ?
    // The exact meaning depends on model config (lfr_m, lfr_n).
    // For Paraformer-Online, [5, 10, 5] is standard.
    std::vector<int> chunk_size = {5, 10, 5}; 
    FUNASR_HANDLE online_handle = FunTpassOnlineInit(tpass_handle, chunk_size);
    if (!online_handle) {
        std::cerr << "Error: FunTpassOnlineInit failed" << std::endl;
        FunTpassUninit(tpass_handle);
        return -1;
    }
    
    // Decoder handle (needed for 2pass API)
    // Using default beam values: global_beam=3.0, lattice_beam=3.0, am_scale=10.0
    FUNASR_DEC_HANDLE decoder_handle = FunASRWfstDecoderInit(tpass_handle, ASR_TWO_PASS, 3.0f, 3.0f, 10.0f);

    std::cout << "Initialization successful." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 3. Load Audio
    funasr::Audio audio(1); // 1=int16
    int32_t sampling_rate = 16000;
    // LoadWav2Char loads audio into char* buffer (raw bytes)
    if (!audio.LoadWav2Char(wav_path.c_str(), &sampling_rate)) {
        std::cerr << "Error: Failed to load wav: " << wav_path << std::endl;
        FunASRWfstDecoderUninit(decoder_handle);
        FunTpassOnlineUninit(online_handle);
        FunTpassUninit(tpass_handle);
        return -1;
    }
    std::cout << "Loaded Wav: " << wav_path << std::endl;
    std::cout << "SampleRate: " << sampling_rate << std::endl;
    std::cout << "Samples:    " << audio.GetSpeechLen() << std::endl;
    
    char* speech_buff = audio.GetSpeechChar();
    int buff_len = audio.GetSpeechLen() * 2; // samples * 2 bytes/sample

    // 4. Inference Loop
    std::cout << "Starting inference..." << std::endl;
    
    // step = 800 samples * 2 bytes = 1600 bytes
    // 800 samples @ 16kHz = 50ms
    // Let's try larger step: 160ms = 2560 samples = 5120 bytes
    int step = 2560 * 2; 
    
    bool is_final = false;
    std::vector<std::vector<std::string>> punc_cache(2); // Punctuation cache
    std::vector<std::vector<float>> hotwords_embedding;  // Empty hotwords

    for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
        if (sample_offset + step >= buff_len - 1) {
            step = buff_len - sample_offset;
            is_final = true;
        } else {
            is_final = false;
        }
        
        // Debug: Print progress
        // if (sample_offset % (step * 20) == 0) std::cout << "." << std::flush;

        // FunTpassInferBuffer processes a chunk of audio
        FUNASR_RESULT result = FunTpassInferBuffer(
            tpass_handle, 
            online_handle,
            speech_buff + sample_offset,
            step,
            punc_cache,
            is_final,
            sampling_rate,
            "pcm",
            ASR_TWO_PASS,
            hotwords_embedding,
            true, // enable itn
            decoder_handle
        );
        
        if (result) {
            // Get Online Result (Streaming)
            std::string online_res = FunASRGetResult(result, 0);
            if (!online_res.empty()) {
                std::cout << "Online Partial: " << online_res << std::endl;
            }
            
            // Get 2pass Result (Offline correction at the end of sentence)
            std::string tpass_res = FunASRGetTpassResult(result, 0);
             if (!tpass_res.empty()) {
                std::cout << "2pass Result:   " << tpass_res << std::endl;
            }
            
            FunASRFreeResult(result);
        }
    }
    std::cout << std::endl; // Newline after progress dots

    // 5. Cleanup
    FunASRWfstDecoderUninit(decoder_handle);
    FunTpassOnlineUninit(online_handle);
    FunTpassUninit(tpass_handle);
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
}
