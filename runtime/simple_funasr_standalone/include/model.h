#pragma once
#include "precomp.h"
#include "vocab.h"
#include "feature-fbank.h"
#include "online-feature.h"

namespace funasr {

class Model {
public:
    virtual ~Model() {};
    virtual void Init(const std::string& model_dir, int thread_num) = 0;
    virtual std::string Forward(float* speech_data, int len) = 0;
};

class Paraformer : public Model {
public:
    Paraformer();
    ~Paraformer();
    
    void Init(const std::string& model_dir, int thread_num) override;
    std::string Forward(float* speech_data, int len) override;

private:
    void InitOnnx(const std::string& model_file, int thread_num);
    void InitFbank();

    // ONNX Runtime objects
    // TODO: When porting to NPU, replace these with NPU engine handles
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::shared_ptr<Ort::SessionOptions> session_options_;
    
    // Feature extraction
    std::unique_ptr<knf::FbankComputer> fbank_computer_;
    std::unique_ptr<knf::FbankOptions> fbank_opts_;

    // Vocab
    std::unique_ptr<Vocab> vocab_;
    
    // Model metadata
    int64_t lfr_m_ = 7;
    int64_t lfr_n_ = 6;
    
    // CMVN stats
    std::vector<float> means_;
    std::vector<float> vars_;
    void LoadCmvn(const std::string& mvn_file);
    void ApplyCmvn(std::vector<float>& feats, int dim);
};

} // namespace funasr
