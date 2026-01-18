#pragma once 

namespace funasr {
#define S_BEGIN  0
#define S_MIDDLE 1
#define S_END    2
#define S_ALL    3
#define S_ERR    4

#ifndef MODEL_SAMPLE_RATE
#define MODEL_SAMPLE_RATE 16000
#endif

// Minimal set of constants needed for Paraformer Offline
#define MODEL_DIR "model-dir"
#define QUANTIZE "quantize"
#define WAV_PATH "wav-path"

#define MODEL_NAME "model.onnx"
#define QUANT_MODEL_NAME "model_quant.onnx"
#define AM_CMVN_NAME "am.mvn"
#define AM_CONFIG_NAME "config.yaml"
#define TOKEN_PATH "tokens.json"

// asr
#ifndef PARA_LFR_M
#define PARA_LFR_M 7
#endif

#ifndef PARA_LFR_N
#define PARA_LFR_N 6
#endif

} // namespace funasr
