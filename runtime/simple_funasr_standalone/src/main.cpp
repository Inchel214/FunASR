#include <iostream>
#include <chrono>
#include "model.h"
#include "audio.h"

using namespace funasr;

/**
 * Usage Example:
 * 
 * ./funasr-standalone \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
 *   /workspace/Python/FunASR/runtime/onnxruntime/demo_assets/asr_example.wav
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <wav_path>" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string wav_path = argv[2];

    // 1. Load Audio
    Audio audio(1); // 1=int16 scale
    int32_t sampling_rate;
    // Pass resample=true to enable internal resampling
    if (!audio.LoadWav(wav_path.c_str(), &sampling_rate, true)) {
        return -1;
    }
    std::cout << "Loaded audio: " << wav_path << ", fs=" << sampling_rate << std::endl;
    
    float audio_duration = (float)audio.GetSpeechLen() / sampling_rate;

    // 2. Init Model
    std::cout << "Loading model from: " << model_dir << std::endl;
    Paraformer model;
    model.Init(model_dir, 1); // 1 thread

    // 3. Inference
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::string text = model.Forward(audio.GetSpeechData(), audio.GetSpeechLen());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float rtf = (float)duration_ms / 1000.0f / audio_duration;

    // 4. Result
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Result: " << text << std::endl;
    std::cout << "Audio Duration: " << audio_duration << " s" << std::endl;
    std::cout << "Inference Time: " << duration_ms << " ms" << std::endl;
    std::cout << "RTF: " << rtf << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
