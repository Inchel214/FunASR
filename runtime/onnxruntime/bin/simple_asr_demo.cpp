#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "funasrruntime.h"
#include "com-define.h"
#include "audio.h"

/**
 * Progress callback function
 * This is required by the API but we can leave it empty or print progress
 */
void ProgressCallback(int cur_step, int n_total) {
    // Optional: Print progress
}

/**
 * Usage Example:
 * 
 * ./simple-asr-demo \
 *   /workspace/Python/FunASR/runtime/simple_funasr_standalone/model/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
 *   /workspace/Python/FunASR/runtime/onnxruntime/demo_assets/asr_example.wav
 */

// Usage: ./simple_asr_demo <model_dir> <wav_path>

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <wav_path> [vad_dir] [punc_dir]" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/model /path/to/audio.wav" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string wav_path = argv[2];
    std::string vad_dir = (argc > 3) ? argv[3] : "";
    std::string punc_dir = (argc > 4) ? argv[4] : "";

    // 1. Prepare configuration map
    std::map<std::string, std::string> model_path;
    model_path[MODEL_DIR] = model_dir;
    model_path[WAV_PATH] = wav_path;
    model_path[QUANTIZE] = "true"; // Default to quantized model
    
    // Add VAD if provided
    if (!vad_dir.empty()) {
        model_path[VAD_DIR] = vad_dir;
        model_path[VAD_QUANT] = "true";
        std::cout << "VAD Enabled: " << vad_dir << std::endl;
    }

    // Add Punctuation if provided
    if (!punc_dir.empty()) {
        model_path[PUNC_DIR] = punc_dir;
        model_path[PUNC_QUANT] = "true";
        std::cout << "Punctuation Enabled: " << punc_dir << std::endl;
    }

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Initializing FunASR..." << std::endl;
    std::cout << "Model Dir: " << model_dir << std::endl;
    std::cout << "Wav Path:  " << wav_path << std::endl;
    std::cout << "Device:    CPU" << std::endl;

    // 2. Initialize Model
    // thread_num=1, use_gpu=false, batch_size=1
    // Note: FunOfflineInit loads the model and resources
    FUNASR_HANDLE asr_handle = FunOfflineInit(model_path, 1, false, 1);

    if (!asr_handle) {
        std::cerr << "Error: FunASR init failed!" << std::endl;
        return -1;
    }
    std::cout << "Initialization successful." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 3. Run Inference
    std::cout << "Starting inference..." << std::endl;
    
    // Empty hotword embedding for this demo
    std::vector<std::vector<float>> hw_emb;
    
    // FunOfflineInfer parameters:
    // handle, filename, mode, callback, hotwords, sampling_rate, itn, decoder_handle
    FUNASR_RESULT result = FunOfflineInfer(
        asr_handle, 
        wav_path.c_str(), 
        RASR_NONE, 
        ProgressCallback, 
        hw_emb, 
        16000, // Default sampling rate
        true,  // Enable ITN (Inverse Text Normalization)
        nullptr // No external decoder handle needed
    );

    // 4. Process Result
    if (result) {
        const char* msg = FunASRGetResult(result, 0);
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Result: " << (msg ? msg : "") << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        
        // Free result memory
        FunASRFreeResult(result);
    } else {
        std::cerr << "Inference failed or returned no result." << std::endl;
    }

    // 5. Clean up
    FunOfflineUninit(asr_handle);
    std::cout << "Done." << std::endl;

    return 0;
}
