#include "audio.h"
#include <fstream>
#include "com-define.h"

namespace funasr {

Audio::Audio(int data_type) : data_type_(data_type) {}

Audio::~Audio() {
    if (speech_data_) free(speech_data_);
    if (speech_buff_) free(speech_buff_);
}

float* Audio::GetSpeechData() { return speech_data_; }
int32_t Audio::GetSpeechLen() { return speech_len_; }

bool Audio::LoadWav(const char *filename, int32_t* sampling_rate, bool resample) {
    WaveHeader header;
    if (speech_data_) { free(speech_data_); speech_data_ = nullptr; }
    if (speech_buff_) { free(speech_buff_); speech_buff_ = nullptr; }
    
    std::ifstream is(filename, std::ifstream::binary);
    if(!is){
        std::cerr << "Failed to read " << filename << std::endl;
        return false;
    }

    // 1. Read RIFF header
    is.read(reinterpret_cast<char *>(&header.chunk_id), 12);
    if (header.chunk_id != 0x46464952 || header.format != 0x45564157) {
        std::cerr << "Invalid RIFF/WAVE header" << std::endl;
        return false;
    }

    // 2. Find fmt chunk
    bool fmt_found = false;
    while (is) {
        uint32_t chunk_id, chunk_size;
        is.read(reinterpret_cast<char *>(&chunk_id), 4);
        is.read(reinterpret_cast<char *>(&chunk_size), 4);
        
        if (chunk_id == 0x20746d66) { // 'fmt '
            header.subchunk1_id = chunk_id;
            header.subchunk1_size = chunk_size;
            is.read(reinterpret_cast<char *>(&header.audio_format), 16);
            if (chunk_size > 16) is.seekg(chunk_size - 16, std::ios::cur);
            fmt_found = true;
            break;
        } else {
            is.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!fmt_found) return false;

    // 3. Find data chunk
    header.subchunk2_id = 0;
    while (is) {
        uint32_t chunk_id, chunk_size;
        is.read(reinterpret_cast<char *>(&chunk_id), 4);
        is.read(reinterpret_cast<char *>(&chunk_size), 4);
        
        if (chunk_id == 0x61746164) { // 'data'
            header.subchunk2_id = chunk_id;
            header.subchunk2_size = chunk_size;
            break;
        } else {
            is.seekg(chunk_size, std::ios::cur);
        }
    }

    if (header.subchunk2_id != 0x61746164) return false;
    
    *sampling_rate = header.sample_rate;
    int num_channels = header.num_channels;
    int total_samples = header.subchunk2_size / 2;
    
    speech_buff_ = (int16_t *)malloc(header.subchunk2_size);
    if (!speech_buff_) return false;
    
    is.read(reinterpret_cast<char *>(speech_buff_), header.subchunk2_size);
    if (!is) return false;

    // Downmix and convert to float
    speech_len_ = total_samples / num_channels;
    speech_data_ = (float*)malloc(sizeof(float) * speech_len_);
    
    float scale = (data_type_ == 1) ? 32768.0f : 1.0f;
    
    for (int32_t i = 0; i < speech_len_; ++i) {
        float sum = 0;
        for (int c = 0; c < num_channels; ++c) {
            sum += speech_buff_[i * num_channels + c];
        }
        speech_data_[i] = (sum / num_channels) / scale;
    }

    // Simple resample check (stub)
    if (resample && *sampling_rate != MODEL_SAMPLE_RATE) {
        // Simple Linear Interpolation Resampler
        int32_t in_rate = *sampling_rate;
        int32_t out_rate = MODEL_SAMPLE_RATE;
        int32_t in_len = speech_len_;
        int32_t out_len = (int64_t)in_len * out_rate / in_rate;
        
        float* out_data = (float*)malloc(sizeof(float) * out_len);
        if (!out_data) return false;
        
        double ratio = (double)in_rate / out_rate;
        for (int i = 0; i < out_len; i++) {
            double src_idx = i * ratio;
            int idx0 = (int)src_idx;
            int idx1 = idx0 + 1;
            if (idx1 >= in_len) idx1 = in_len - 1;
            
            float frac = src_idx - idx0;
            out_data[i] = speech_data_[idx0] * (1.0 - frac) + speech_data_[idx1] * frac;
        }
        
        free(speech_data_);
        speech_data_ = out_data;
        speech_len_ = out_len;
        *sampling_rate = out_rate;
        
        std::cout << "Resampled audio from " << in_rate << " to " << out_rate << std::endl;
    }

    return true;
}

} // namespace funasr
