#pragma once
#include "precomp.h"

namespace funasr {

struct WaveHeader {
    uint32_t chunk_id;      // "RIFF"
    uint32_t chunk_size;
    uint32_t format;        // "WAVE"
    
    uint32_t subchunk1_id;  // "fmt "
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    
    uint32_t subchunk2_id;  // "data"
    uint32_t subchunk2_size;

    bool Validate() const {
        return chunk_id == 0x46464952 && format == 0x45564157;
    }
};

class Audio {
public:
    Audio(int data_type = 1);
    ~Audio();
    bool LoadWav(const char* filename, int32_t* sampling_rate, bool resample = false);
    float* GetSpeechData();
    int32_t GetSpeechLen();

private:
    float* speech_data_ = nullptr;
    int16_t* speech_buff_ = nullptr;
    int32_t speech_len_ = 0;
    int data_type_ = 1; // 1: int16, 2: float
};

} // namespace funasr
