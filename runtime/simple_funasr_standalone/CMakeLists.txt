cmake_minimum_required(VERSION 3.10)
project(simple_funasr_standalone)

set(CMAKE_CXX_STANDARD 14)

# 1. ONNX Runtime (Find or set manually)
# You can set ONNXRUNTIME_DIR env var or pass -DONNXRUNTIME_DIR=...
if (NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "/workspace/Python/FunASR/runtime/onnxruntime/tools/onnxruntime-linux-x64-1.14.0")
endif()
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 2. Source Files
include_directories(include)
include_directories(src/kaldi-native-fbank)

file(GLOB_RECURSE SRC_FILES 
    "src/*.cpp" 
    "src/kaldi-native-fbank/*.cc"
    "src/kaldi-native-fbank/*.c"
)

# 3. Executable
add_executable(funasr-standalone ${SRC_FILES})

# 4. Link Libraries
# Need to link pthread and dl for onnxruntime
target_link_libraries(funasr-standalone 
    onnxruntime 
    pthread 
    dl
)

# Set rpath to find libonnxruntime.so at runtime
set_target_properties(funasr-standalone PROPERTIES 
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)

message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
