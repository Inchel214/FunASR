import os
import shutil

src_base = "/workspace/Python/FunASR/runtime/onnxruntime/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc"
dst_base = "/workspace/Python/FunASR/runtime/simple_funasr_standalone/src/kaldi-native-fbank"

# We only need the core feature extraction files
files_to_copy = [
    "feature-fbank.h", "feature-fbank.cc",
    "feature-functions.h", "feature-functions.cc",
    "feature-window.h", "feature-window.cc",
    "mel-computations.h", "mel-computations.cc",
    "online-feature.h", "online-feature.cc",
    "rfft.h", "rfft.cc",
    "log.h" # Need to patch this
]

if not os.path.exists(dst_base):
    os.makedirs(dst_base)

for f in files_to_copy:
    src_path = os.path.join(src_base, f)
    dst_path = os.path.join(dst_base, f)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied {f}")
    else:
        print(f"Warning: {f} not found")
