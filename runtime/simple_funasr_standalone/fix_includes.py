import os

src_dir = "/workspace/Python/FunASR/runtime/simple_funasr_standalone/src/kaldi-native-fbank"

for filename in os.listdir(src_dir):
    if not (filename.endswith(".cc") or filename.endswith(".h")):
        continue
        
    path = os.path.join(src_dir, filename)
    with open(path, "r") as f:
        content = f.read()
    
    # Replace include paths
    # "kaldi-native-fbank/csrc/xxx.h" -> "xxx.h"
    # <kaldi-native-fbank/csrc/xxx.h> -> "xxx.h"
    
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '#include "kaldi-native-fbank/csrc/' in line:
            line = line.replace('kaldi-native-fbank/csrc/', '')
        elif '#include <kaldi-native-fbank/csrc/' in line:
            line = line.replace('<kaldi-native-fbank/csrc/', '"').replace('>', '"')
        new_lines.append(line)
    
    with open(path, "w") as f:
        f.write('\n'.join(new_lines))
        
print("Fixed include paths.")
