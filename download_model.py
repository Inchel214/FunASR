from modelscope.hub.snapshot_download import snapshot_download
import os

model_dir = "runtime/simple_funasr_standalone/model"
os.makedirs(model_dir, exist_ok=True)

# Download ONNX offline model for Paraformer-large
model_id_offline = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
try:
    path = snapshot_download(model_id_offline, cache_dir=model_dir)
    print(f"Offline Model downloaded to: {path}")
except Exception as e:
    print(f"Error downloading offline model: {e}")

# Download ONNX online model for Paraformer-large
model_id_online = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
try:
    path = snapshot_download(model_id_online, cache_dir=model_dir)
    print(f"Online Model downloaded to: {path}")
except Exception as e:
    print(f"Error downloading online model: {e}")

# Download VAD model
model_id_vad = "damo/speech_fsmn_vad_zh-cn-16k-common-onnx"
try:
    path = snapshot_download(model_id_vad, cache_dir=model_dir)
    print(f"VAD Model downloaded to: {path}")
except Exception as e:
    print(f"Error downloading VAD model: {e}")

# Download Punc model
model_id_punc = "damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
try:
    path = snapshot_download(model_id_punc, cache_dir=model_dir)
    print(f"Punc Model downloaded to: {path}")
except Exception as e:
    print(f"Error downloading Punc model: {e}")
