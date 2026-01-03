import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
import traceback


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="grpc server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    help="model from modelscope",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="model from modelscope",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--certfile",
    type=str,
    default="../../ssl_key/server.crt",
    required=False,
    help="certfile for ssl",
)

parser.add_argument(
    "--keyfile",
    type=str,
    default="../../ssl_key/server.key",
    required=False,
    help="keyfile for ssl",
)
args = parser.parse_args()


websocket_users = set()

print("model loading")
from funasr import AutoModel

# asr
model_asr = AutoModel(
    model=args.asr_model,
    model_revision=args.asr_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
# asr
model_asr_streaming = AutoModel(
    model=args.asr_model_online,
    model_revision=args.asr_model_online_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
# vad
print("DEBUG: Loading VAD model...", flush=True)
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    # chunk_size=60,
)
print(f"DEBUG: VAD model loaded on {args.device}", flush=True)

if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )
else:
    model_punc = None


print("model loaded! only support one client at the same time now!!!!")


async def ws_reset(websocket):
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path=None):
    frames = []
    frames_asr = []
    frames_asr_online = []
    global websocket_users
    # await clear_websocket()
    websocket_users.add(websocket)
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    websocket.total_frames_received = 0  # <--- 新增计数器
    print("new user connected", flush=True)

    try:
        print("DEBUG: Entering receive loop", flush=True)
        async for message in websocket:
            try:
                # print(f"DEBUG: Received message type: {type(message)}", flush=True)
                if isinstance(message, str):
                    print(f"DEBUG: Received text message: {message[:100]}", flush=True)
                    messagejson = json.loads(message)

                    if "is_speaking" in messagejson:
                        websocket.is_speaking = messagejson["is_speaking"]
                        websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                    if "chunk_interval" in messagejson:
                        websocket.chunk_interval = messagejson["chunk_interval"]
                    if "wav_name" in messagejson:
                        websocket.wav_name = messagejson.get("wav_name")
                    if "chunk_size" in messagejson:
                        chunk_size = messagejson["chunk_size"]
                        if isinstance(chunk_size, str):
                            chunk_size = chunk_size.split(",")
                        websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
                    if "encoder_chunk_look_back" in messagejson:
                        websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson[
                            "encoder_chunk_look_back"
                        ]
                    if "decoder_chunk_look_back" in messagejson:
                        websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson[
                            "decoder_chunk_look_back"
                        ]
                    if "hotwords" in messagejson:
                        websocket.status_dict_asr["hotword"] = messagejson["hotwords"]
                    if "mode" in messagejson:
                        websocket.mode = messagejson["mode"]
                    
                    print("DEBUG: Processed text message", flush=True)

                websocket.status_dict_vad["chunk_size"] = int(
                    websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
                )
                if len(frames_asr_online) > 0 or len(frames_asr) >= 0 or not isinstance(message, str):
                    if not isinstance(message, str):
                        websocket.total_frames_received += 1
                        if websocket.total_frames_received < 5 or websocket.total_frames_received % 100 == 0:
                            print(f"DEBUG: Processing binary frame {websocket.total_frames_received}", flush=True)
                        
                        frames.append(message)
                        duration_ms = len(message) // 32
                        websocket.vad_pre_idx += duration_ms

                        # asr online
                        frames_asr_online.append(message)
                        websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                        
                        # print(f"DEBUG: Frame {websocket.total_frames_received} - Check ASR Online. Len: {len(frames_asr_online)}, Interval: {websocket.chunk_interval}", flush=True)
                        if (
                            len(frames_asr_online) % websocket.chunk_interval == 0
                            or websocket.status_dict_asr_online["is_final"]
                        ):
                            if websocket.mode == "2pass" or websocket.mode == "online":
                                audio_in = b"".join(frames_asr_online)
                                try:
                                    print(f"DEBUG: Calling async_asr_online, len={len(frames_asr_online)}", flush=True)
                                    await async_asr_online(websocket, audio_in)
                                    print("DEBUG: Finished async_asr_online", flush=True)
                                except Exception as e:
                                    print(f"error in asr streaming: {e}, {websocket.status_dict_asr_online}", flush=True)
                                    traceback.print_exc()
                            frames_asr_online = []
                        
                        if speech_start or websocket.mode == "offline":
                            frames_asr.append(message)
                        
                        # vad online
                        try:
                            # print("DEBUG: Calling async_vad", flush=True)
                            if websocket.mode == "offline":
                                speech_start_i, speech_end_i = -1, -1
                            else:
                                if websocket.total_frames_received < 5 or websocket.total_frames_received % 100 == 0:
                                    print(f"DEBUG: Calling async_vad, total: {websocket.total_frames_received}", flush=True)
                                
                                speech_start_i, speech_end_i = await async_vad(websocket, message)
                                
                                if websocket.total_frames_received < 5 or websocket.total_frames_received % 100 == 0:
                                    print(f"DEBUG: Finished async_vad, total: {websocket.total_frames_received}", flush=True)
                            
                            if speech_start_i != -1:
                                print(f"DEBUG: VAD speech start detected at {speech_start_i}", flush=True)
                                speech_start = True
                                beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                                frames_pre = frames[-beg_bias:]
                                frames_asr = []
                                frames_asr.extend(frames_pre)
                        except Exception as e:
                            print(f"error in vad: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                        
                        # print(f"DEBUG: Frame {websocket.total_frames_received} - Finished processing", flush=True)
                    
                    # asr punc offline
                    if speech_end_i != -1 or not websocket.is_speaking:
                        # print("vad end point")
                        if websocket.mode == "2pass" or websocket.mode == "offline":
                            audio_in = b"".join(frames_asr)
                            # Reset frames_asr to avoid processing the same audio again if loop continues
                            frames_asr = [] 
                            try:
                                print(f"DEBUG: Calling async_asr offline, audio len: {len(audio_in)}", flush=True)
                                
                                # DEBUG: Save received audio to file for inspection
                                # with open("debug_received_audio.pcm", "wb") as f:
                                #     f.write(audio_in)
                                # print("DEBUG: Saved received audio to debug_received_audio.pcm", flush=True)

                                # Convert bytes to numpy array (int16 -> float32)
                                audio_array = np.frombuffer(audio_in, dtype=np.int16)
                                audio_array = audio_array.astype(np.float32)
                                
                                print(f"DEBUG: Converted audio to numpy array, shape={audio_array.shape}, dtype={audio_array.dtype}", flush=True)
                                
                                # Use VAD for segmentation (Industry Standard Approach)
                                full_text = ""
                                try:
                                    print("DEBUG: Starting VAD for segmentation...", flush=True)
                                    
                                    # Fix VAD hanging on long audio: Process VAD in chunks of 60 seconds
                                    # Use independent chunks to prevent cache accumulation (which caused the hang/OOM)
                                    
                                    VAD_CHUNK_S = 60 # 1 minute
                                    VAD_CHUNK_SAMPLES = VAD_CHUNK_S * 16000
                                    total_samples = len(audio_array)
                                    all_segments = []
                                    
                                    for v_start in range(0, total_samples, VAD_CHUNK_SAMPLES):
                                        v_end = min(v_start + VAD_CHUNK_SAMPLES, total_samples)
                                        vad_audio_chunk = audio_array[v_start:v_end]
                                        
                                        print(f"DEBUG: Running VAD on chunk {v_start//VAD_CHUNK_SAMPLES + 1}/{(total_samples+VAD_CHUNK_SAMPLES-1)//VAD_CHUNK_SAMPLES}, len={len(vad_audio_chunk)/16000:.2f}s", flush=True)
                                        
                                        try:
                                            # Run VAD on this 60s chunk independently
                                            # We use a fresh cache for each chunk to avoid memory accumulation
                                            chunk_vad_params = {"cache": {}, "is_final": True}
                                            vad_res = model_vad.generate(input=[vad_audio_chunk], **chunk_vad_params)[0]
                                            chunk_segments = vad_res.get("value", [])
                                            
                                            # Adjust timestamps: add offset (v_start in ms)
                                            offset_ms = v_start / 16.0
                                            for seg in chunk_segments:
                                                # seg is [start_ms, end_ms] relative to chunk
                                                abs_start = seg[0] + offset_ms
                                                abs_end = seg[1] + offset_ms
                                                all_segments.append([abs_start, abs_end])
                                                
                                        except Exception as e:
                                            print(f"ERROR in VAD chunk {v_start}: {e}", flush=True)
                                            traceback.print_exc()
                                            # Fallback: treat this chunk as one big segment
                                            offset_ms = v_start / 16.0
                                            duration_ms = (v_end - v_start) / 16.0
                                            all_segments.append([offset_ms, offset_ms + duration_ms])
                                    
                                    segments = all_segments
                                    print(f"DEBUG: VAD finished. Found {len(segments)} total segments.", flush=True)
                                    
                                    if len(segments) == 0:
                                        print("DEBUG: No speech segments found by VAD. Trying fallback to full audio.", flush=True)
                                        segments = [[0, len(audio_array)/16]] # [start_ms, end_ms]
                                    
                                    for i, segment in enumerate(segments):
                                        start_ms = segment[0]
                                        end_ms = segment[1]
                                        
                                        start_sample = int(start_ms * 16)
                                        end_sample = int(end_ms * 16)
                                        
                                        start_sample = max(0, start_sample)
                                        end_sample = min(len(audio_array), end_sample)
                                        
                                        if end_sample - start_sample < 160: # Skip < 10ms
                                            continue
                                            
                                        chunk_audio = audio_array[start_sample:end_sample]
                                        
                                        MAX_ASR_LEN_S = 60
                                        if len(chunk_audio) > MAX_ASR_LEN_S * 16000:
                                            print(f"WARNING: Segment {i} is too long ({len(chunk_audio)/16000:.2f}s). Forcing split.", flush=True)
                                            sub_len = MAX_ASR_LEN_S * 16000
                                            for j in range(0, len(chunk_audio), sub_len):
                                                sub_chunk = chunk_audio[j:j+sub_len]
                                                rec_result_chunk = model_asr.generate(input=[sub_chunk], **websocket.status_dict_asr)[0]
                                                full_text += rec_result_chunk.get("text", "")
                                        else:
                                            rec_result_chunk = model_asr.generate(input=[chunk_audio], **websocket.status_dict_asr)[0]
                                            full_text += rec_result_chunk.get("text", "")
                                            
                                        if i < 5 or i % 100 == 0:
                                            print(f"DEBUG: Processed segment {i}/{len(segments)}: {start_ms}-{end_ms}ms", flush=True)
                                            
                                except Exception as e:
                                    print(f"ERROR in VAD/ASR loop: {e}", flush=True)
                                    import traceback
                                    traceback.print_exc()
                                
                                print(f"DEBUG: Finished processing all chunks. Total text len: {len(full_text)}", flush=True)

                                rec_result = {"text": full_text}
                                print("offline_asr, ", rec_result)
                                
                                if model_punc is not None and len(rec_result["text"]) > 0:
                                    try:
                                        rec_result = model_punc.generate(
                                            input=rec_result["text"], **websocket.status_dict_punc
                                        )[0]
                                    except Exception as e:
                                        print(f"ERROR in punctuation: {e}", flush=True)
                                
                                if len(rec_result["text"]) > 0:
                                    mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
                                    message = json.dumps(
                                        {
                                            "mode": mode,
                                            "text": rec_result["text"],
                                            "wav_name": websocket.wav_name,
                                            "is_final": True,
                                        }
                                    )
                                    await websocket.send(message)
                                else:
                                    mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
                                    message = json.dumps(
                                        {
                                            "mode": mode,
                                            "text": "",
                                            "wav_name": websocket.wav_name,
                                            "is_final": True,
                                        }
                                    )
                                    await websocket.send(message)
                            except Exception as e:
                                print(f"error in asr offline: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        frames_asr = []
                        speech_start = False
                        frames_asr_online = []
                        websocket.status_dict_asr_online["cache"] = {}
                        if not websocket.is_speaking:
                            websocket.vad_pre_idx = 0
                            frames = []
                            websocket.status_dict_vad["cache"] = {}
                        else:
                            frames = frames[-20:]
            except Exception as e:
                print(f"ERROR inside loop processing: {e}", flush=True)
                traceback.print_exc()

    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users, flush=True)
        await ws_reset(websocket)
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)


async def async_vad(websocket, audio_in):

    # print("DEBUG: Inside async_vad, generating...", flush=True)
    segments_result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)[0]["value"]
    # print(segments_result)

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, audio_in):
    if len(audio_in) > 0:
        # print(len(audio_in))
        rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
        print(f"DEBUG: Full rec_result: {rec_result}", flush=True)
        print("offline_asr, ", rec_result)
        if model_punc is not None and len(rec_result["text"]) > 0:
            # print("offline, before punc", rec_result, "cache", websocket.status_dict_punc)
            rec_result = model_punc.generate(
                input=rec_result["text"], **websocket.status_dict_punc
            )[0]
            # print("offline, after punc", rec_result)
        if len(rec_result["text"]) > 0:
            print("offline", rec_result)
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": True,
                }
            )
            await websocket.send(message)

    else:
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": websocket.wav_name,
                "is_final": True,
            }
        )
        await websocket.send(message)    

async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        # print(websocket.status_dict_asr_online.get("is_final", False))
        rec_result = model_asr_streaming.generate(
            input=audio_in, **websocket.status_dict_asr_online
        )[0]
        # print("online, ", rec_result)
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return
            #     websocket.status_dict_asr_online["cache"] = dict()
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)


async def main():
    if len(args.certfile) > 0:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Generate with Lets Encrypt, copied to this location, chown to current user and 400 permissions
        ssl_cert = args.certfile
        ssl_key = args.keyfile

        ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
        async with websockets.serve(
            ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ):
            await asyncio.get_running_loop().create_future()
    else:
        async with websockets.serve(
            ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None
        ):
            await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())
