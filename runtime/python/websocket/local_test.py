from funasr import AutoModel
import argparse
import time

def main():
    # 1. 定义参数（和你服务端用的模型一致）
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/workspace/models/longaudio.wav", help="Path to the input wav file")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    print(f"Loading models on {args.device}...")
    start_load = time.time()

    # 2. 直接加载 AutoModel
    # FunASR 的 AutoModel 会自动组合 VAD(端点检测) + ASR(识别) + PUNC(标点)
    # 只需要定义这就行，不需要手动处理切片！
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        # spk_model="iic/speech_campplus_sv_zh-cn_16k-common", # 如果需要区分说话人，可以加上这个
        device=args.device,
        disable_log=True, 
    )
    
    print(f"Models loaded in {time.time() - start_load:.2f}s")

    # 3. 运行推理
    print(f"Processing {args.input} ...")
    start_infer = time.time()
    
    # generate 函数内部会自动处理：
    # - 读取音频
    # - VAD 切分长音频（自动处理内存，不会爆显存）
    # - 批量送入 GPU 加速
    # - 加标点
    res = model.generate(input=args.input, batch_size_s=1) # batch_size_s 控制批处理大小，越大越快（吃显存）
    
    end_infer = time.time()
    
    # 4. 打印结果
    print("-" * 50)
    # res 是一个列表，通常只有一个结果
    text = res[0]['text']
    print(f"Result: {text[:200]}... (Total {len(text)} chars)")
    print("-" * 50)
    print(f"Inference time: {end_infer - start_infer:.2f}s")
    
    # 保存结果
    output_file = args.input + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
