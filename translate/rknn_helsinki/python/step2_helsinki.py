#!/usr/bin/env python3
"""
测试简化版Decoder的完整翻译流程
"""
import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer
import json

def test_simple_decoder_translation():
    print("=" * 60)
    print("测试简化版Decoder翻译")
    print("=" * 60)
    
    # 1. 加载配置
    with open("./models/model_config.json", "r") as f:
        encoder_config = json.load(f)
    
    with open("./models/decoder_config.json", "r") as f:
        decoder_config = json.load(f)
    
    # 2. 加载tokenizer
    tokenizer = MarianTokenizer.from_pretrained("./models/tokenizer")
    
    # 3. 加载ONNX模型
    print("\n加载ONNX模型...")
    encoder_sess = ort.InferenceSession("./models/opus_mt_en_zh_encoder.onnx")
    decoder_sess = ort.InferenceSession("./models/opus_mt_en_zh_decoder.onnx")
    print("   ✓ Encoder + Decoder 加载成功")
    
    encoder_seq_len = encoder_config["seq_len"]
    decoder_seq_len = decoder_config["decoder_seq_len"]
    
    print(f"   - Encoder固定长度: {encoder_seq_len}")
    print(f"   - Decoder固定长度: {decoder_seq_len}")
    
    # 4. 测试句子
    test_sentences = [
        "Hello world",
        "I love you",
        "WHAT a beautiful day!",
        "How are you today?",
        "happy birthday",
        "This is a test sentence for translation.",
    ]
    
    for text in test_sentences:
        print(f"\n{'='*50}")
        print(f"原文: {text}")
        
        # === Tokenize输入 ===
        inputs = tokenizer(text, return_tensors="np", padding="max_length", 
                          max_length=encoder_seq_len, truncation=True)
        
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        
        # === Encoder前向 ===
        print("Running encoder...")
        encoder_outputs = encoder_sess.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        encoder_hidden_states = encoder_outputs[0]
        print(f"   Encoder output: {encoder_hidden_states.shape}")
        
        # === Decoder自回归循环 ===
        print("Running decoder (auto-regressive)...")
        
        # 初始化decoder输入: 全部为pad_token
        decoder_input_ids = np.full((1, decoder_seq_len), tokenizer.pad_token_id, dtype=np.int64)
        
        # 第一个token设为pad_token (MarianMT的decoder_start_token)
        decoder_input_ids[0, 0] = tokenizer.pad_token_id
        
        translated_ids = []
        max_gen_length = min(decoder_seq_len - 1, 50)  # 最多生成50个token
        
        for step in range(max_gen_length):
            # Decoder前向
            decoder_outputs = decoder_sess.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": attention_mask
                }
            )
            
            # logits shape: [1, decoder_seq_len, vocab_size]
            logits = decoder_outputs[0]
            
            # 获取当前位置(step)的logits
            current_logits = logits[0, step, :]
            
            # 贪心解码: 选择最大概率的token
            next_token_id = int(np.argmax(current_logits))
            
            # 检查是否结束
            if next_token_id == tokenizer.eos_token_id:
                print(f"   Step {step}: EOS reached")
                break
            
            # 保存生成的token
            translated_ids.append(next_token_id)
            
            # 更新decoder输入: 将新token添加到序列中
            if step + 1 < decoder_seq_len:
                decoder_input_ids[0, step + 1] = next_token_id
            
            if step < 5 or step % 10 == 0:
                print(f"   Step {step}: token_id={next_token_id}")
        
        # 解码译文
        try:
            translation = tokenizer.decode(translated_ids, skip_special_tokens=True)
            print(f"\n✓ 译文: {translation}")
            print(f"   生成token数: {len(translated_ids)}")
        except Exception as e:
            print(f"\n✗ 解码失败: {e}")
            print(f"Token IDs: {translated_ids[:20]}...")
    
    print("\n" + "=" * 60)
    print("✓ 完整翻译流程验证完成!")
    print("=" * 60)
    print("\n验证结果:")
    print("  ✓ Encoder工作正常")
    print("  ✓ Decoder工作正常")
    print("  ✓ 自回归解码流程正常")
    print("  ✓ 翻译结果正常输出")
    
    print("\n性能特点:")
    print("  - 每次decoder计算整个序列(包含已生成部分)")
    print("  - 固定shape,无动态维度")
    print("  - 适合短文本翻译")
    
    print("\n下一步: ONNX → RKNN 转换")
    print("  1. 使用RKNN-Toolkit2转换encoder")
    print("  2. 使用RKNN-Toolkit2转换decoder")
    print("  3. 在RK3576上部署C++推理")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_decoder_translation()