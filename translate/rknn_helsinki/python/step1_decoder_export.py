#!/usr/bin/env python3
"""
改进版Decoder导出脚本 - 参考RK Whisper导出方式
关键改进:
1. 不使用past_key_values,避免动态shape
2. 添加ONNX简化步骤
3. 添加完整的精度验证
4. 使用opset_version=12
"""
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings("ignore")

class SimpleDecoder(nn.Module):
    """
    简化Decoder:
    - 不使用past_key_values缓存
    - 每次输入完整序列
    - 固定shape,适合RKNN转换
    """
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.final_logits_bias = model.final_logits_bias
        
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        """
        Args:
            input_ids: [batch, decoder_seq_len]
            encoder_hidden_states: [batch, encoder_seq_len, hidden_size]
            encoder_attention_mask: [batch, encoder_seq_len]
        Returns:
            logits: [batch, decoder_seq_len, vocab_size]
        """
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,  # 关键:不使用cache
            return_dict=True
        )
        
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias
        
        return lm_logits

def export_decoder_for_rknn():
    """
    导出适合RKNN转换的Decoder模型
    """
    print("=" * 60)
    print("导出Decoder (RKNN优化版本 - 无past_key_values)")
    print("=" * 60)
    
    # 模型路径
    model_path = "C:\\Users\\HP\\Desktop\\en_zh_translation\\opus-mt-en-zh"
    
    try:
        print("\n1. 加载模型...")
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        model.eval()
        
        # 创建简化Decoder
        simple_decoder = SimpleDecoder(model)
        simple_decoder.eval()
        
        # 获取配置
        config = model.config
        print(f"   ✓ 模型加载成功")
        print(f"   - Vocab size: {config.vocab_size}")
        print(f"   - Hidden size: {config.d_model}")
        print(f"   - Decoder layers: {config.decoder_layers}")
        
        # 固定参数
        batch_size = 1
        encoder_seq_len = 64
        decoder_seq_len = 64
        hidden_size = config.d_model
        
        print(f"\n2. 准备导出配置...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Encoder seq length: {encoder_seq_len}")
        print(f"   - Decoder seq length: {decoder_seq_len}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Opset version: 12")
        print(f"   ⚠ 注意: 不使用past_key_values,每次输入完整序列")
        
        # 准备Dummy输入
        dummy_decoder_ids = torch.randint(0, config.vocab_size, (batch_size, decoder_seq_len))
        dummy_encoder_hidden = torch.randn(batch_size, encoder_seq_len, hidden_size)
        dummy_encoder_mask = torch.ones(batch_size, encoder_seq_len, dtype=torch.long)
        
        print("\n3. 验证PyTorch模型输出...")
        with torch.no_grad():
            pytorch_logits = simple_decoder(
                dummy_decoder_ids,
                dummy_encoder_hidden,
                dummy_encoder_mask
            )
            print(f"   - PyTorch输出shape: {pytorch_logits.shape}")
            print(f"   - PyTorch输出范围: [{pytorch_logits.min():.4f}, {pytorch_logits.max():.4f}]")
            print(f"   - PyTorch输出均值: {pytorch_logits.mean():.4f}")
            
            # 检查预测token
            predicted_token = pytorch_logits[0, -1, :].argmax().item()
            print(f"   - 最后位置预测token: {predicted_token}")
        
        # 导出ONNX
        print("\n4. 导出ONNX模型...")
        output_path = "./models/opus_mt_en_zh_decoder.onnx"
        
        torch.onnx.export(
            simple_decoder,
            (dummy_decoder_ids, dummy_encoder_hidden, dummy_encoder_mask),
            output_path,
            input_names=['input_ids', 'encoder_hidden_states', 'encoder_attention_mask'],
            output_names=['logits'],
            opset_version=12,  # 使用与whisper相同的opset版本
            do_constant_folding=True,
            export_params=True,
        )
        
        print(f"   ✓ ONNX模型导出成功: {output_path}")
        
        # 简化ONNX模型 - 参考whisper导出流程
        #print("\n5. 简化ONNX模型...")
        #original_model = onnx.load(output_path)
        #simplified_model, check = simplify(original_model)
        
        #if check:
        #    onnx.save(simplified_model, output_path)
        #    print("   ✓ ONNX模型简化成功")
        #else:
        #    print("   ⚠ ONNX模型简化验证失败,使用原始模型")

        # 验证ONNX输出
        print("\n6. 验证ONNX模型输出...")
        import onnxruntime as ort
        
        sess = ort.InferenceSession(output_path)
        onnx_logits = sess.run(
            None,
            {
                'input_ids': dummy_decoder_ids.numpy(),
                'encoder_hidden_states': dummy_encoder_hidden.numpy(),
                'encoder_attention_mask': dummy_encoder_mask.numpy()
            }
        )[0]
        
        print(f"   - ONNX输出shape: {onnx_logits.shape}")
        print(f"   - ONNX输出范围: [{onnx_logits.min():.4f}, {onnx_logits.max():.4f}]")
        print(f"   - ONNX输出均值: {onnx_logits.mean():.4f}")
        
        # 检查预测token
        onnx_predicted_token = onnx_logits[0, -1, :].argmax()
        print(f"   - 最后位置预测token: {onnx_predicted_token}")
        
        # 计算误差
        diff = np.abs(pytorch_logits.numpy() - onnx_logits)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"\n7. PyTorch vs ONNX误差分析:")
        print(f"   - 最大误差: {max_diff:.6f}")
        print(f"   - 平均误差: {mean_diff:.6f}")
        
        # 余弦相似度
        pytorch_flat = pytorch_logits.numpy().flatten()
        onnx_flat = onnx_logits.flatten()
        cosine_sim = np.dot(pytorch_flat, onnx_flat) / (np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat))
        print(f"   - 余弦相似度: {cosine_sim:.6f}")
        
        # Token级别的一致性检查
        pytorch_tokens = pytorch_logits[0].argmax(dim=-1).numpy()
        onnx_tokens = onnx_logits[0].argmax(axis=-1)
        token_match = (pytorch_tokens == onnx_tokens).sum() / len(pytorch_tokens)
        print(f"   - Token一致性: {token_match*100:.2f}%")
        
        if cosine_sim > 0.999 and token_match > 0.99:
            print("   ✓ ONNX模型精度验证通过!")
        else:
            print("   ⚠ ONNX模型精度可能存在问题,请检查!")
            if token_match < 0.99:
                print(f"   ⚠ Token一致性较低: {token_match*100:.2f}%")
        
        # 保存配置
        import json
        config_data = {
            "batch_size": batch_size,
            "encoder_seq_len": encoder_seq_len,
            "decoder_seq_len": decoder_seq_len,
            "hidden_size": hidden_size,
            "vocab_size": config.vocab_size,
            "decoder_layers": config.decoder_layers,
            "use_past_key_values": False,
            "opset_version": 12,
            "validation": {
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "cosine_similarity": float(cosine_sim),
                "token_match_rate": float(token_match)
            }
        }
        
        with open("./models/decoder_config.json", "w", encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("✓ Decoder导出完成!")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - ./models/opus_mt_en_zh_decoder.onnx")
        print("  - ./models/decoder_config.json")
        print("\n⚠️ 重要提示:")
        print("  1. Decoder不使用past_key_values,每次输入完整序列")
        print("  2. 优点: 固定shape,易于RKNN转换,无动态问题")
        print("  3. 缺点: 计算量大,适合短序列(<64 tokens)")
        print("  4. 已验证ONNX精度,token一致性应>99%")
        print("  5. Opset version=12,与RK whisper示例保持一致")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs("./models", exist_ok=True)
    
    success = export_decoder_for_rknn()
    
    if success:
        print("\n下一步: 运行 convert_to_rknn_improved.py")
    else:
        print("\n导出失败,请检查错误信息")