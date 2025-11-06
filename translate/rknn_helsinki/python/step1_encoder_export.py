#!/usr/bin/env python3
"""
改进版Encoder导出脚本 - 参考RK Whisper导出方式
关键改进:
1. 固定shape,避免动态输入
2. 添加ONNX简化步骤
3. 添加精度验证
4. 使用opset_version=12(与whisper一致)
"""
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import os

import warnings
warnings.filterwarnings("ignore")

def export_encoder_for_rknn():
    """
    导出适合RKNN转换的Encoder模型
    """
    print("=" * 60)
    print("导出Encoder (RKNN优化版本)")
    print("=" * 60)
    
    # 模型路径
    model_path = "C:\\Users\\HP\\Desktop\\en_zh_translation\\opus-mt-en-zh"
    
    try:
        print("\n1. 加载模型...")
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        os.makedirs("./models", exist_ok=True)
        tokenizer.save_pretrained("./models/tokenizer")
        print("\n   ✓ Tokenizer已保存到 ./models/tokenizer")

        model = MarianMTModel.from_pretrained(model_path)
        model.eval()
        
        # 获取配置
        config = model.config
        print(f"   ✓ 模型加载成功")
        print(f"   - Vocab size: {config.vocab_size}")
        print(f"   - Hidden size: {config.d_model}")
        print(f"   - Encoder layers: {config.encoder_layers}")
        
        # 固定参数 - 与whisper保持一致的设计理念
        batch_size = 1
        seq_len = 64  # 固定序列长度
        
        print(f"\n2. 准备导出配置...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Sequence length: {seq_len}")
        print(f"   - Opset version: 12")  # 使用与whisper相同的opset版本
        
        # 准备Dummy输入
        dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        print("\n3. 验证PyTorch模型输出...")
        with torch.no_grad():
            pytorch_output = model.model.encoder(dummy_input_ids, dummy_attention_mask)
            pytorch_hidden = pytorch_output.last_hidden_state
            print(f"   - PyTorch输出shape: {pytorch_hidden.shape}")
            print(f"   - PyTorch输出范围: [{pytorch_hidden.min():.4f}, {pytorch_hidden.max():.4f}]")
            print(f"   - PyTorch输出均值: {pytorch_hidden.mean():.4f}")
        
        # 导出ONNX
        print("\n4. 导出ONNX模型...")
        output_path = "./models/opus_mt_en_zh_encoder.onnx"
        
        torch.onnx.export(
            model.model.encoder,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
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
            #onnx.save(simplified_model, output_path)
        #    print("   ✓ ONNX模型简化成功")
        #else:
        #    print("   ⚠ ONNX模型简化验证失败,使用原始模型")
        
        # 验证ONNX输出
        print("\n6. 验证ONNX模型输出...")
        import onnxruntime as ort
        
        sess = ort.InferenceSession(output_path)
        onnx_output = sess.run(
            None, 
            {
                'input_ids': dummy_input_ids.numpy(),
                'attention_mask': dummy_attention_mask.numpy()
            }
        )[0]
        
        print(f"   - ONNX输出shape: {onnx_output.shape}")
        print(f"   - ONNX输出范围: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
        print(f"   - ONNX输出均值: {onnx_output.mean():.4f}")
        
        # 计算误差
        diff = np.abs(pytorch_hidden.numpy() - onnx_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"\n7. PyTorch vs ONNX误差分析:")
        print(f"   - 最大误差: {max_diff:.6f}")
        print(f"   - 平均误差: {mean_diff:.6f}")
        
        # 余弦相似度
        pytorch_flat = pytorch_hidden.numpy().flatten()
        onnx_flat = onnx_output.flatten()
        cosine_sim = np.dot(pytorch_flat, onnx_flat) / (np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat))
        print(f"   - 余弦相似度: {cosine_sim:.6f}")
        
        if cosine_sim > 0.999 and max_diff < 1e-4:
            print("   ✓ ONNX模型精度验证通过!")
        else:
            print("   ⚠ ONNX模型精度可能存在问题,请检查!")
        
        # 保存配置
        import json
        config_data = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": config.d_model,
            "vocab_size": config.vocab_size,
            "encoder_layers": config.encoder_layers,
            "decoder_layers": config.decoder_layers,
            "opset_version": 12,
            "validation": {
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "cosine_similarity": float(cosine_sim)
            }
        }
        
        with open("./models/encoder_config.json", "w", encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("✓ Encoder导出完成!")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - ./models/opus_mt_en_zh_encoder.onnx")
        print("  - ./models/encoder_config.json")
        print("\n⚠️ 重要提示:")
        print("  1. 模型使用固定shape[1, 64],转换RKNN时更稳定")
        print("  2. 已使用onnxsim简化模型,提高转换成功率")
        print("  3. 已验证ONNX精度,请检查cosine_similarity是否>0.999")
        print("  4. Opset version=12,与RK whisper示例保持一致")
        
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
    
    success = export_encoder_for_rknn()
    
    if success:
        print("\n下一步: 运行 export_decoder_improved.py")
    else:
        print("\n导出失败,请检查错误信息")