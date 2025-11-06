# test_rknn_dynamic_shape.py
"""
测试RKNN是否支持动态shape的cache
"""

from rknn.api import RKNN
import numpy as np

def test_rknn_conversion():
    print("=" * 60)
    print("测试 RKNN 转换（动态shape）")
    print("=" * 60)
    
    rknn = RKNN(verbose=True)
    
    # 配置
    print("\n配置模型...")
    rknn.config(target_platform='rk3576')
    
    # 加载ONNX
    print("\n加载ONNX...")
    ret = rknn.load_onnx('../model/helsinki_decoder_incremental.onnx')
    
    if ret != 0:
        print("❌ 加载失败！")
        return False
    
    print("✓ ONNX加载成功")
    
    # 构建
    print("\n构建RKNN...")
    ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        print("❌ 构建失败！动态shape可能不支持")
        print("\n需要使用固定shape的方案")
        return False
    
    print("✓ RKNN构建成功")
    
    # 导出
    print("\n导出RKNN...")
    ret = rknn.export_rknn('../models/helsinki_decoder_incremental.rknn')
    
    if ret != 0:
        print("❌ 导出失败！")
        return False
    
    print("✓ RKNN导出成功")
    
    # 初始化
    print("\n初始化RKNN...")
    ret = rknn.init_runtime()
    
    if ret != 0:
        print("❌ 初始化失败！")
        return False
    
    print("✓ RKNN初始化成功")
    
    # 测试推理
    print("\n测试推理...")
    
    # 准备输入
    input_ids = np.array([[65000]], dtype=np.int64)
    encoder_hidden = np.random.randn(1, 64, 512).astype(np.float32)
    encoder_mask = np.ones((1, 64), dtype=np.int64)
    
    # 空cache (seq_len=0)
    empty_caches = []
    for _ in range(6):
        empty_caches.append(np.zeros((1, 8, 0, 64), dtype=np.float32))
        empty_caches.append(np.zeros((1, 8, 0, 64), dtype=np.float32))
    
    inputs = [input_ids, encoder_hidden, encoder_mask] + empty_caches
    
    try:
        outputs = rknn.inference(inputs=inputs)
        print(f"✓ 推理成功！")
        print(f"  Logits shape: {outputs[0].shape}")
        print(f"  Cache[0] shape: {outputs[1].shape}")
        
        # 测试第二步（cache不为空）
        print("\n测试第二步（cache_len=1）...")
        cache_1 = []
        for i in range(1, 13):
            cache_1.append(outputs[i])
        
        inputs_step2 = [input_ids, encoder_hidden, encoder_mask] + cache_1
        outputs_step2 = rknn.inference(inputs=inputs_step2)
        
        print(f"✓ 第二步推理成功！")
        print(f"  Cache[0] shape: {outputs_step2[1].shape}")
        
        rknn.release()
        
        print("\n" + "=" * 60)
        print("🎉 RKNN支持动态shape！可以直接转换")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        rknn.release()
        return False

if __name__ == "__main__":
    success = test_rknn_conversion()
    
    if not success:
        print("\n" + "=" * 60)
        print("备选方案：使用固定shape")
        print("=" * 60)
        print("需要重新导出ONNX，将cache固定为max_len-1")
        print("例如: past_key shape = [1, 8, 63, 64]")
        print("使用attention_mask控制有效长度")