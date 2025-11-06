#!/usr/bin/env python3
"""
极简 RKNN 转换脚本
完全模仿 Rockchip 官方 lite-transformer 示例
python step3_convert_to_rknn.py rk3576
"""
from rknn.api import RKNN
import sys

def convert_model(model_path, platform, output_path):
    """
    使用最简配置转换模型
    完全按照 lite-transformer 的方式
    """
    # Create RKNN object
    rknn = RKNN(verbose=False)
    
    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')
    
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        return False
    print('done')
    
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return False
    print('done')
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        return False
    print('done')
    
    # Release
    rknn.release()
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {} [platform]".format(sys.argv[0]))
        print("       platform: rk3562, rk3566, rk3568, rk3576, rk3588, etc.")
        print("\n默认使用 rk3576")
        platform = 'rk3576'
    else:
        platform = sys.argv[1]
    
    print("=" * 60)
    print("Helsinki MT 模型转换 (极简版本)")
    print("完全模仿 Rockchip lite-transformer 示例")
    print("=" * 60)
    print(f"目标平台: {platform}")
    print("转换模式: FP16 (不量化)")
    print("=" * 60)
    
    # Convert Encoder
    print("\n=== 步骤 1: 转换 Encoder ===")
    encoder_success = convert_model(
        "../model/opus_mt_zh_en_encoder.onnx",
        platform,
        "../model/opus_mt_zh_en_encoder.rknn"
    )
    
    if not encoder_success:
        print("\n✗ Encoder 转换失败!")
        return False
    
    print("✓ Encoder 转换成功")
    
    # Convert Decoder
    print("\n=== 步骤 2: 转换 Decoder ===")
    decoder_success = convert_model(
        "../model/opus_mt_zh_en_decoder.onnx",
        platform,
        "../model/opus_mt_zh_en_decoder.rknn"
    )
    
    if not decoder_success:
        print("\n✗ Decoder 转换失败!")
        return False
    
    print("✓ Decoder 转换成功")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ 转换完成!")
    print("=" * 60)
    print("\n生成的 RKNN 模型:")
    print("  - ./model/opus_mt_xx_xx_encoder.rknn")
    print("  - ./model/opus_mt_xx_xx_decoder.rknn")

    print("\n配置说明:")
    print("  - 使用默认的 optimization_level (不显式设置)")
    print("  - FP16 精度 (do_quantization=False)")
    print("  - 完全模仿 Rockchip 官方示例")
    
    print("\n下一步:")
    print("  1. 部署到板子: adb push ./model/*.rknn /data/local/tmp/rknn_helsinki_demo/model/")
    print("  2. 运行测试: ./rknn_helsinki_demo \"WHAT a beautiful day!\"")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)