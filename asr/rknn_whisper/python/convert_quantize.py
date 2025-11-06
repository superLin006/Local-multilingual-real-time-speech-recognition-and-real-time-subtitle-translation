#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_whisper.py

量化 Whisper 的 encoder 和 decoder 两个 ONNX 模型，
分别生成 encoder_int8.rknn 和 decoder_int8.rknn。
python convert.py --encoder_onnx ../model/whisper_encoder_base_20s.onnx  --decoder_onnx ../model/whisper_decoder_base_20s.onnx --enc_dataset encoder_dataset.txt  --dec_dataset decoder_dataset.txt  --platform rk3576
"""

import argparse
from rknn.api import RKNN

def quantize_encoder(onnx_path, dataset_txt, target, platform, output_rknn):
    print(f"[+] Quantizing encoder:\n    onnx: {onnx_path}\n    dataset: {dataset_txt}")
    rknn = RKNN(verbose=True)
    # 配置：按照你的平台／算法需求调整
    rknn.config(
        target_platform=platform,
        quantized_method='channel',      # 或 'sym', 'asym' 等
        quantized_algorithm='normal'       # 或 'kl', 'minmax' 等
    )
    # 加载 encoder.onnx
    # 假设它的输入名叫 "x"，shape 是 [1,80,2000]
    rknn.load_onnx(
        model=onnx_path,
        inputs=['x'],
        input_size_list=[[1, 80, 2000]]
    )
    # 量化，并使用前面生成的 encoder_dataset.txt
    rknn.build(do_quantization=True, dataset=dataset_txt)
    # 导出结果
    rknn.export_rknn(output_rknn)
    print(f"[+] Encoder INT8 rknn saved to: {output_rknn}")
    rknn.release()


def quantize_decoder(onnx_path, dataset_txt, target, platform, output_rknn):
    print(f"[+] Quantizing decoder:\n    onnx: {onnx_path}\n    dataset: {dataset_txt}")
    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform=platform,
        quantized_method='channel',
        quantized_algorithm='normal'
    )
    # 加载 decoder.onnx
    # 假设它的输入名叫 "tokens" 和 "audio"，shape 分别是 [1,12], [1,1000,512]
    rknn.load_onnx(
        model=onnx_path,
        inputs=['tokens', 'audio'],
        input_size_list=[[1, 12], [1, 1000, 512]]
    )
    # 使用 decoder_dataset.txt
    rknn.build(do_quantization=True, dataset=dataset_txt)
    rknn.export_rknn(output_rknn)
    print(f"[+] Decoder INT8 rknn saved to: {output_rknn}")
    rknn.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize Whisper encoder & decoder to INT8 RKNN")
    parser.add_argument('--encoder_onnx',   type=str, required=True, help='路径到 whisper_encoder.onnx')
    parser.add_argument('--decoder_onnx',   type=str, required=True, help='路径到 whisper_decoder.onnx')
    parser.add_argument('--enc_dataset',    type=str, default='encoder_dataset.txt', help='encoder 校准集列表')
    parser.add_argument('--dec_dataset',    type=str, default='decoder_dataset.txt', help='decoder 校准集列表')
    parser.add_argument('--platform',       type=str, default='rk3576', help='RK目标平台')
    parser.add_argument('--output_enc',     type=str, default='encoder_int8.rknn', help='输出 encoder_int8.rknn 文件名')
    parser.add_argument('--output_dec',     type=str, default='decoder_int8.rknn', help='输出 decoder_int8.rknn 文件名')
    args = parser.parse_args()

    quantize_encoder(
        onnx_path    = args.encoder_onnx,
        dataset_txt  = args.enc_dataset,
        target       = None,
        platform     = args.platform,
        output_rknn  = args.output_enc
    )
    quantize_decoder(
        onnx_path    = args.decoder_onnx,
        dataset_txt  = args.dec_dataset,
        target       = None,
        platform     = args.platform,
        output_rknn  = args.output_dec
    )

