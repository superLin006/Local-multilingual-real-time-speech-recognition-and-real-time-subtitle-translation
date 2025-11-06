#!/usr/bin/env python3
import os
import re
import sys

def main(calib_dir):
    # 正则匹配文件名，提取 run_id
    enc_re     = re.compile(r'^encoder_input_(.+)\.npy$')
    tok_re     = re.compile(r'^decoder_tokens_(.+)\.npy$')
    encout_re  = re.compile(r'^decoder_encoder_output_(.+)\.npy$')

    enc_files    = {}  # run_id -> encoder_input_*.npy
    tok_files    = {}  # run_id -> decoder_tokens_*.npy
    encout_files = {}  # run_id -> decoder_encoder_output_*.npy

    for fname in os.listdir(calib_dir):
        path = os.path.join(calib_dir, fname)
        if not os.path.isfile(path) or not fname.endswith('.npy'):
            continue
        m = enc_re.match(fname)
        if m:
            enc_files[m.group(1)] = path
            continue
        m = tok_re.match(fname)
        if m:
            tok_files[m.group(1)] = path
            continue
        m = encout_re.match(fname)
        if m:
            encout_files[m.group(1)] = path
            continue

    # 写 encoder_dataset.txt
    with open('encoder_dataset.txt', 'w') as f_enc:
        for run_id, path in sorted(enc_files.items()):
            f_enc.write(path + '\n')
    print(f"[INFO] encoder_dataset.txt: {len(enc_files)} entries")

    # 写 decoder_dataset.txt
    with open('decoder_dataset.txt', 'w') as f_dec:
        cnt = 0
        for run_id, tok_path in sorted(tok_files.items()):
            encout_path = encout_files.get(run_id)
            if not encout_path:
                print(f"[WARN] no encoder_output for run_id={run_id}, skip")
                continue
            f_dec.write(f"{tok_path} {encout_path}\n")
            cnt += 1
    print(f"[INFO] decoder_dataset.txt: {cnt} entries")

if __name__ == '__main__':
    calib_dir = sys.argv[1] if len(sys.argv) > 1 else 'calibration_data'
    if not os.path.isdir(calib_dir):
        print(f"Error: '{calib_dir}' not a directory")
        sys.exit(1)
    main(calib_dir)
