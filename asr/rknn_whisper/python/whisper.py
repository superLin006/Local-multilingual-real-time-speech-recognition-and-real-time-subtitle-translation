import numpy as np
from rknn.api import RKNN
import argparse
import soundfile as sf
import onnxruntime
import torch
import torch.nn.functional as F
import scipy
import os # <== MODIFICATION START: Import os module
import sys # <== MODIFICATION START: Import sys module

# Whisper 支持的所有语言 task_code 映射，直接写在脚本里
lang2code = {
  "en":50259, "zh":50260, "de":50261, "es":50262, "ru":50263,
  "ko":50264, "fr":50265, "ja":50266, "pt":50267, "tr":50268,
  "pl":50269, "ca":50270, "nl":50271, "ar":50272, "sv":50273,
  "it":50274, "id":50275, "hi":50276, "fi":50277, "vi":50278,
  "he":50279, "uk":50280, "el":50281, "ms":50282, "cs":50283,
  "ro":50284, "da":50285, "hu":50286, "ta":50287, "no":50288,
  "th":50289, "ur":50290, "hr":50291, "bg":50292, "lt":50293,
  "la":50294, "mi":50295, "ml":50296, "cy":50297, "sk":50298,
  "te":50299, "fa":50300, "lv":50301, "bn":50302, "sr":50303,
  "az":50304, "sl":50305, "kn":50306, "et":50307, "mk":50308,
  "br":50309, "eu":50310, "is":50311, "hy":50312, "ne":50313,
  "mn":50314, "bs":50315, "kk":50316, "sq":50317, "sw":50318,
  "gl":50319, "mr":50320, "pa":50321, "si":50322, "km":50323,
  "sn":50324, "yo":50325, "so":50326, "af":50327, "oc":50328,
  "ka":50329, "be":50330, "tg":50331, "sd":50332, "gu":50333,
  "am":50334, "yi":50335, "lo":50336, "uz":50337, "fo":50338,
  "ht":50339, "ps":50340, "tk":50341, "nn":50342, "mt":50343,
  "sa":50344, "lb":50345, "my":50346, "bo":50347, "tl":50348,
  "mg":50349, "as":50350, "tt":50351, "haw":50352, "ln":50353,
  "ha":50354, "ba":50355, "jw":50356, "su":50357, "translate":50358,
  "transcribe":50359, "startoflm":50360, "startofprev":50361,
  "nospeech":50362, "notimestamps":50363
}

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
MAX_LENGTH = CHUNK_LENGTH * 100
N_MELS = 80

def ensure_sample_rate(waveform, original_sr, desired_sr=16000):
    if original_sr != desired_sr:
        desired_len = int(round(len(waveform)/original_sr*desired_sr))
        waveform = scipy.signal.resample(waveform, desired_len)
    return waveform, desired_sr

def ensure_channels(waveform, orig_ch, desired_ch=1):
    if orig_ch != desired_ch:
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_ch

def get_char_index(c):
    if 'A'<=c<='Z': return ord(c)-65
    if 'a'<=c<='z': return ord(c)-97+26
    if '0'<=c<='9': return ord(c)-48+52
    if c=='+': return 62
    if c=='/': return 63
    print(f"Unknown char {c}"); exit(-1)

def base64_decode(enc):
    if not enc: return ""
    out = bytearray(len(enc)//4*3)
    i=o=0
    while i<len(enc):
        if enc[i]=='=': break
        b1 = (get_char_index(enc[i])<<2)|((get_char_index(enc[i+1])&0x30)>>4)
        out[o]=b1
        if enc[i+2]!='=':
            b2 = ((get_char_index(enc[i+1])&0x0f)<<4)|((get_char_index(enc[i+2])&0x3c)>>2)
            out[o+1]=b2
            if enc[i+3]!='=':
                b3 = ((get_char_index(enc[i+2])&0x03)<<6)|get_char_index(enc[i+3])
                out[o+2]=b3; o+=3
            else:
                o+=2
        else:
            o+=1
        i+=4
    return out.decode('utf-8', errors='replace')

def read_vocab(path):
    vocab = {}
    with open(path,'r') as f:
        for line in f:
            parts = line.strip().split(' ',1)
            k = parts[0]; v = parts[1] if len(parts)>1 else ""
            vocab[k] = v
    return vocab

def pad_or_trim(x):
    mel = np.zeros((N_MELS, MAX_LENGTH), np.float32)
    L = min(x.shape[1], MAX_LENGTH)
    mel[:, :L] = x[:, :L]
    return mel

def mel_filters():
    data = np.loadtxt("../model/mel_80_filters.txt",dtype=np.float32).reshape(80,201)
    return torch.from_numpy(data)

def log_mel_spectrogram(audio):
    if not torch.is_tensor(audio): audio = torch.from_numpy(audio)
    win = torch.hann_window(N_FFT)
    st = torch.stft(audio, N_FFT, HOP_LENGTH, window=win, return_complex=True)
    mags = st[..., :-1].abs()**2
    f = mel_filters()
    m = f @ mags
    lg = torch.clamp(m, min=1e-10).log10()
    lg = torch.maximum(lg, lg.max()-8.0)
    return ((lg+4.0)/4.0)

def init_model(path, target=None, device_id=None):
    if path.endswith(".rknn"):
        m = RKNN()
        m.load_rknn(path)
        m.init_runtime(target=target, device_id=device_id)
    else:
        m = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
    return m

def run_encoder(model, inp):
    if 'rknn' in str(type(model)):
        return model.inference(inputs=inp)[0]
    else:
        return model.run(None, {"x":inp})[0]

def _decode(model, tokens, enc_out):
    if 'rknn' in str(type(model)):
        return model.inference([np.array([tokens], 'int64'), enc_out])[0]
    else:
        return model.run(None, {"tokens":np.array([tokens],'int64'), "audio":enc_out})[0]

# <== MODIFICATION START: Modify function signature to accept calibration args
def run_decoder(dec_model, enc_out, vocab, task_code, save_path=None, run_id=None):
# <== MODIFICATION END
    end_id = 50257
    tokens = [50258, task_code, 50359, 50363] * 3
    ts_begin = 50364
    next_id = 50258
    out_str = ""
     
    # <== MODIFICATION START: Add flag to save only on the first loop
    is_first_loop = True
    # <== MODIFICATION END

    while next_id != end_id:
        # <== MODIFICATION START: Save decoder inputs on the first loop
        if is_first_loop and save_path is not None and run_id is not None:
            # Prepare decoder input data
            decoder_tokens_input = np.array([tokens], 'int64')
            decoder_encoder_output = enc_out

            # Define file paths
            token_file = os.path.join(save_path, f"decoder_tokens_{run_id}.npy")
            encoder_out_file = os.path.join(save_path, f"decoder_encoder_output_{run_id}.npy")
            
            # Save the numpy arrays
            np.save(token_file, decoder_tokens_input)
            np.save(encoder_out_file, decoder_encoder_output)
            print(f"    [INFO] Saved decoder token input to: {token_file}")
            print(f"    [INFO] Saved decoder encoder output to: {encoder_out_file}")

            is_first_loop = False
        # <== MODIFICATION END
        dec = _decode(dec_model, tokens, enc_out)
        next_id = dec[0,-1].argmax()
        tokens.append(next_id)
        if next_id==end_id: break
        if next_id>ts_begin: continue
        if len(tokens)>12: tokens.pop(4)
        out_str += vocab.get(str(next_id),"")
    out_str = out_str.replace('\u0120',' ').replace('<|endoftext|>','').replace('\n','')
    try:
        out_str = base64_decode(out_str)
    except:
        pass
    return out_str

def release_model(m):
    if 'rknn' in str(type(m)): m.release()
    else: del m

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Whisper Python Demo')
    parser.add_argument('--encoder_model_path', required=True)
    parser.add_argument('--decoder_model_path', required=True)
    parser.add_argument('--task', required=True,
                        help='language code, e.g. en, zh, de, …')
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--target', default='rk3576')
    parser.add_argument('--device_id', default=None)
     # <== MODIFICATION START: Add new arguments for saving calibration data
    parser.add_argument('--save_calibration_path', default=None, type=str, help='Path to save calibration data. If set, will run in calibration data generation mode.')
    parser.add_argument('--run_id', default='00', type=str, help='A unique ID for this run, e.g., "01", "sample_zh_1".')
    # <== MODIFICATION END
    args = parser.parse_args()


    if args.task not in lang2code:
        print("Unsupported language. Supported:", list(lang2code.keys()))
        exit(1)
    task_code = lang2code[args.task]

    # <== MODIFICATION START: Check if we need to create the output directory
    if args.save_calibration_path:
        if not os.path.exists(args.save_calibration_path):
            os.makedirs(args.save_calibration_path)
            print(f"[INFO] Created directory for calibration data: {args.save_calibration_path}")
    # <== MODIFICATION END

    vocab = read_vocab("../model/vocab.txt")
    audio, sr = sf.read(args.audio_path)
    ch = audio.ndim
    audio, _ = ensure_channels(audio, ch)
    audio, sr = ensure_sample_rate(audio, sr)
    arr = log_mel_spectrogram(np.array(audio, np.float32)).numpy()
    x_mel = np.expand_dims(pad_or_trim(arr), 0)

     # <== MODIFICATION START: Save encoder input if in calibration mode
    if args.save_calibration_path:
        encoder_input_file = os.path.join(args.save_calibration_path, f"encoder_input_{args.run_id}.npy")
        np.save(encoder_input_file, x_mel)
        print(f"[INFO] Saved encoder input to: {encoder_input_file}")
    # <== MODIFICATION END

    enc_model = init_model(args.encoder_model_path, args.target, args.device_id)
    dec_model = init_model(args.decoder_model_path, args.target, args.device_id)
    enc_out = run_encoder(enc_model, x_mel)
    
# <== MODIFICATION START: Pass calibration args to run_decoder
    result = run_decoder(dec_model, enc_out, vocab, task_code, args.save_calibration_path, args.run_id)
    # <== MODIFICATION END

    print("\nWhisper output:", result)

    release_model(enc_model)
    release_model(dec_model)
