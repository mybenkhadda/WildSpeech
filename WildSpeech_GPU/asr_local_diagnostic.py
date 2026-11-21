"""Standalone offline model diagnostic and optional authorized-audio ASR smoke test.

Does not modify WildSpeech model reviews, download models, or approve competition use.
Usage:
 python asr_local_diagnostic.py --inspect
 python asr_local_diagnostic.py --audio path/to/authorized_clip.wav --acknowledge-review
"""
import argparse
import json
from pathlib import Path
import sys

MODEL_ID = 'facebook/wav2vec2-base-960h'
REVISION = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'


def find_snapshot():
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id == MODEL_ID:
            for rev in repo.revisions:
                if rev.commit_hash == REVISION:
                    return Path(rev.snapshot_path)
    return None


def inspect():
    import torch
    result = {'model_id': MODEL_ID, 'revision': REVISION,
              'torch': torch.__version__, 'cuda_available': bool(torch.cuda.is_available())}
    snap = find_snapshot()
    result['snapshot'] = str(snap) if snap else None
    required = ['config.json','preprocessor_config.json','vocab.json']
    if snap:
        result['files'] = {name: (snap / name).is_file() for name in required}
        result['weight_files'] = [p.name for p in snap.iterdir() if p.name in ('model.safetensors','pytorch_model.bin') and p.is_file()]
        result['technical_files_ready'] = all(result['files'].values()) and bool(result['weight_files'])
    else:
        result['files'] = {}
        result['weight_files'] = []
        result['technical_files_ready'] = False
    print(json.dumps(result, indent=2))
    return snap if result['technical_files_ready'] else None


def transcribe(path, snapshot):
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from transformers import AutoProcessor, AutoModelForCTC
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Audio is empty or contains nonfinite values')
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    if len(x) / 16000 > 30:
        raise ValueError('Smoke test expects a recording of at most 30 seconds')
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCTC.from_pretrained(str(snapshot),local_files_only=True,use_safetensors=(snapshot / 'model.safetensors').is_file())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    inputs = processor(x, sampling_rate=16000, return_tensors='pt')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).logits.argmax(dim=-1)
    transcript = processor.batch_decode(prediction.cpu())[0]
    print(json.dumps({'model':MODEL_ID,'audio_file':str(path),'device':device,'transcript':transcript},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--audio',type=Path)
    p.add_argument('--acknowledge-review',action='store_true',help='Confirms you reviewed rights and permission for the selected audio/model. Does not change competition eligibility.')
    args=p.parse_args()
    snap=inspect()
    if args.audio:
        if not args.acknowledge_review:
            p.error('--audio requires --acknowledge-review; this does not approve competition eligibility')
        if not args.audio.is_file():
            p.error('Audio file does not exist')
        if snap is None:
            p.error('Checkpoint is incomplete; see inspection result')
        transcribe(args.audio,snap)
    elif not args.inspect:
        p.print_help()

if __name__=='__main__':
    main()
