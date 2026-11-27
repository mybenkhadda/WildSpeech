"""Direct seekable .tar and local-file decoding; private audio is not extracted."""
from __future__ import annotations
import io,tarfile,hashlib,math
from pathlib import Path
import numpy as np
import pandas as pd

def canonical(name):return str(name).replace("\\","/").lstrip("./")

class AudioSource:
    def __init__(self,tar_path="",directory="",index_path=None):
        self.tar=Path(tar_path) if tar_path else None
        self.root=Path(directory) if directory else None
        if bool(self.tar)==bool(self.root):raise ValueError("Exactly one tar or audio directory required")
        self.index={}
        if self.tar:
            if not self.tar.is_file() or self.tar.suffix!=".tar":
                raise ValueError("Require uncompressed, seekable .tar")
            if index_path and Path(index_path).is_file():
                t=pd.read_parquet(index_path)
                self.index={r.name:(int(r.offset),int(r.size)) for r in t.itertuples(index=False)}
            else:
                with tarfile.open(self.tar,mode="r:") as f:
                    for member in f:
                        if member.isfile():
                            name=canonical(member.name)
                            if name in self.index:raise ValueError(f"Duplicate TAR member {name}")
                            self.index[name]=(member.offset_data,member.size)
                if index_path:
                    p=Path(index_path);p.parent.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame([{"name":k,"offset":v[0],"size":v[1]}
                                  for k,v in self.index.items()]).to_parquet(p,index=False)
            aliases={}
            for name in self.index:aliases.setdefault(Path(name).name,[]).append(name)
            self.alias={k:v[0] for k,v in aliases.items() if len(v)==1}
        else:
            if not self.root.is_dir():raise FileNotFoundError(self.root)

    def get_bytes(self,name):
        key=canonical(name)
        if self.tar:
            member=key if key in self.index else self.alias.get(Path(key).name)
            if member not in self.index:raise FileNotFoundError(f"Unindexed TAR audio: {key}")
            start,size=self.index[member]
            with self.tar.open("rb") as file:
                file.seek(start);raw=file.read(size)
            if len(raw)!=size:raise IOError(f"Truncated archive read: {member}")
            return raw
        p=(self.root/key).resolve()
        if not p.is_relative_to(self.root.resolve()):raise ValueError("Unsafe filename")
        return p.read_bytes()

    def decode(self,name,sr=16000):
        import soundfile as sf
        from scipy.signal import resample_poly
        raw=self.get_bytes(name)
        wave,original_sr=sf.read(io.BytesIO(raw),dtype="float32",always_2d=True)
        mono=np.mean(wave,axis=1,dtype=np.float32)
        if original_sr!=sr:
            d=math.gcd(original_sr,sr)
            mono=resample_poly(mono,sr//d,original_sr//d).astype(np.float32)
        if not len(mono) or not np.isfinite(mono).all():raise ValueError("Empty or nonfinite audio")
        return mono,sr,hashlib.sha256(raw).hexdigest(),original_sr
