"""I3: deterministic, licensed-by-construction orthographic prompt complexity.

These are prompt TEXT features, not inferred phonemes or child pronunciation.
If a verified phoneme lexicon is supplied, it must have explicit provenance and
may be integrated separately; no unlicensed word-frequency lookup is performed.
"""
from __future__ import annotations
import re
import unicodedata
import numpy as np
import pandas as pd
from .contracts import TASKS,validate_inputs

def text_complexity(value):
    if value is None or pd.isna(value):
        return {"chars":0.,"words":0.,"syllable_proxy":0.,
                "consonant_clusters":0.,"long_words":0.,
                "ending_s":0.,"ending_ed":0.,"missing_text":1.}
    s=unicodedata.normalize("NFC",str(value)).casefold()
    words=re.findall(r"[a-z]+(?:'[a-z]+)?",s)
    core="".join(words)
    vowelgroups=len(re.findall(r"[aeiouy]+",core))
    cons=len(re.findall(r"[^aeiouy]{2,}",core))
    return {"chars":float(len(core)),"words":float(len(words)),
            "syllable_proxy":float(vowelgroups),
            "consonant_clusters":float(cons),
            "long_words":float(sum(len(w)>=7 for w in words)),
            "ending_s":float(sum(w.endswith("s") for w in words)),
            "ending_ed":float(sum(w.endswith("ed") for w in words)),
            "missing_text":float(not bool(words))}

KEYS=tuple(text_complexity("hello").keys())

def complexity_features(students,clips):
    validate_inputs(students,clips)
    d=clips[["subject_id","task","expected_text","item_score"]].copy()
    d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
    unique={str(v):text_complexity(v) for v in d.expected_text.drop_duplicates().astype(str)}
    by_student={str(s):g for s,g in d.groupby("subject_id",sort=False)}
    rows=[]
    for sid in students.index.astype(str):
        g=by_student.get(sid,d.iloc[:0])
        feat={}
        for task in TASKS:
            t=g[g.task==task]
            for name in KEYS:
                vals=np.asarray([unique.get(str(v),text_complexity(v))[name]
                                 for v in t.expected_text],float)
                correct=t.item_score.to_numpy(float)
                observed=np.isfinite(correct)
                for suffix,mask in (("all",np.ones(len(t),bool)),
                                    ("correct",observed&(correct>=.5)),
                                    ("incorrect",observed&(correct<.5))):
                    feat[f"i3_{task}_{name}_{suffix}"]=float(np.mean(vals[mask])) if mask.any() else 0.
            feat[f"i3_{task}_complexity_item_count"]=float(len(t))
        rows.append(feat)
    return pd.DataFrame(rows,index=students.index).fillna(0.)
