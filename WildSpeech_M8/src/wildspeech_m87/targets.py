"""Audited expected_text -> IPA CTC target registry, never pseudo-transcribed.

Model tokenizer alphabet includes IPA unicode characters and separators.
Unmapped or unknown G2P outputs are flagged, never forced to nearest token.
No item order or removed phone is guessed from file names.
"""
from __future__ import annotations
import unicodedata,re
import pandas as pd
from .contracts import TASKS
from wildspeech_m84.data import atomic_json

def normalize_text(s):
    return " ".join(str(s or "").lower().strip().split())

def ipa_normalize(s):
    return unicodedata.normalize("NFC",str(s)).replace("ˈ","").replace("ˌ","").strip()

class TargetRegistry:
    def __init__(self,tokenizer,overrides=None):
        self.tokenizer=tokenizer
        self.vocab=tokenizer.get_vocab()
        self.unk=int(getattr(tokenizer,"unk_token_id",-1) or -1)
        self.blank=int(tokenizer.pad_token_id)
        self.overrides=overrides or {}
        try:import eng_to_ipa
        except ImportError:eng_to_ipa=None
        self.g2p=eng_to_ipa

    def map_text(self,text):
        text=normalize_text(text)
        if not text:return {"valid":False,"ipa":"","ids":[],"reason":"EMPTY_PROMPT"}
        if text in self.overrides:
            ipa=ipa_normalize(self.overrides[text]);source="OVERRIDE"
        elif self.g2p is not None:
            ipa=ipa_normalize(self.g2p.convert(text));source="ENG_TO_IPA"
        else:
            return {"valid":False,"ipa":"","ids":[],"reason":"G2P_UNAVAILABLE"}
        if "*" in ipa or not ipa:
            return {"valid":False,"ipa":ipa,"ids":[],"reason":"UNRECOGNIZED_G2P"}
        tokens=self.tokenizer.tokenize(ipa)
        ids=self.tokenizer.convert_tokens_to_ids(tokens)
        if not isinstance(ids,list):ids=[ids]
        if not ids or any(k is None or int(k)==self.unk or int(k)==self.blank
                          for k in ids):
            return {"valid":False,"ipa":ipa,"ids":[],"reason":"UNMAPPED_IPA"}
        return {"valid":True,"ipa":ipa,"ids":[int(k) for k in ids],
                "reason":"OK","source":source}

    def make(self,clips):
        unique=clips[["task","expected_text"]].drop_duplicates().sort_values(["task","expected_text"])
        records=[]
        for row in unique.itertuples(index=False):
            if str(row.task) not in TASKS:raise ValueError("Unexpected task")
            record=self.map_text(row.expected_text)
            records.append({"task":row.task,"expected_text":row.expected_text,
                            **record})
        return pd.DataFrame(records)

    @staticmethod
    def lookups(registry):
        return {(str(r.task),normalize_text(r.expected_text)):r._asdict()
                for r in registry.itertuples(index=False)}
