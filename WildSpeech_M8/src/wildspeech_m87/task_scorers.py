"""Prompt-conditioned task evidence, graded correctness and partial recall.

These are ASR-derived observational proxies; not human-verified phonology.
The expected prompt NEVER enters the free word or phone decoder.
"""
from __future__ import annotations
import re,unicodedata
import numpy as np
from difflib import SequenceMatcher
from .ctc_alignment import edit_counts,weighted_edit,greedy_tokens,posterior_features

FUNCTION={"a","an","the","and","or","but","to","of","in","on","at","for",
          "from","with","is","are","was","were","be","been","it","he","she",
          "they","i","you","we","my","your","his","her","their","not"}
def words(s):return re.findall(r"[a-z]+(?:'[a-z]+)?",str(s or "").lower())
def lcs(a,b):
    n=len(a);m=len(b);dp=np.zeros((n+1,m+1),dtype=int)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):
            dp[i,j]=1+dp[i-1,j-1] if x==y else max(dp[i-1,j],dp[i,j-1])
    return int(dp[n,m])

def word_partial_credit(target,predicted):
    ref=words(target);hyp=words(predicted)
    if not ref:return {"word_partial_available":0.}
    counts=edit_counts(ref,hyp)
    matched=lcs(ref,hyp)
    ref_content=[x for x in ref if x not in FUNCTION]
    ref_function=[x for x in ref if x in FUNCTION]
    hits=[x for x in hyp]
    def overlap(refs):
        pool=list(hits);count=0
        for x in refs:
            if x in pool:pool.remove(x);count+=1
        return count/max(1,len(refs))
    endings=["s","ed","ing"]
    endwords=[x for x in ref if any(x.endswith(s) for s in endings)]
    return {
        "word_partial_available":1.,
        "word_coverage":overlap(ref),
        "word_lcs_ratio":matched/len(ref),
        "word_order_ratio":matched/max(1,len(hyp)),
        "word_content_coverage":overlap(ref_content) if ref_content else 0.,
        "word_function_coverage":overlap(ref_function) if ref_function else 0.,
        "word_endings_coverage":overlap(endwords) if endwords else 0.,
        "word_normalized_edits":counts["normalized_edits"],
        "word_insertions":float(counts["insertions"]),
        "word_deletions":float(counts["deletions"]),
        "word_substitutions":float(counts["substitutions"]),
        "word_hyp_length":float(len(hyp)),
        "word_ref_length":float(len(ref)),
    }

def phone_task_score(target_ids,pred_ids,task,confidence=0.):
    if not target_ids:return {"task_phone_available":0.}
    edits=edit_counts(target_ids,pred_ids)
    n=len(target_ids)
    loc=edits["error_locations"]
    return {
        "task_phone_available":1.,
        "task_phone_graded_accuracy":float(max(0,1-edits["normalized_edits"])),
        "task_phone_order_ratio":float(lcs(target_ids,pred_ids)/n),
        "task_phone_omission_rate":float(edits["deletions"]/n),
        "task_phone_insertion_rate":float(edits["insertions"]/n),
        "task_phone_substitution_rate":float(edits["substitutions"]/n),
        "task_phone_initial_errors":float(sum(i<n/3 for i in loc)/n),
        "task_phone_middle_errors":float(sum(n/3<=i<2*n/3 for i in loc)/n),
        "task_phone_final_errors":float(sum(i>=2*n/3 for i in loc)/n),
        "task_phone_confidence_x_accuracy":float(
            max(0,1-edits["normalized_edits"])*confidence),
        "task_phone_segmented_proxy":float(task=="blending" and
                                            len(pred_ids)>len(target_ids)*1.5),
    }

def phone_evidence(logprobs,target_ids,blank,token_map=None):
    x=np.asarray(logprobs,float)
    evidence=posterior_features(x,target_ids,blank)
    pred=greedy_tokens(x,blank)
    if target_ids:
        changes=edit_counts(target_ids,pred)
        evidence.update({
            "phone_edit_rate":changes["normalized_edits"],
            "phone_weighted_edit_rate":weighted_edit(target_ids,pred,token_map),
            "phone_deletions":float(changes["deletions"]),
            "phone_substitutions":float(changes["substitutions"]),
            "phone_insertions":float(changes["insertions"])})
    else:
        evidence.update({"phone_edit_rate":np.nan,
                         "phone_weighted_edit_rate":np.nan})
    return evidence

def clip_task_features(task,target_ids,whole_logp,child_logp,random_logp,blank,
                       localization,expected_text,word_transcript=""):
    outputs={}
    for prefix,logp in (("whole",whole_logp),("child",child_logp),
                        ("random",random_logp)):
        if logp is None or not len(logp):continue
        ev=phone_evidence(logp,target_ids,blank)
        outputs.update({prefix+"_"+k:v for k,v in ev.items()
                        if isinstance(v,(int,float,np.number))})
    child=child_logp if child_logp is not None and len(child_logp) else whole_logp
    pred=greedy_tokens(child,blank)
    outputs.update(phone_task_score(target_ids,pred,task,
                                    localization.get("localization_confidence",0.)))
    if task=="sentence_repetition" and word_transcript:
        outputs.update(word_partial_credit(expected_text,word_transcript))
    else:
        outputs["word_partial_available"]=0.
    # No claim this is prompt-end-to-child-answer time; VAD/prompt end is unknown.
    outputs["child_start_fraction"]=float(localization.get("response_start_s",0.)/
                                          max(1e-6,localization.get("duration_s",1.)))
    outputs["child_duration_s"]=float(
        localization.get("response_end_s",0.)-localization.get("response_start_s",0.))
    return outputs
