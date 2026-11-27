"""Pure NumPy CTC forward+Viterbi, IPA edits and bounded posterior evidence.

No torchaudio forced_align: that API was deprecated/removed in 2.9.
Invalid/unmappable targets produce flags, not an artificially high score.
The Viterbi path through a target is NOT proof the response is correct.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

def _states(target,blank):
    z=[blank]
    for t in target:z.extend((int(t),blank))
    return z

def ctc_align(logp,target,blank=0,with_path=True):
    x=np.asarray(logp,dtype=float)
    q=[int(z) for z in target]
    if x.ndim!=2 or not len(x) or not q or not np.isfinite(x).all():
        return {"valid":False,"reason":"INVALID_LOGITS_OR_TARGET"}
    if any(z<0 or z>=x.shape[1] or z==blank for z in q):
        return {"valid":False,"reason":"OUT_OF_VOCAB"}
    minframes=len(q)+sum(a==b for a,b in zip(q[:-1],q[1:]))
    if len(x)<minframes:return {"valid":False,"reason":"INSUFFICIENT_FRAMES"}
    states=_states(q,blank);T=len(x);S=len(states)
    last=np.full(S,-np.inf);last[0]=x[0,blank]
    if S>1:last[1]=x[0,states[1]]
    history=[last.copy()]
    back=np.full((T,S),-1,np.int16) if with_path else None
    for t in range(1,T):
        nxt=np.full(S,-np.inf)
        for j,symbol in enumerate(states):
            froms=[j]
            if j>=1:froms.append(j-1)
            if j>=2 and symbol!=blank and symbol!=states[j-2]:froms.append(j-2)
            terms=last[froms]
            nxt[j]=x[t,symbol]+logsumexp(terms)
            if with_path:back[t,j]=froms[int(np.argmax(terms))]
        last=nxt
        if with_path:history.append(last.copy())
    final=float(logsumexp(last[-2:]))
    if not np.isfinite(final):return {"valid":False,"reason":"UNREACHABLE_TARGET"}
    result={"valid":True,"log_likelihood":final,
            "score_per_phone":final/max(1,len(q)),
            "score_per_frame":final/max(1,T),
            "target_length":len(q),"frames":T}
    if with_path:
        # The forward recursion marginalizes all valid paths. The path below
        # is independently computed with MAX recursion for actual alignment.
        path=_viterbi(x,states,blank)
        result["path"]=path
        result["phone_frames"]=[np.flatnonzero(np.asarray(path)==2*i+1).tolist()
                                for i in range(len(q))]
    return result

def _viterbi(x,states,blank):
    T=len(x);S=len(states)
    d=np.full((T,S),-np.inf,dtype=float)
    b=np.full((T,S),-1,np.int16)
    d[0,0]=x[0,blank]
    if S>1:d[0,1]=x[0,states[1]]
    for t in range(1,T):
        for j,sym in enumerate(states):
            ii=[j]
            if j>=1:ii.append(j-1)
            if j>=2 and sym!=blank and sym!=states[j-2]:ii.append(j-2)
            prev=np.asarray([d[t-1,k] for k in ii])
            best=int(np.argmax(prev))
            d[t,j]=prev[best]+x[t,sym]
            b[t,j]=ii[best]
    state=int(np.argmax(d[-1,-2:]))+S-2
    path=np.zeros(T,np.int16)
    for t in range(T-1,-1,-1):
        path[t]=state
        state=int(b[t,state]) if t>0 else state
    return path.tolist()

def greedy_tokens(logp,blank=0):
    x=np.argmax(logp,axis=-1)
    out=[];last=None
    for z in x:
        if int(z)!=blank and int(z)!=last:out.append(int(z))
        last=z
    return out

def posterior_features(logp,target,blank=0):
    x=np.asarray(logp,float)
    if x.ndim!=2 or not len(x):return {"ctc_valid":0.}
    logits=np.clip(x,-80,0);prob=np.exp(logits)
    h=-np.sum(prob*logits,axis=1)
    order=np.partition(logits,-2,axis=1)
    ans={"ctc_valid":0.,"posterior_entropy":float(h.mean()),
         "posterior_margin":float((order[:,-1]-order[:,-2]).mean()),
         "blank_share":float((np.argmax(x,axis=1)==blank).mean()),
         "greedy_phone_count":float(len(greedy_tokens(x,blank)))}
    alignment=ctc_align(x,target,blank)
    if not alignment.get("valid"):
        ans["ctc_failure"]=alignment.get("reason")
        return ans
    decoded=greedy_tokens(x,blank)
    free=ctc_align(x,decoded,blank,with_path=False) if decoded else {"valid":False}
    ans["ctc_valid"]=1.
    ans["target_logp_per_phone"]=alignment["score_per_phone"]
    ans["target_logp_per_frame"]=alignment["score_per_frame"]
    ans["target_vs_free_per_frame"]=float(
        alignment["score_per_frame"]-free["score_per_frame"]
        if free.get("valid") else 0.)
    conf=[];margins=[];ent=[]
    for t,frames in zip(target,alignment["phone_frames"]):
        if not frames:continue
        ix=np.asarray(frames,int)
        conf.append(float(np.exp(x[ix,int(t)]).mean()))
        competition=np.max(np.delete(x[ix],int(t),axis=1),axis=1)
        margins.append(float((x[ix,int(t)]-competition).mean()))
        ent.append(float(h[ix].mean()))
    ans["aligned_phone_confidence"]=float(np.mean(conf)) if conf else 0.
    ans["aligned_phone_margin"]=float(np.mean(margins)) if margins else 0.
    ans["aligned_phone_entropy"]=float(np.mean(ent)) if ent else 0.
    ans["phone_coverage"]=float(len(conf)/len(target))
    ans["phone_timing_variance"]=float(np.var([len(z) for z in alignment["phone_frames"]]))
    return ans

def edit_counts(a,b):
    """Levenshtein with substitutions/deletions/insertions via traceback."""
    a=list(a);b=list(b);n=len(a);m=len(b)
    dp=np.zeros((n+1,m+1),int)
    dp[:,0]=np.arange(n+1);dp[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,
                        dp[i-1,j-1]+int(a[i-1]!=b[j-1]))
    i=n;j=m;sub=dele=ins=0;locations=[]
    while i or j:
        if i and j and dp[i,j]==dp[i-1,j-1]+int(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:sub+=1;locations.append(i-1)
            i-=1;j-=1
        elif i and dp[i,j]==dp[i-1,j]+1:
            dele+=1;locations.append(i-1);i-=1
        else:ins+=1;j-=1
    return {"edits":float(dp[n,m]),"substitutions":sub,"deletions":dele,
            "insertions":ins,"error_locations":locations,
            "normalized_edits":float(dp[n,m]/max(1,n))}

def weighted_edit(a,b,token_map=None):
    """Simple IPA class similarity; optional vetted panphon can replace map."""
    a=list(a);b=list(b)
    classes={
        "p":"stop","b":"stop","t":"stop","d":"stop","k":"stop","g":"stop",
        "f":"fricative","v":"fricative","s":"fricative","z":"fricative",
        "ʃ":"fricative","ʒ":"fricative","θ":"fricative","ð":"fricative",
        "m":"nasal","n":"nasal","ŋ":"nasal",
    }
    a=[token_map.get(x,x) if token_map else x for x in a]
    b=[token_map.get(x,x) if token_map else x for x in b]
    d=np.zeros((len(a)+1,len(b)+1),float)
    d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):
            cost=0. if x==y else .65 if classes.get(x,"") and classes.get(x)==classes.get(y) else 1.
            d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+cost)
    return float(d[-1,-1]/max(1,len(a)))
