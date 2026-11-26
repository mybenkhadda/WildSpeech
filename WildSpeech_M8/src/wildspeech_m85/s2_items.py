"""S2: student-leave-one-out prompt statistics, fit on training screening clips ONLY.

No hidden test row is used for estimation. For each training student, subtract
ALL of that student's scored attempts from both prompt- and task-level counts.
For validation/test students, use the full outer-training item statistics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import TASKS,normalize_prompt,validate_student_inputs

STAT_NAMES=("prompt_count","seen_prompt_share","mean_expected_difficulty",
            "difficulty_weighted_success","mean_success_surprise",
            "mean_item_log_support","unscored_fraction","mean_expected_success")


class S2ItemEncoder:
    def __init__(self,kappa=10.):
        if kappa<=0:raise ValueError("kappa must be positive")
        self.kappa=float(kappa)

    def fit(self,students,clips):
        validate_student_inputs(students,clips)
        self.train_ids=set(students.index.astype(str))
        scored=clips[["subject_id","task","expected_text","item_score"]].copy()
        scored["item_score"]=pd.to_numeric(scored.item_score,errors="coerce")
        scored=scored[scored.item_score.notna()].copy()
        scored["prompt"]=scored.expected_text.map(normalize_prompt)
        self.task_global={}
        self.item_global={}
        self.task_student={}
        self.item_student={}
        for key,g in scored.groupby(["task"],sort=False):
            self.task_global[str(key[0])]=(float(g.item_score.sum()),len(g))
        for (task,prompt),g in scored.groupby(["task","prompt"],sort=False):
            self.item_global[(str(task),str(prompt))]=(float(g.item_score.sum()),len(g))
        for (task,sid),g in scored.groupby(["task","subject_id"],sort=False):
            self.task_student[(str(task),str(sid))]=(float(g.item_score.sum()),len(g))
        for (task,prompt,sid),g in scored.groupby(["task","prompt","subject_id"],sort=False):
            self.item_student[(str(task),str(prompt),str(sid))]=(float(g.item_score.sum()),len(g))
        self.n_scored=len(scored)
        return self

    def expected_probability(self,task,prompt,subject_id=None,exclude_self=False):
        """Return prompt success, effective support; self removal proven by unit tests."""
        task=str(task);prompt=normalize_prompt(prompt);sid=str(subject_id)
        item_s,item_n=self.item_global.get((task,prompt),(0.,0))
        task_s,task_n=self.task_global.get(task,(0.,0))
        if exclude_self:
            si,ni=self.item_student.get((task,prompt,sid),(0.,0))
            st,nt=self.task_student.get((task,sid),(0.,0))
            item_s-=si; item_n-=ni
            task_s-=st; task_n-=nt
        item_n=max(0,item_n);task_n=max(0,task_n)
        task_mean=task_s/task_n if task_n else .5
        p=(item_s+self.kappa*task_mean)/(item_n+self.kappa)
        return float(np.clip(p,0.,1.)),int(item_n)

    def _build(self,students,clips,exclude_self):
        validate_student_inputs(students,clips)
        if not exclude_self and (self.train_ids & set(students.index.astype(str))):
            raise ValueError("Cannot transform training IDs without leave-one-student-out encoding")
        if exclude_self and not set(students.index.astype(str)).issubset(self.train_ids):
            raise ValueError("LOO training transform called with non-training student")
        d=clips[["subject_id","task","expected_text","item_score"]].copy()
        d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
        d["prompt"]=d.expected_text.map(normalize_prompt)
        rows=[]
        task_cols=[f"s2_{task}_{name}" for task in TASKS for name in STAT_NAMES]
        for sid in students.index.astype(str):
            student=d[d.subject_id==sid]
            feature={}
            for task in TASKS:
                subset=student[student.task==task]
                scored=subset[subset.item_score.notna()]
                ps=[];supports=[];differences=[];weighted=[]
                for row in scored.itertuples():
                    p,n=self.expected_probability(task,row.prompt,sid,exclude_self)
                    ps.append(p);supports.append(n)
                    differences.append(float(row.item_score)-p)
                    weighted.append(float(row.item_score)*(2.-p))
                for row in subset[subset.item_score.isna()].itertuples():
                    # Prompt coverage can be observed even if response was not scored.
                    pass
                prefix=f"s2_{task}_"
                vals={
                    "prompt_count":float(subset.prompt.nunique()),
                    "seen_prompt_share":float(np.mean([self.expected_probability(
                        task,p,sid,exclude_self)[1]>0 for p in subset.prompt])) if len(subset) else 0.,
                    "mean_expected_difficulty":float(np.mean([1-p for p in ps])) if ps else 0.,
                    "difficulty_weighted_success":float(np.mean(weighted)) if weighted else 0.,
                    "mean_success_surprise":float(np.mean(differences)) if differences else 0.,
                    "mean_item_log_support":float(np.mean(np.log1p(supports))) if supports else 0.,
                    "unscored_fraction":float(subset.item_score.isna().mean()) if len(subset) else 0.,
                    "mean_expected_success":float(np.mean(ps)) if ps else 0.,
                }
                feature.update({prefix+k:v for k,v in vals.items()})
            rows.append(feature)
        result=pd.DataFrame(rows,index=students.index).reindex(columns=task_cols)
        if result.isna().any().any():
            raise ValueError("S2 produced NaN")
        return result

    def transform_train(self,students,clips):
        return self._build(students,clips,True)

    def transform(self,students,clips):
        return self._build(students,clips,False)
