from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from wildspeech_m84.data import load_official,sha_json
from wildspeech_m84.reference import metadata_features,make_reference,REFERENCE_COLUMNS,reference_frames


@pytest.fixture
def synthetic_official(tmp_path):
    rng=np.random.default_rng(4242)
    n=60
    ids=[f"{i:012d}" for i in range(n)]
    test_ids=[f"T{i:011d}" for i in range(18)]
    tr=pd.DataFrame({"subject_id":ids,"instrument":["wj4" if i%2 else "ktea3" for i in range(n)],
                     "original_sample_rate":[11025 if i%3 else 16000 for i in range(n)],
                     "school_id":[f"s{i%10}" for i in range(n)]})
    te=pd.DataFrame({"subject_id":test_ids,"instrument":["wj4" if i%2 else "ktea3" for i in range(18)],
                     "original_sample_rate":[11025 if i%3 else 16000 for i in range(18)]})
    for task in ("blending","deletion","sentence_repetition"):
        tr[task+"_theta"]=rng.normal(size=n)
        te[task+"_theta"]=rng.normal(size=len(te))
    outcomes=90+4*tr.blending_theta+3*tr.deletion_theta+rng.normal(0,7,n)
    lab=pd.DataFrame({"subject_id":ids[::-1],"outcome_score":outcomes.iloc[::-1].to_numpy()})
    fmt=pd.DataFrame({"subject_id":test_ids[::-1],"outcome_score":[0]*18})
    clips=[]
    for split,all_ids in (("train",ids),("test",test_ids)):
        rows=[]
        for sid in all_ids:
            for task in ("blending","deletion","sentence_repetition"):
                if task=="deletion" and int(sid[-1],36)%4==0:continue
                for j in range(2+(int(sid[-1],36)%3)):
                    r={"subject_id":sid,"filename":f"{sid}_{task}_{j}.flac",
                       "task":task,"expected_text":"apple","duration_sec":float(j+1)}
                    if split=="train":r["item_score"] = float((j+int(sid[-1],36))%2)
                    rows.append(r)
        pd.DataFrame(rows).to_csv(tmp_path/f"{split}_clip_metadata.csv",index=False)
    tr.to_csv(tmp_path/"train_student_metadata.csv",index=False)
    te.to_csv(tmp_path/"test_student_metadata.csv",index=False)
    lab.to_csv(tmp_path/"train_labels.csv",index=False)
    fmt.to_csv(tmp_path/"submission_format.csv",index=False)
    return tmp_path


def test_exact_reference_matrix_and_model(synthetic_official):
    data=load_official(synthetic_official)
    train,test,y=reference_frames(data)
    assert train.shape==(60,14) and test.shape==(18,14)
    assert list(train)==REFERENCE_COLUMNS
    assert train.index.tolist()==data.train_students.index.tolist()
    assert test.index.tolist()==data.submission_format.index.tolist()
    assert test.index.tolist()!=data.test_students.index.tolist()
    assert data.order_manifest["labels_original_order_equal"] is False
    assert all(train[c].notna().all() for c in REFERENCE_COLUMNS if c.endswith("_n_clips"))
    model=make_reference()
    model.fit(train,y)
    assert isinstance(model,Pipeline)
    assert np.isfinite(model.predict(test)).all()


def test_invalid_official_ids_fail(synthetic_official):
    file=synthetic_official/"train_labels.csv"
    frame=pd.read_csv(file,dtype={"subject_id":str})
    frame.loc[0,"subject_id"]="MISSING_ID"
    frame.to_csv(file,index=False)
    with pytest.raises(ValueError,match="do not match"):
        load_official(synthetic_official)


def test_hash_repeatable():
    assert sha_json({"a":[1,2]})==sha_json({"a":[1,2]})
