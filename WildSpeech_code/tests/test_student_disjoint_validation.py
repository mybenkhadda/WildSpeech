import numpy as np

from wildspeech.validation import assert_student_disjoint, iter_folds, make_cv_assignments


def test_every_scheme_partitions_students(synthetic_students, cfg):
    assign, status = make_cv_assignments(synthetic_students, cfg)
    assert_student_disjoint(assign)
    for scheme, st in status.items():
        if st["status"] != "OK":
            continue
        for rep, fold, tr, va in iter_folds(assign, scheme):
            assert len(np.intersect1d(tr, va)) == 0
        sub = assign[assign["scheme"] == scheme]
        for rep, g in sub.groupby("repeat"):
            assert g["student_id"].is_unique
            assert set(g["student_id"]) == set(synthetic_students.index)


def test_school_grouped_folds_keep_schools_together(synthetic_students, cfg):
    assign, status = make_cv_assignments(synthetic_students, cfg)
    assert status["CV_B_SCHOOL_GROUP_KFOLD"]["status"] == "OK"
    sub = assign[assign["scheme"] == "CV_B_SCHOOL_GROUP_KFOLD"].set_index("student_id")
    schools = synthetic_students["school_id"]
    folds_per_school = sub.join(schools).groupby("school_id")["fold"].nunique()
    assert (folds_per_school == 1).all()


def test_missing_metadata_reports_fallback(synthetic_students, cfg):
    meta = synthetic_students.copy()
    meta["school_id"] = np.nan
    assign, status = make_cv_assignments(meta, cfg)
    assert status["CV_B_SCHOOL_GROUP_KFOLD"]["status"].startswith("UNAVAILABLE")
    assert "CV_B_SCHOOL_GROUP_KFOLD" not in set(assign["scheme"])
