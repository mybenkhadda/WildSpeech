"""Compact, fold-fitted student-level representations of V3 phonetic information (P28).

student_aggregation   label-free item-conditioned family providers (ASRW/ASRP/ASRC/ASRT/ASRK), consistency (ASRS),
                      nested supervised soft correctness (ASRN), phoneme-confusion embeddings (ASRX)
embedding_pooling     pooled frozen encoder embeddings (ASRE) and nested attention-MIL pooling
compact_features      budgeted reducers (selection / PCA / PLS / reduced-rank ridge) and the provider factory
Every provider implements FoldFeatureProvider.fit_table(train_ids, target_ids): anything fitted uses training-fold
students only; supervised parts are additionally cross-fitted by student inside the training fold.
"""
