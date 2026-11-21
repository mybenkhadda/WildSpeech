"""P19 — reference-conditioned speech modelling: each recording is compared with training-fold exemplars of
correct responses to the same expected word/sentence (DTW on CMVN-MFCC and log-mel sequences, posterior and
embedding similarity), with nested cross-fitting so a student's own clips never shape their reference."""
