"""P17 — clip-level correctness intelligence: P(correct | audio, expected text, task) learned with
student-disjoint, nested cross-fitting, then turned into compact student-level soft-correctness features.

Classification confidence is NOT a verified measure of pronunciation quality; it is a learned summary of how
a response's acoustics resemble responses that assessors scored correct for the same kind of item.
"""
