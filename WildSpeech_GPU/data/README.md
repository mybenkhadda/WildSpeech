# data/

The competition files are **not copied** here. They are discovered automatically (read-only, never modified) by
searching `paths.data_search_roots` in `configs/default.yaml` — by default `data/raw`, `data`, `../data`, … relative to
the repository root. With the current layout they are found in `C:\Users\myben\Downloads\WildSpeech_P01\data`.

To pin a location explicitly set `paths.data_dir: "C:/path/to/data"`.

Expected files: `train_clip_metadata.csv`, `test_clip_metadata.csv`, `train_student_metadata.csv`,
`test_student_metadata.csv`, `train_labels.csv`, `submission_format.csv`, `train_audio.tar`, `test_audio.tar`.
