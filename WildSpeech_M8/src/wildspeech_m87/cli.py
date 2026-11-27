"""WildSpeech M8.7 CLI: preflight, phonetic cache, word, QA, A00–A16, export.

Examples from repository:
  wildspeech-m87 --stage preflight --drive-root /content/drive/MyDrive
  wildspeech-m87 --stage extract --acknowledge-license-review --train-tar ... --test-tar ...
  wildspeech-m87 --stage word --acknowledge-license-review --train-tar ... --test-tar ...
  wildspeech-m87 --stage evaluate --mode SCREEN
  wildspeech-m87 --stage evaluate --mode FULL
  wildspeech-m87 --stage qa-template
  wildspeech-m87 --stage qa-review --qa-csv ...
  wildspeech-m87 --stage submit
Audio never goes to a remote inference API. All source artifacts and
human QA files remain in user's private Drive.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from wildspeech_m84.data import load_official,locate_data_root
from .contracts import Config,clip_inventory,validate_baselines,write_provenance

def _parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("preflight","extract","word","evaluate",
         "qa-template","qa-review","stability","submit","all"),default="preflight")
    p.add_argument("--drive-root",default="/content/drive/MyDrive")
    p.add_argument("--data-root",default="")
    p.add_argument("--cache-root",default="")
    p.add_argument("--phase0-root",default="")
    p.add_argument("--m86-root",default="")
    p.add_argument("--train-tar",default="")
    p.add_argument("--test-tar",default="")
    p.add_argument("--train-audio-dir",default="")
    p.add_argument("--test-audio-dir",default="")
    p.add_argument("--phonetic-revision",default="")
    p.add_argument("--word-revision",default="")
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--shard-size",type=int,default=48)
    p.add_argument("--max-clips",type=int,default=0)
    p.add_argument("--mode",choices=("SMOKE","SCREEN","FULL"),default="SCREEN")
    p.add_argument("--vad-method",choices=("silero","energy"),default="silero")
    p.add_argument("--qa-csv",default="")
    p.add_argument("--rasch-iters",type=int,default=7)
    p.add_argument("--no-word",action="store_true")
    p.add_argument("--acknowledge-license-review",action="store_true")
    p.add_argument("--allow-synthetic-folds",action="store_true")
    return p

def run(args):
    config=Config(
        drive_root=args.drive_root,data_root=args.data_root,
        output_root=args.cache_root,
        train_tar=args.train_tar,test_tar=args.test_tar,
        train_audio_dir=args.train_audio_dir,test_audio_dir=args.test_audio_dir,
        phonetic_revision=args.phonetic_revision,word_revision=args.word_revision,
        acknowledge_license_review=args.acknowledge_license_review,
        batch_size=args.batch_size,mode=args.mode)
    drive=Path(args.drive_root)
    root=config.output()
    p0=Path(args.phase0_root) if args.phase0_root else drive/"WildSpeech_M84"/"phase0"
    m86=Path(args.m86_root) if args.m86_root else drive/"WildSpeech_M86"/"m0pp"/"full"
    data=load_official(locate_data_root(drive,args.data_root))
    folds,ref,inventory,anchors=validate_baselines(
        data,p0,m86 if args.stage not in ("preflight","extract","word","qa-template",
                         "qa-review","stability") else None,
        official=not args.allow_synthetic_folds)
    root.mkdir(parents=True,exist_ok=True)
    from wildspeech_m84.data import atomic_json
    atomic_json(root/"manifests"/"preflight_PRIVATE.json",anchors)
    if args.stage=="preflight":
        print(json.dumps(anchors,indent=2));return anchors
    if args.stage in ("extract","word","all","stability"):
        config.validate()
        write_provenance(config,root)
    if args.stage in ("extract","all"):
        from .gpu_cache import extract
        res=extract(config,inventory,max_clips=args.max_clips,
                    shard_size=args.shard_size,vad_method=args.vad_method)
        print("[M87] Phonetic cache:",res)
        if args.stage=="extract":return res
    if args.stage=="word":
        from .gpu_cache import extract_words
        return extract_words(config,inventory,max_clips=args.max_clips,
                             sentence_only=not args.no_word)
    if args.stage=="qa-template":
        from .gpu_cache import load_features
        from .localization_qa import create_template
        path=root/"qa"/"training_qa_template_PRIVATE.csv"
        result=create_template(data,inventory,load_features(root),path)
        print("Training-only QA template:",path,"rows:",len(result))
        return str(path)
    if args.stage=="qa-review":
        from .localization_qa import evaluate_qa
        path=Path(args.qa_csv) if args.qa_csv else root/"qa"/"training_qa_template_PRIVATE.csv"
        return evaluate_qa(path,root/"qa")
    if args.stage=="stability":
        from .gpu_cache import create_sources
        from .phonetic_asr import PhoneticASR
        from .targets import TargetRegistry,normalize_text
        from .stability import run_stability
        source=create_sources(config,root)["train"]
        model=PhoneticASR(revision=config.phonetic_revision,layers=(6,))
        reg=TargetRegistry(model.processor.tokenizer).make(inventory[inventory.split=="train"])
        mapping={(r.task,normalize_text(r.expected_text)):r.ids
                 for r in reg.itertuples(index=False) if r.valid}
        return run_stability(inventory,source,model,mapping,
                             root/"stability",max_clips=min(100,args.max_clips or 45))
    if args.stage in ("evaluate","all"):
        from .evaluation import evaluate
        evalmode="SCREEN" if config.mode=="SMOKE" else config.mode
        result=evaluate(drive_root=drive,data_root=args.data_root,
                        cache_root=root,phase0_root=p0,m86_root=m86,
                        mode=evalmode,rasch_iters=args.rasch_iters,
                        official=not args.allow_synthetic_folds)
        print("[M87] Result:",result["decision"],"winner:",result["winner"])
        return result
    if args.stage=="submit":
        from .submission import export_if_eligible
        return export_if_eligible(drive_root=drive,cache_root=root,
                 eval_root=root/"evaluations"/"full",
                 data_root=args.data_root,rasch_iters=args.rasch_iters)
    raise ValueError("Unrecognized stage")

def main():
    args=_parser().parse_args()
    run(args)
if __name__=="__main__":main()
