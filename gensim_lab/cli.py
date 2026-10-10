from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    from .runtime import required_environment as runtime_environment
    required_environment = runtime_environment()
    if any(os.environ.get(key) != value for key, value in required_environment.items()):
        import subprocess
        env = os.environ.copy()
        env.update(required_environment)
        argv = [x for x in sys.argv[1:] if x != "--worker-env"]
        raise SystemExit(subprocess.call([sys.executable, "-m", "gensim_lab", *argv, "--worker-env"], env=env))
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Gensim / FastText advanced page classification lab")
    parser.add_argument("--worker-env", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--force", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--profile", choices=["smoke", "full", "ablations", "temporal", "supplemental"], default="full")
    run.add_argument("--seeds", default="42,2026,7")
    run.add_argument("--finalize", action="store_true")
    sub.add_parser("report")
    sub.add_parser("evidence")
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--tokenized", action="store_true")
    # Accept the internal flag before or after subcommand arguments.
    argv = [x for x in sys.argv[1:] if x != "--worker-env"]
    args = parser.parse_args(argv)
    if args.command == "predict":
        dispatch(args, parser)
    else:
        from .integrity import artifact_lock
        with artifact_lock():
            dispatch(args, parser)


def dispatch(args, parser):
    if args.command == "prepare":
        from .integrity import ensure_dataset
        ensure_dataset(force=args.force)
    elif args.command == "run":
        from .runner import finalize, run_ablations, run_main, run_temporal
        seeds = [int(x) for x in args.seeds.split(",")]
        if args.profile == "smoke":
            run_main([42], smoke=True)
        elif args.profile == "full":
            run_main(seeds)
        elif args.profile == "ablations":
            run_ablations()
        elif args.profile == "supplemental":
            from .supplemental import train_all, run_scales, run_chronological_native
            from .supplemental_report import report
            train_all()
            run_scales()
            run_chronological_native()
            report()
        else:
            run_temporal(seeds)
        if args.finalize:
            finalize()
    elif args.command == "report":
        from .reporting import report
        report()
    elif args.command == "evidence":
        from .evidence import run_evidence
        run_evidence()
    elif args.command == "predict":
        from .models import Bundle
        import unicodedata
        if not args.text.strip():
            parser.error("--text must not be empty")
        if args.tokenized:
            tokens = args.text.split()
        else:
            import jieba
            from .common import ARTIFACTS
            cache_path = ARTIFACTS / "tokenizer_cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            jieba.dt.tmp_dir = str(cache_path)
            tokens = jieba.lcut(unicodedata.normalize("NFKC", args.text))
        tokens = ["".join(c for c in unicodedata.normalize("NFKC", t) if unicodedata.category(c) != "Cc") for t in tokens]
        tokens = [t for t in tokens if t.strip() and not all(unicodedata.category(c)[0] in "PS" for c in t)]
        if not tokens:
            parser.error("--text must contain at least one non-punctuation token")
        doc = {"id": "input-" + hashlib.sha256(args.text.encode()).hexdigest()[:12], "tokens": tokens}
        bundle = Bundle.load(args.model.resolve())
        if bundle.method == "doc2vec" and bundle.spec.get("doc2vec_inference_seed") != "tokens_sha256_v1":
            parser.error("Legacy Doc2Vec model seeds inference from article IDs; use a rebuilt tokens_sha256_v1 model. Historical models remain readable through Bundle for reproduction.")
        labels, scores, coverage = bundle.predict([doc])
        print(json.dumps({"page_label": labels[0], "score": scores[0], "score_type": "probability" if bundle.method == "native_fasttext" else "constant" if bundle.method == "majority" else "decision_margin", "coverage": coverage, "tokenization": "provided" if args.tokenized else "jieba (different from gold corpus segmentation)"}, ensure_ascii=False, allow_nan=False))
