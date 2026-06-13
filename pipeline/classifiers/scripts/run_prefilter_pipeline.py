r"""Build a BERT-ready candidate parquet with the legacy prefilter classifiers.

Example from the project root:

    python classifiers/scripts/run_prefilter_pipeline.py ^
      --input corpus_test.parquet ^
      --output data/prefilter/bert_candidates.parquet ^
      --report data/prefilter/prefilter_report.json ^
      --work-dir data/prefilter/work ^
      --topics-python C:\path\to\topics311\python.exe ^
      --econom-python C:\path\to\econom_extractor\python.exe ^
      --overwrite

The public CLI runs the full pipeline. Hidden internal modes are used by this
same script as subprocess workers so topics and econom can run in different
Python environments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LABEL_COLUMNS_TO_DROP = [
    "relevant",
    "t1_relevant",
    "t2_relevant",
    "t3_relevant",
    "t4_relevant",
    "t5_relevant",
    "annotated",
    "reasoning",
    "source",
]

TOPICS_ORDER = [
    "Защита прав потребителей",
    "ДКП",
    "Финансовый рынок",
    "НБП",
    "Финтех",
    "Геополитика",
    "НДО",
]
BINARY_TOPICS = [
    "Защита прав потребителей",
    "ДКП",
    "Финансовый рынок",
    "Финтех",
    "Геополитика",
    "НДО",
]

TOO_SHORT_THRESHOLD = 20
DEFAULT_BATCH_SIZE = 5_000
NBP_FALLBACK_THRESHOLD = 2.8560
INTERNAL_SOURCE_ROW_ID = "_prefilter_source_row_id"

BERT_MINIMUM_COLUMNS = [
    "post_uid",
    "channel_id",
    "channel_name",
    "channel_type",
    "message_id",
    "message_type",
    "post_date",
    "post_date_unixtime",
    "edited",
    "edited_unixtime",
    "from_id",
    "text",
    "text_entities",
    "reactions_total",
    "has_reactions",
    "reactions",
    "has_media",
    "poll",
    "inline_bot_buttons",
    "source_file",
    "text_hash",
]
PREFILTER_COLUMNS = [
    "prefilter_topic_any",
    "prefilter_nbp",
    "prefilter_is_econom",
    "prefilter_union",
    "prefilter_source",
]
ECONOM_PROBA_COLUMNS = ["p_econom"] + [f"p_c{i}" for i in range(6)]
ECONOM_DEBUG_COLUMNS = ["econom_class", "is_econom"] + ECONOM_PROBA_COLUMNS
WORK_FILE_NAMES = [
    "input_for_prefilter_clean.parquet",
    "input_prepared.parquet",
    "row_keys.parquet",
    "unique_lemmas.parquet",
    "unique_pred_topics.parquet",
    "unique_pred_econom.parquet",
    "full_prefilter_labeled.parquet",
]


@dataclass(frozen=True)
class ModelPaths:
    classifiers_dir: Path
    lemmatizer: Path
    binary_root: Path
    binary_src: Path
    topics_svc: Path
    svc_root: Path
    svc_src: Path
    nbp_stacker: Path
    nbp_predictor_dir: Path
    nbp_predictor: Path
    econom_tfidf: Path
    econom_xgboost: Path
    econom_stop_words: Path
    econom_trash_phrases: Path


@dataclass(frozen=True)
class ResolvedPaths:
    input: Path
    output: Path
    report: Optional[Path]
    work_dir: Path
    logs_dir: Path
    script: Path
    topics_python: Path
    econom_python: Path
    models: ModelPaths


def topic_col(topic: str) -> str:
    return "topic_" + topic.replace(" ", "_")


def score_col(topic: str) -> str:
    return "score_" + topic.replace(" ", "_")


def topic_columns() -> List[str]:
    return [topic_col(t) for t in TOPICS_ORDER]


def score_columns() -> List[str]:
    return [score_col(t) for t in TOPICS_ORDER]


def debug_columns() -> List[str]:
    return topic_columns() + score_columns() + ECONOM_DEBUG_COLUMNS + ["lemmas", "text_too_short"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy topic/econom prefilters and write a BERT-ready "
            "parquet with candidate Telegram posts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Two-env example (PowerShell):\n"
            "  python classifiers/scripts/run_prefilter_pipeline.py `\n"
            "    --input corpus_test.parquet `\n"
            "    --output data/prefilter/bert_candidates.parquet `\n"
            "    --report data/prefilter/prefilter_report.json `\n"
            "    --work-dir data/prefilter/work `\n"
            "    --topics-python C:\\path\\to\\topics311\\python.exe `\n"
            "    --econom-python C:\\path\\to\\econom_extractor\\python.exe `\n"
            "    --overwrite"
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Input parquet path.")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--work-dir", type=Path, default=Path("data/prefilter/work"))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Read only the first N rows for a test run.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output/report.")
    parser.add_argument(
        "--include-service-posts",
        action="store_true",
        help='Do not filter to message_type == "message".',
    )
    parser.add_argument(
        "--keep-debug-columns",
        action="store_true",
        help="Keep topic_*, score_*, econom_class, p_*, lemmas and text_too_short in output.",
    )
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate work files.")
    parser.add_argument(
        "--keep-labels-in-output",
        action="store_true",
        help="Debug only: keep BERT target/source label columns in final output.",
    )
    parser.add_argument("--topics-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--econom-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--classifiers-dir",
        type=Path,
        help="Path to the classifiers directory. Auto-detected by default.",
    )
    parser.add_argument("--internal-mode", choices=["lemmatize", "topics", "econom"], help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def setup_logging(log_file: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def find_executable(executable: Path) -> Optional[str]:
    value = str(executable).strip()
    if not value:
        return None

    has_separator = any(sep and sep in value for sep in (os.sep, os.altsep))
    if executable.is_absolute() or has_separator:
        candidate = executable.expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        return None

    return shutil.which(value)


def validate_python_executable(name: str, executable: Path) -> str:
    resolved = find_executable(executable)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"{name} Python executable not found: {executable}\n"
        f"Pass a real python.exe path for --{name}-python, or omit the flag to use the current interpreter."
    )


def resolve_paths(args: argparse.Namespace) -> ResolvedPaths:
    script = Path(__file__).resolve()
    classifiers_dir = args.classifiers_dir.resolve() if args.classifiers_dir else script.parents[1]
    topics_dir = classifiers_dir / "topics"
    binary_root = topics_dir / "binary_per_topic_classifier"
    svc_root = topics_dir / "svc_focused_v2"
    econom_model_dir = classifiers_dir / "econom" / "models" / "class_model"
    models = ModelPaths(
        classifiers_dir=classifiers_dir,
        lemmatizer=topics_dir / "lemmatizer.py",
        binary_root=binary_root,
        binary_src=binary_root / "src",
        topics_svc=binary_root / "results" / "svc_per_topic_pipeline.joblib",
        svc_root=svc_root,
        svc_src=svc_root / "src",
        nbp_stacker=svc_root / "results" / "nbp_v3" / "nbp_stacker.joblib",
        nbp_predictor_dir=svc_root / "results" / "nbp_v3",
        nbp_predictor=svc_root / "results" / "nbp_v3" / "nbp_predictor.py",
        econom_tfidf=econom_model_dir / "tfidf.pkl",
        econom_xgboost=econom_model_dir / "xgboost_6classes.pkl",
        econom_stop_words=econom_model_dir / "stop_words.pkl",
        econom_trash_phrases=econom_model_dir / "trash_phrases.pkl",
    )
    work_dir = args.work_dir.resolve()
    return ResolvedPaths(
        input=args.input.resolve(),
        output=args.output.resolve(),
        report=args.report.resolve() if args.report else None,
        work_dir=work_dir,
        logs_dir=work_dir / "logs",
        script=script,
        topics_python=args.topics_python,
        econom_python=args.econom_python,
        models=models,
    )


def validate_paths(args: argparse.Namespace, paths: ResolvedPaths) -> None:
    if not paths.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {paths.input}")
    if paths.input.resolve() == paths.output.resolve():
        raise ValueError("Output path must differ from input path; the source parquet is never modified.")
    if paths.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {paths.output}")
    if paths.report and paths.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists, pass --overwrite to replace it: {paths.report}")

    required_paths = {
        "classifiers_dir": paths.models.classifiers_dir,
        "lemmatizer": paths.models.lemmatizer,
        "binary_root": paths.models.binary_root,
        "binary_src": paths.models.binary_src,
        "topics_svc": paths.models.topics_svc,
        "svc_root": paths.models.svc_root,
        "svc_src": paths.models.svc_src,
        "nbp_stacker": paths.models.nbp_stacker,
        "nbp_predictor_dir": paths.models.nbp_predictor_dir,
        "nbp_predictor": paths.models.nbp_predictor,
        "econom_tfidf": paths.models.econom_tfidf,
        "econom_xgboost": paths.models.econom_xgboost,
        "econom_stop_words": paths.models.econom_stop_words,
        "econom_trash_phrases": paths.models.econom_trash_phrases,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required classifier files are missing:\n" + "\n".join(missing))

    topics_python = validate_python_executable("topics", paths.topics_python)
    econom_python = validate_python_executable("econom", paths.econom_python)
    logging.info("Topics Python: %s", topics_python)
    logging.info("Econom Python: %s", econom_python)

    paths.output.parent.mkdir(parents=True, exist_ok=True)
    if paths.report:
        paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for the orchestrator. Run with the topics311 environment "
            "or another environment that has pandas and pyarrow."
        ) from exc
    return pd


def load_input_frame(args: argparse.Namespace, paths: ResolvedPaths) -> Any:
    pd = import_pandas()
    logging.info("Reading input parquet: %s", paths.input)
    df = pd.read_parquet(paths.input)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative.")
        df = df.head(args.limit).copy()
        logging.info("Applied --limit=%s, rows now %s", args.limit, len(df))
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' was not found in input parquet.")
    return df


def prepare_input_copy_without_labels(df: Any, paths: ResolvedPaths) -> Tuple[Any, Any, List[str]]:
    pd = import_pandas()
    label_columns = [c for c in LABEL_COLUMNS_TO_DROP if c in df.columns]
    label_frame = pd.DataFrame({INTERNAL_SOURCE_ROW_ID: range(len(df))})
    for col in label_columns:
        label_frame[col] = df[col].values

    clean_df = df.drop(columns=label_columns, errors="ignore").copy()
    clean_path = paths.work_dir / "input_for_prefilter_clean.parquet"
    clean_df.to_parquet(clean_path, index=False, compression="zstd")
    logging.info("Saved cleaned prefilter input without labels: %s", clean_path)

    clean_df.insert(0, INTERNAL_SOURCE_ROW_ID, range(len(clean_df)))
    return clean_df, label_frame, label_columns


def blake2b_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def normalize_string_series(series: Any) -> Any:
    return series.fillna("").astype(str).str.strip()


def build_text_hash(df: Any, text_col: str) -> Any:
    pd = import_pandas()
    if "text_hash" in df.columns:
        hashes = normalize_string_series(df["text_hash"])
    else:
        hashes = pd.Series([""] * len(df), index=df.index, dtype="object")
    missing = hashes.eq("")
    if bool(missing.any()):
        hashes.loc[missing] = df.loc[missing, text_col].map(blake2b_text)
    df["text_hash"] = hashes.astype(str)
    return df


def build_post_uid(df: Any) -> Any:
    pd = import_pandas()
    if "post_uid" in df.columns:
        existing_uid = normalize_string_series(df["post_uid"])
    else:
        existing_uid = pd.Series([""] * len(df), index=df.index, dtype="object")

    if {"channel_id", "message_id"}.issubset(df.columns):
        channel = normalize_string_series(df["channel_id"])
        message = normalize_string_series(df["message_id"])
        uid = channel + "_" + message
        missing = channel.eq("") | message.eq("")
        fallback = existing_uid.mask(existing_uid.eq(""), df["text_hash"])
        uid.loc[missing] = fallback.loc[missing]
    else:
        uid = existing_uid.mask(existing_uid.eq(""), df["text_hash"])

    df["post_uid"] = uid.astype(str)
    return df


def filter_and_clean_text(
    args: argparse.Namespace,
    clean_df: Any,
    paths: ResolvedPaths,
) -> Tuple[Any, Dict[str, int]]:
    df = clean_df.copy()
    df[args.text_col] = normalize_string_series(df[args.text_col])
    if args.text_col != "text":
        df["text"] = df[args.text_col]

    if "message_type" in df.columns and not args.include_service_posts:
        df = df[df["message_type"].astype(str).eq("message")].copy()
    rows_after_message_filter = int(len(df))

    df = df[df[args.text_col].ne("")].copy()
    rows_after_text_filter = int(len(df))

    df = build_text_hash(df, args.text_col)
    df = build_post_uid(df)
    df = df.reset_index(drop=True)

    prepared_path = paths.work_dir / "input_prepared.parquet"
    df.to_parquet(prepared_path, index=False, compression="zstd")
    logging.info("Saved prepared input: %s rows=%s", prepared_path, len(df))

    row_keys_cols = ["post_uid", "channel_id", "message_id", "text_hash"]
    row_keys = df[[c for c in row_keys_cols if c in df.columns]].copy()
    row_keys.insert(0, "row_id", range(len(row_keys)))
    row_keys.to_parquet(paths.work_dir / "row_keys.parquet", index=False, compression="zstd")

    return df, {
        "rows_after_message_filter": rows_after_message_filter,
        "rows_after_text_filter": rows_after_text_filter,
    }


def run_subprocess(command: List[str], log_path: Path, step_name: str) -> None:
    logging.info("Running %s; log: %s", step_name, log_path)
    logging.info("Command: %s", subprocess.list2cmdline(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    logging.info("Finished %s", step_name)


def build_unique_lemmas(args: argparse.Namespace, prepared_df: Any, paths: ResolvedPaths) -> Tuple[Path, Dict[str, int]]:
    pd = import_pandas()
    columns = ["text_hash", "text"]
    unique = prepared_df.drop_duplicates("text_hash", keep="first")[columns].copy()
    if "lemmas" in prepared_df.columns:
        lemma_map = prepared_df.drop_duplicates("text_hash", keep="first")[["text_hash", "lemmas"]]
        unique = unique.merge(lemma_map, on="text_hash", how="left")
    else:
        unique["lemmas"] = ""

    unique_lemmas_path = paths.work_dir / "unique_lemmas.parquet"
    unique.to_parquet(unique_lemmas_path, index=False, compression="zstd")

    command = [
        str(paths.topics_python),
        str(paths.script),
        "--internal-mode",
        "lemmatize",
        "--input",
        str(unique_lemmas_path),
        "--output",
        str(unique_lemmas_path),
        "--classifiers-dir",
        str(paths.models.classifiers_dir),
        "--batch-size",
        str(args.batch_size),
        "--overwrite",
    ]
    run_subprocess(command, paths.logs_dir / "step1_lemmatize.log", "lemmatization")

    unique_done = pd.read_parquet(unique_lemmas_path)
    too_short = int(unique_done["text_too_short"].sum()) if "text_too_short" in unique_done.columns else 0
    return unique_lemmas_path, {
        "unique_texts_total": int(len(unique_done)),
        "unique_texts_too_short": too_short,
    }


def run_topics_inference(args: argparse.Namespace, paths: ResolvedPaths, unique_lemmas_path: Path) -> Path:
    output = paths.work_dir / "unique_pred_topics.parquet"
    command = [
        str(paths.topics_python),
        str(paths.script),
        "--internal-mode",
        "topics",
        "--input",
        str(unique_lemmas_path),
        "--output",
        str(output),
        "--classifiers-dir",
        str(paths.models.classifiers_dir),
        "--batch-size",
        str(args.batch_size),
        "--overwrite",
    ]
    run_subprocess(command, paths.logs_dir / "step2_topics.log", "topics inference")
    return output


def run_econom_inference(args: argparse.Namespace, paths: ResolvedPaths, unique_lemmas_path: Path) -> Path:
    output = paths.work_dir / "unique_pred_econom.parquet"
    command = [
        str(paths.econom_python),
        str(paths.script),
        "--internal-mode",
        "econom",
        "--input",
        str(unique_lemmas_path),
        "--output",
        str(output),
        "--classifiers-dir",
        str(paths.models.classifiers_dir),
        "--batch-size",
        str(args.batch_size),
        "--overwrite",
    ]
    run_subprocess(command, paths.logs_dir / "step3_econom.log", "econom inference")
    return output


def assemble_predictions(
    prepared_df: Any,
    unique_lemmas_path: Path,
    topics_path: Path,
    econom_path: Path,
    paths: ResolvedPaths,
) -> Any:
    pd = import_pandas()
    df = prepared_df.copy()
    unique_lemmas = pd.read_parquet(unique_lemmas_path)
    lemma_cols = [c for c in ["text_hash", "lemmas", "text_too_short"] if c in unique_lemmas.columns]
    df = df.merge(unique_lemmas[lemma_cols], on="text_hash", how="left")

    topics = pd.read_parquet(topics_path)
    econom = pd.read_parquet(econom_path)
    df = df.merge(topics, on="text_hash", how="left")
    df = df.merge(econom, on="text_hash", how="left")
    df = build_prefilter_columns(df)

    full_path = paths.work_dir / "full_prefilter_labeled.parquet"
    df.to_parquet(full_path, index=False, compression="zstd")
    logging.info("Saved full prefilter labels: %s rows=%s", full_path, len(df))
    return df


def build_prefilter_columns(df: Any) -> Any:
    for col in topic_columns():
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int8")
    for col in score_columns():
        if col not in df.columns:
            df[col] = math.nan
        df[col] = df[col].astype("float32")

    if "is_econom" not in df.columns:
        df["is_econom"] = 0
    if "econom_class" not in df.columns:
        df["econom_class"] = -1
    df["is_econom"] = df["is_econom"].fillna(0).astype("int8")
    df["econom_class"] = df["econom_class"].fillna(-1).astype("int8")

    df["prefilter_nbp"] = df[topic_col("НБП")].fillna(0).astype("int8")
    df["prefilter_topic_any"] = (df[topic_columns()].sum(axis=1) > 0).astype("int8")
    df["prefilter_is_econom"] = df["is_econom"].fillna(0).astype("int8")
    df["prefilter_union"] = (
        (df["prefilter_topic_any"] == 1) | (df["prefilter_is_econom"] == 1)
    ).astype("int8")

    both = (df["prefilter_topic_any"] == 1) & (df["prefilter_is_econom"] == 1)
    topics_only = (df["prefilter_topic_any"] == 1) & (df["prefilter_is_econom"] == 0)
    econom_only = (df["prefilter_topic_any"] == 0) & (df["prefilter_is_econom"] == 1)
    df["prefilter_source"] = ""
    df.loc[topics_only, "prefilter_source"] = "topics"
    df.loc[econom_only, "prefilter_source"] = "econom"
    df.loc[both, "prefilter_source"] = "topics+econom"
    return df


def select_bert_candidates(df: Any) -> Tuple[Any, Dict[str, int]]:
    selected = df[df["prefilter_union"] == 1].copy()
    before_dedup = int(len(selected))
    selected = selected.drop_duplicates("post_uid", keep="first").copy()
    duplicates_removed = before_dedup - int(len(selected))
    return selected, {
        "selected_by_topics": int((df["prefilter_topic_any"] == 1).sum()),
        "selected_by_nbp": int((df["prefilter_nbp"] == 1).sum()),
        "selected_by_econom": int((df["prefilter_is_econom"] == 1).sum()),
        "selected_by_union": before_dedup,
        "candidates_total": int(len(selected)),
        "duplicates_removed": duplicates_removed,
    }


def attach_debug_labels(candidates: Any, label_frame: Any) -> Any:
    label_cols = [c for c in LABEL_COLUMNS_TO_DROP if c in label_frame.columns]
    if not label_cols:
        return candidates
    return candidates.merge(label_frame[[INTERNAL_SOURCE_ROW_ID] + label_cols], on=INTERNAL_SOURCE_ROW_ID, how="left")


def build_bert_output(args: argparse.Namespace, candidates: Any, label_frame: Any) -> Any:
    if args.keep_labels_in_output:
        candidates = attach_debug_labels(candidates, label_frame)

    internal_cols = {INTERNAL_SOURCE_ROW_ID}
    debug_set = set(debug_columns())
    label_set = set(LABEL_COLUMNS_TO_DROP)
    prefilter_set = set(PREFILTER_COLUMNS)

    base_cols = [c for c in BERT_MINIMUM_COLUMNS if c in candidates.columns]
    passthrough_cols = [
        c
        for c in candidates.columns
        if c not in base_cols
        and c not in internal_cols
        and c not in debug_set
        and c not in label_set
        and c not in prefilter_set
    ]
    output_cols = base_cols + passthrough_cols + [c for c in PREFILTER_COLUMNS if c in candidates.columns]

    if args.keep_debug_columns:
        output_cols.extend([c for c in debug_columns() if c in candidates.columns and c not in output_cols])
    if args.keep_labels_in_output:
        output_cols.extend([c for c in LABEL_COLUMNS_TO_DROP if c in candidates.columns and c not in output_cols])

    output_df = candidates[output_cols].copy()
    if not args.keep_labels_in_output:
        output_df = output_df.drop(columns=LABEL_COLUMNS_TO_DROP, errors="ignore")
        for col in LABEL_COLUMNS_TO_DROP:
            assert col not in output_df.columns, f"Label column leaked into BERT-ready output: {col}"
    if "text" not in output_df.columns:
        raise ValueError("BERT-ready output is missing required text column 'text'.")
    return output_df


def truthy_series(series: Any) -> Any:
    pd = import_pandas()
    if str(series.dtype) == "bool":
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.notna().any()):
        return numeric.fillna(0) != 0
    lowered = normalize_string_series(series).str.lower()
    return lowered.isin({"1", "true", "yes", "y", "да", "истина"})


def build_label_sanity_check(original_df: Any, selected_source_ids: Iterable[int]) -> Dict[str, Any]:
    pd = import_pandas()
    target_cols = [c for c in ["relevant", "t1_relevant", "t2_relevant", "t3_relevant", "t4_relevant", "t5_relevant"] if c in original_df.columns]
    if not target_cols:
        return {"has_bert_labels": False}

    selected_ids = set(int(x) for x in selected_source_ids)
    selected_mask = pd.Series([idx in selected_ids for idx in range(len(original_df))], index=original_df.index)

    if "relevant" in original_df.columns:
        relevant_mask = truthy_series(original_df["relevant"])
    else:
        relevant_mask = pd.Series(False, index=original_df.index)
        for col in target_cols:
            relevant_mask = relevant_mask | truthy_series(original_df[col])

    relevant_total = int(relevant_mask.sum())
    relevant_selected = int((relevant_mask & selected_mask).sum())
    selected_total = int(selected_mask.sum())
    sanity: Dict[str, Any] = {
        "has_bert_labels": True,
        "bert_relevant_total": relevant_total,
        "bert_relevant_selected": relevant_selected,
        "bert_relevant_missed": max(0, relevant_total - relevant_selected),
        "prefilter_recall_on_labeled_test": safe_divide(relevant_selected, relevant_total),
        "bert_relevant_selected_share": safe_divide(relevant_selected, selected_total),
    }

    if "annotated" in original_df.columns:
        annotated_mask = truthy_series(original_df["annotated"])
        annotated_relevant = annotated_mask & relevant_mask
        annotated_relevant_selected = annotated_relevant & selected_mask
        annotated_relevant_total = int(annotated_relevant.sum())
        sanity.update(
            {
                "annotated_total": int(annotated_mask.sum()),
                "annotated_relevant_total": annotated_relevant_total,
                "annotated_relevant_selected": int(annotated_relevant_selected.sum()),
                "prefilter_recall_on_annotated_relevant": safe_divide(
                    int(annotated_relevant_selected.sum()),
                    annotated_relevant_total,
                ),
            }
        )
    return sanity


def safe_divide(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def read_bert_config(paths: ResolvedPaths) -> Dict[str, Any]:
    candidates = [
        paths.models.classifiers_dir.parent / "model" / "topic_model_config.json",
        paths.models.classifiers_dir.parent / "model" / "topic_model_config (1).json",
        paths.models.classifiers_dir / "model" / "topic_model_config.json",
        paths.models.classifiers_dir / "model" / "topic_model_config (1).json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {
                    "config_path": display_path(path),
                    "text_col": data.get("text_col", "text"),
                    "max_length": data.get("max_length", 512),
                    "labels": data.get("labels", []),
                    "thresholds": data.get("thresholds", {}),
                }
            except Exception as exc:  # pragma: no cover - report-only best effort
                logging.warning("Could not read BERT config %s: %s", path, exc)
    return {"text_col": "text", "max_length": 512, "labels": [], "thresholds": {}}


def build_report(
    args: argparse.Namespace,
    paths: ResolvedPaths,
    stats: Dict[str, int],
    output_df: Any,
    label_sanity_check: Dict[str, Any],
) -> Dict[str, Any]:
    contains_label_columns = any(col in output_df.columns for col in LABEL_COLUMNS_TO_DROP)
    rows_after_text_filter = stats.get("rows_after_text_filter", 0)
    bert_config = read_bert_config(paths)
    return {
        "input": str(paths.input),
        "output": str(paths.output),
        "work_dir": str(paths.work_dir),
        "rows_input_total": stats.get("rows_input_total", 0),
        "rows_after_message_filter": stats.get("rows_after_message_filter", 0),
        "rows_after_text_filter": rows_after_text_filter,
        "unique_texts_total": stats.get("unique_texts_total", 0),
        "unique_texts_too_short": stats.get("unique_texts_too_short", 0),
        "selected_by_topics": stats.get("selected_by_topics", 0),
        "selected_by_nbp": stats.get("selected_by_nbp", 0),
        "selected_by_econom": stats.get("selected_by_econom", 0),
        "selected_by_union": stats.get("selected_by_union", 0),
        "candidates_total": stats.get("candidates_total", 0),
        "share_selected": safe_divide(stats.get("selected_by_union", 0), rows_after_text_filter),
        "duplicates_removed": stats.get("duplicates_removed", 0),
        "models": {
            "topics_svc": display_path(paths.models.topics_svc),
            "nbp_stacker": display_path(paths.models.nbp_stacker),
            "econom_tfidf": display_path(paths.models.econom_tfidf),
            "econom_xgboost": display_path(paths.models.econom_xgboost),
        },
        "environments": {
            "topics_python": str(paths.topics_python),
            "econom_python": str(paths.econom_python),
        },
        "bert_input_schema": {
            "dropped_label_columns": LABEL_COLUMNS_TO_DROP,
            "input_text_col": args.text_col,
            "text_col": "text",
            "is_bert_ready": ("text" in output_df.columns) and not contains_label_columns,
            "contains_label_columns": contains_label_columns,
            "bert_config": bert_config,
        },
        "label_sanity_check": label_sanity_check,
    }


def write_outputs(output_df: Any, report: Dict[str, Any], paths: ResolvedPaths) -> None:
    output_df.to_parquet(paths.output, index=False, compression="zstd")
    logging.info("Wrote BERT-ready candidates: %s rows=%s", paths.output, len(output_df))
    if paths.report:
        paths.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote report: %s", paths.report)


def cleanup_work_files(paths: ResolvedPaths) -> None:
    for name in WORK_FILE_NAMES:
        path = paths.work_dir / name
        if path.exists():
            path.unlink()
    logging.info("Removed intermediate parquet files from %s (logs kept).", paths.work_dir)


def load_lemmatizer_module(lemmatizer_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("prefilter_lemmatizer_bundle", str(lemmatizer_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load lemmatizer from {lemmatizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def internal_lemmatize(args: argparse.Namespace, paths: ResolvedPaths) -> None:
    pd = import_pandas()
    df = pd.read_parquet(paths.input)
    if "text_hash" not in df.columns or "text" not in df.columns:
        raise ValueError("Internal lemmatize input must contain text_hash and text columns.")
    if "lemmas" not in df.columns:
        df["lemmas"] = ""

    df["text"] = normalize_string_series(df["text"])
    df["lemmas"] = normalize_string_series(df["lemmas"])
    df["text_too_short"] = df["text"].str.len() < TOO_SHORT_THRESHOLD
    df.loc[df["text_too_short"], "lemmas"] = ""

    missing = (~df["text_too_short"]) & df["lemmas"].eq("")
    n_missing = int(missing.sum())
    logging.info("Unique texts=%s, lemmatization needed=%s", len(df), n_missing)
    if n_missing:
        lemmatizer = load_lemmatizer_module(paths.models.lemmatizer)
        morph = lemmatizer.get_morph()
        stopwords = lemmatizer.get_stopwords()
        idxs = list(df.index[missing])
        started = time.monotonic()
        for done, idx in enumerate(idxs, 1):
            df.at[idx, "lemmas"] = lemmatizer.lemmatize_text(df.at[idx, "text"], morph=morph, stopwords=stopwords)
            if done % max(1, args.batch_size) == 0 or done == n_missing:
                elapsed = time.monotonic() - started
                rate = done / max(elapsed, 0.001)
                logging.info("lemmatized %s/%s unique texts (%.0f text/s)", done, n_missing, rate)

    out_cols = ["text_hash", "text", "lemmas", "text_too_short"]
    write_parquet_atomic(df[out_cols], paths.output)


def load_topic_models(paths: ResolvedPaths) -> Tuple[Any, Any]:
    import joblib

    sys.path.insert(0, str(paths.models.binary_root))
    sys.path.insert(0, str(paths.models.binary_src))
    logging.info("Loading binary SVC mix: %s", paths.models.topics_svc)
    mix = joblib.load(paths.models.topics_svc)

    sys.path.insert(0, str(paths.models.svc_root))
    sys.path.insert(0, str(paths.models.nbp_predictor_dir))
    sys.path.insert(0, str(paths.models.svc_src))
    logging.info("Loading NBP stacker: %s", paths.models.nbp_stacker)
    nbp = joblib.load(paths.models.nbp_stacker)
    return mix, nbp


def internal_topics(args: argparse.Namespace, paths: ResolvedPaths) -> None:
    pd = import_pandas()
    import numpy as np

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    warnings.simplefilter("ignore")

    df = pd.read_parquet(paths.input)
    if "text_hash" not in df.columns or "lemmas" not in df.columns:
        raise ValueError("Internal topics input must contain text_hash and lemmas columns.")

    result = pd.DataFrame({"text_hash": df["text_hash"].astype(str)})
    for col in topic_columns():
        result[col] = np.zeros(len(df), dtype="int8")
    for col in score_columns():
        result[col] = np.full(len(df), np.nan, dtype="float32")

    if "text_too_short" in df.columns:
        active_mask = ~df["text_too_short"].fillna(False).astype(bool)
    else:
        active_mask = normalize_string_series(df["lemmas"]).ne("")
    if not bool(active_mask.any()):
        write_parquet_atomic(result, paths.output)
        return

    mix, nbp = load_topic_models(paths)
    active = df.loc[active_mask].copy()
    lemmas = pd.Series([x or "" for x in active["lemmas"].tolist()])
    logging.info("Running topics on %s unique non-short texts", len(active))

    for topic in BINARY_TOPICS:
        model = mix[topic]
        threshold = getattr(model, "threshold", 0.0)
        scores = np.asarray(model.decision_function(lemmas), dtype="float32")
        result.loc[active.index, topic_col(topic)] = (scores >= threshold).astype("int8")
        result.loc[active.index, score_col(topic)] = scores

    nbp_scores = np.asarray(nbp.decision_function(lemmas), dtype="float32")
    if hasattr(nbp, "predict"):
        nbp_pred = np.asarray(nbp.predict(lemmas), dtype="int8")
    else:
        threshold = getattr(nbp, "threshold", NBP_FALLBACK_THRESHOLD)
        nbp_pred = (nbp_scores >= threshold).astype("int8")
    result.loc[active.index, topic_col("НБП")] = nbp_pred
    result.loc[active.index, score_col("НБП")] = nbp_scores

    write_parquet_atomic(result, paths.output)


def econom_preproc_patterns(stop_re: str, trash_re: str) -> Tuple[Any, Any, Any, Any]:
    stop_pat = re.compile(stop_re)
    trash_pat = re.compile(trash_re)
    non_text_pat = re.compile(r"[^A-Za-zА-Яа-яё\s]")
    ws_pat = re.compile(r"\s+")
    return stop_pat, trash_pat, non_text_pat, ws_pat


def clean_econom_lemma(text: Any, patterns: Tuple[Any, Any, Any, Any]) -> str:
    if not isinstance(text, str):
        return ""
    stop_pat, trash_pat, non_text_pat, ws_pat = patterns
    text = stop_pat.sub(" ", text)
    text = trash_pat.sub(" ", text)
    text = non_text_pat.sub("", text)
    return ws_pat.sub(" ", text).strip()


def patch_legacy_econom_pickle_support() -> None:
    try:
        import sklearn.metrics._scorer as scorer
    except Exception:
        return

    if not hasattr(scorer, "_PredictScorer"):
        class _PredictScorer:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Legacy sklearn scorer is not available during inference.")

        scorer._PredictScorer = _PredictScorer


def restore_legacy_tfidf_idf(tfidf: Any) -> Any:
    transformer = getattr(tfidf, "_tfidf", None)
    if transformer is None or hasattr(transformer, "idf_"):
        return tfidf
    idf_diag = getattr(transformer, "_idf_diag", None)
    if idf_diag is not None:
        transformer.idf_ = idf_diag.diagonal()
    return tfidf


def internal_econom(args: argparse.Namespace, paths: ResolvedPaths) -> None:
    pd = import_pandas()
    import numpy as np

    warnings.simplefilter("ignore")
    try:
        import xgboost as xgb_lib
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for econom inference. Run this worker with the econom_extractor environment."
        ) from exc

    df = pd.read_parquet(paths.input)
    if "text_hash" not in df.columns or "lemmas" not in df.columns:
        raise ValueError("Internal econom input must contain text_hash and lemmas columns.")

    result = pd.DataFrame({"text_hash": df["text_hash"].astype(str)})
    result["econom_class"] = np.full(len(df), -1, dtype="int8")
    result["is_econom"] = np.zeros(len(df), dtype="int8")
    for col in ECONOM_PROBA_COLUMNS:
        result[col] = np.full(len(df), np.nan, dtype="float32")

    if "text_too_short" in df.columns:
        active_mask = ~df["text_too_short"].fillna(False).astype(bool)
    else:
        active_mask = normalize_string_series(df["lemmas"]).ne("")
    active_mask = active_mask & normalize_string_series(df["lemmas"]).ne("")
    if not bool(active_mask.any()):
        write_parquet_atomic(result, paths.output)
        return

    logging.info("Loading econom models from %s", paths.models.econom_tfidf.parent)
    patch_legacy_econom_pickle_support()
    with paths.models.econom_tfidf.open("rb") as fh:
        tfidf = restore_legacy_tfidf_idf(pickle.load(fh))
    with paths.models.econom_xgboost.open("rb") as fh:
        xgb_grid = pickle.load(fh)
    with paths.models.econom_stop_words.open("rb") as fh:
        stop_re = pickle.load(fh)
    with paths.models.econom_trash_phrases.open("rb") as fh:
        trash_re = pickle.load(fh)

    patterns = econom_preproc_patterns(stop_re, trash_re)
    active = df.loc[active_mask].copy()
    cleaned = [clean_econom_lemma(x, patterns) for x in active["lemmas"].tolist()]
    matrix = tfidf.transform(cleaned)

    proba = None
    try:
        booster = xgb_grid.best_estimator_.get_booster()
        proba = booster.predict(xgb_lib.DMatrix(matrix))
        pred = proba.argmax(axis=1).astype("int8")
    except Exception as exc:
        logging.warning("Booster probability path failed, falling back to estimator predict: %s", exc)
        estimator = getattr(xgb_grid, "best_estimator_", xgb_grid)
        pred = estimator.predict(matrix).astype("int8")
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(matrix)

    result.loc[active.index, "econom_class"] = pred
    result.loc[active.index, "is_econom"] = (pred == 0).astype("int8")
    if proba is not None:
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 1:
            result.loc[active.index, "p_econom"] = proba[:, 0].astype("float32")
            for i in range(min(6, proba.shape[1])):
                result.loc[active.index, f"p_c{i}"] = proba[:, i].astype("float32")

    write_parquet_atomic(result, paths.output)


def write_parquet_atomic(df: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(output)
    logging.info("Saved %s rows=%s", output, len(df))


def run_main(args: argparse.Namespace, paths: ResolvedPaths) -> None:
    validate_paths(args, paths)
    original_df = load_input_frame(args, paths)
    stats: Dict[str, int] = {"rows_input_total": int(len(original_df))}

    clean_df, label_frame, label_columns = prepare_input_copy_without_labels(original_df, paths)
    if label_columns:
        logging.info("Input labels detected for report only: %s", ", ".join(label_columns))

    prepared_df, prep_stats = filter_and_clean_text(args, clean_df, paths)
    stats.update(prep_stats)

    unique_lemmas_path, unique_stats = build_unique_lemmas(args, prepared_df, paths)
    stats.update(unique_stats)

    topics_path = run_topics_inference(args, paths, unique_lemmas_path)
    econom_path = run_econom_inference(args, paths, unique_lemmas_path)
    full_df = assemble_predictions(prepared_df, unique_lemmas_path, topics_path, econom_path, paths)

    candidates, select_stats = select_bert_candidates(full_df)
    stats.update(select_stats)
    output_df = build_bert_output(args, candidates, label_frame)
    selected_source_ids = candidates[INTERNAL_SOURCE_ROW_ID].tolist() if INTERNAL_SOURCE_ROW_ID in candidates.columns else []
    label_sanity = build_label_sanity_check(original_df, selected_source_ids)
    report = build_report(args, paths, stats, output_df, label_sanity)
    write_outputs(output_df, report, paths)

    if not args.keep_work:
        cleanup_work_files(paths)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    setup_logging(paths.logs_dir / "prefilter_pipeline.log" if not args.internal_mode else None)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.internal_mode == "lemmatize":
        validate_internal_paths(args, paths, ["lemmatizer"])
        internal_lemmatize(args, paths)
    elif args.internal_mode == "topics":
        validate_internal_paths(args, paths, ["topics"])
        internal_topics(args, paths)
    elif args.internal_mode == "econom":
        validate_internal_paths(args, paths, ["econom"])
        internal_econom(args, paths)
    else:
        run_main(args, paths)


def validate_internal_paths(args: argparse.Namespace, paths: ResolvedPaths, groups: List[str]) -> None:
    if not paths.input.exists():
        raise FileNotFoundError(f"Internal input not found: {paths.input}")
    if paths.output.exists() and not args.overwrite:
        raise FileExistsError(f"Internal output exists, pass --overwrite: {paths.output}")
    required: Dict[str, Path] = {}
    if "lemmatizer" in groups:
        required["lemmatizer"] = paths.models.lemmatizer
    if "topics" in groups:
        required.update(
            {
                "binary_root": paths.models.binary_root,
                "binary_src": paths.models.binary_src,
                "topics_svc": paths.models.topics_svc,
                "svc_root": paths.models.svc_root,
                "svc_src": paths.models.svc_src,
                "nbp_stacker": paths.models.nbp_stacker,
                "nbp_predictor_dir": paths.models.nbp_predictor_dir,
                "nbp_predictor": paths.models.nbp_predictor,
            }
        )
    if "econom" in groups:
        required.update(
            {
                "econom_tfidf": paths.models.econom_tfidf,
                "econom_xgboost": paths.models.econom_xgboost,
                "econom_stop_words": paths.models.econom_stop_words,
                "econom_trash_phrases": paths.models.econom_trash_phrases,
            }
        )
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))
    paths.output.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
