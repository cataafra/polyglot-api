"""
Prepare research-grade Polyglot evaluation manifests.

Raw datasets are intentionally kept outside git. This script writes WAV clips to
evaluation/corpora/ and CSV manifests to evaluation/manifests/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyglot_api.text_semantics import normalize_transcript  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "evaluation"
DEFAULT_SEED = 20260530


@dataclass(frozen=True)
class PreparedExample:
    case_id: str
    dataset: str
    split: str
    workload: str
    audio_path: Path
    source_language: str
    target_language: str
    source_text: str
    reference_text: str
    group_id: str
    speaker_id: str = "0"
    domain: str = "general"
    privacy_level: str = "public"
    expected_cache_layer: str = "miss"
    expected_reuse_allowed: bool = True
    reuse_group: str = ""
    source_clip_id: str = ""
    notes: str = ""

    def to_manifest_row(self, manifest_path: Path) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "split": self.split,
            "workload": self.workload,
            "audio_path": relative_to(self.audio_path, manifest_path.parent),
            "source_language": self.source_language,
            "target_language": self.target_language,
            "speaker_id": self.speaker_id,
            "domain": self.domain,
            "privacy_level": self.privacy_level,
            "source_text": self.source_text,
            "reference_text": self.reference_text,
            "group_id": self.group_id,
            "reuse_group": self.reuse_group or self.group_id,
            "source_clip_id": self.source_clip_id,
            "expected_cache_layer": self.expected_cache_layer,
            "expected_reuse_allowed": str(self.expected_reuse_allowed).lower(),
            "quality_required": str(bool(self.reference_text)).lower(),
            "notes": self.notes,
        }


MANIFEST_COLUMNS = [
    "case_id",
    "dataset",
    "split",
    "workload",
    "audio_path",
    "source_language",
    "target_language",
    "speaker_id",
    "domain",
    "privacy_level",
    "source_text",
    "reference_text",
    "group_id",
    "reuse_group",
    "source_clip_id",
    "expected_cache_layer",
    "expected_reuse_allowed",
    "quality_required",
    "notes",
]


def relative_to(path: Path, parent: Path) -> str:
    try:
        return str(path.resolve().relative_to(parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def write_manifest(path: Path, rows: list[PreparedExample], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_manifest_row(path))
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_id(*parts: Any, length: int = 12) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def write_wav(audio: Any, destination: Path, sample_rate: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, dict) and "array" in audio:
        sf.write(destination, audio["array"], int(audio.get("sampling_rate") or sample_rate or 16000), format="WAV")
        return
    if isinstance(audio, dict) and audio.get("bytes"):
        samples, decoded_rate = sf.read(BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        sf.write(destination, samples, int(decoded_rate or sample_rate or 16000), format="WAV")
        return
    if isinstance(audio, dict) and audio.get("path"):
        copy_or_convert_audio(Path(audio["path"]), destination)
        return
    if isinstance(audio, (str, Path)):
        copy_or_convert_audio(Path(audio), destination)
        return
    raise ValueError("unsupported audio value; expected datasets Audio dict or file path")


def copy_or_convert_audio(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav":
        shutil.copyfile(source, destination)
        return
    librosa = import_optional("librosa")
    audio, sample_rate = librosa.load(source, sr=16000, mono=True)
    sf.write(destination, audio, sample_rate, format="WAV")


def import_optional(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"{module_name} is required for this command. Install with: pip install -r requirements-eval.txt"
        ) from exc


def load_dataset(*args, **kwargs):
    datasets = import_optional("datasets")
    return datasets.load_dataset(*args, **kwargs)


def disable_audio_decode(dataset):
    if "audio" not in dataset.column_names:
        return dataset
    datasets = import_optional("datasets")
    return dataset.cast_column("audio", datasets.Audio(decode=False))


def prepare_fleurs(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    rows: list[PreparedExample] = []
    manifest_path = output_root / "manifests" / "fleurs_manifest.csv"
    language_pairs = parse_language_pairs(args.language_pairs)

    for source_config, target_config, source_language, target_language in language_pairs:
        source_ds = disable_audio_decode(load_dataset("google/fleurs", source_config, split=args.split))
        target_ds = load_dataset("google/fleurs", target_config, split=args.split)
        if "audio" in target_ds.column_names:
            target_ds = target_ds.remove_columns("audio")
        target_by_id = {example["id"]: example for example in target_ds}
        indices = list(range(len(source_ds)))
        rng.shuffle(indices)
        if args.max_samples:
            indices = indices[: args.max_samples]

        for index in indices:
            source = source_ds[index]
            target = target_by_id.get(source["id"])
            if not target:
                continue
            case = f"fleurs-{source_language}-{target_language}-{source['id']}"
            audio_path = output_root / "corpora" / "fleurs" / source_language / args.split / f"{case}.wav"
            write_wav(source["audio"], audio_path)
            rows.append(
                PreparedExample(
                    case_id=case,
                    dataset="fleurs",
                    split=args.split,
                    workload="quality",
                    audio_path=audio_path,
                    source_language=source_language,
                    target_language=target_language,
                    source_text=source.get("transcription") or source.get("raw_transcription") or "",
                    reference_text=target.get("transcription") or target.get("raw_transcription") or "",
                    group_id=f"fleurs-{source_language}-{target_language}",
                    source_clip_id=str(source["id"]),
                    notes="FLEURS parallel quality/generalization sample.",
                )
            )

    write_manifest(
        manifest_path,
        rows,
        {
            "dataset": "google/fleurs",
            "split": args.split,
            "language_pairs": args.language_pairs,
            "seed": args.seed,
            "max_samples_per_pair": args.max_samples,
            "rows": len(rows),
        },
    )
    return manifest_path


def parse_language_pairs(value: str) -> list[tuple[str, str, str, str]]:
    mapping = {
        "ro-en": ("ro_ro", "en_us", "ron", "eng"),
        "fr-en": ("fr_fr", "en_us", "fra", "eng"),
        "de-en": ("de_de", "en_us", "deu", "eng"),
        "es-en": ("es_419", "en_us", "spa", "eng"),
    }
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in mapping:
            raise SystemExit(f"Unsupported FLEURS language pair: {item}")
        pairs.append(mapping[item])
    return pairs


def prepare_covost2(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    manifest_path = output_root / "manifests" / "covost2_manifest.csv"
    rows: list[PreparedExample] = []
    mdc_root = Path(args.covost_mdc_root) if args.covost_mdc_root else None
    common_voice_root = Path(args.covost_common_voice_root) if args.covost_common_voice_root else None

    for config in [item.strip() for item in args.configs.split(",") if item.strip()]:
        if mdc_root:
            rows.extend(prepare_covost2_mdc_config(args, output_root, mdc_root, config, rng))
            continue
        source_language_code = config.split("_", 1)[0]
        load_kwargs = {"split": args.split, "trust_remote_code": True}
        if common_voice_root:
            language_dir = common_voice_root / source_language_code
            if not language_dir.exists():
                raise SystemExit(f"CoVoST Common Voice directory not found for {config}: {language_dir}")
            load_kwargs["data_dir"] = str(language_dir)
        dataset = disable_audio_decode(load_dataset("facebook/covost2", config, **load_kwargs))
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        if args.max_samples:
            indices = indices[: args.max_samples]

        source_language, target_language = covost_languages(config)
        for index in indices:
            example = dataset[index]
            case = f"covost2-{config}-{stable_id(index, example.get('translation', ''))}"
            audio_path = output_root / "corpora" / "covost2" / config / args.split / f"{case}.wav"
            write_wav(example["audio"], audio_path)
            rows.append(
                PreparedExample(
                    case_id=case,
                    dataset="covost2",
                    split=args.split,
                    workload="quality",
                    audio_path=audio_path,
                    source_language=source_language,
                    target_language=target_language,
                    source_text=example.get("sentence", ""),
                    reference_text=example.get("translation", ""),
                    group_id=f"covost2-{config}",
                    source_clip_id=str(example.get("path") or index),
                    notes="CoVoST 2 speech translation sample.",
                )
            )

    write_manifest(
        manifest_path,
        rows,
        {
            "dataset": "facebook/covost2",
            "split": args.split,
            "configs": args.configs,
            "seed": args.seed,
            "max_samples_per_config": args.max_samples,
            "rows": len(rows),
        },
    )
    return manifest_path


def prepare_covost2_mdc_config(
    args: argparse.Namespace,
    output_root: Path,
    mdc_root: Path,
    config: str,
    rng: random.Random,
) -> list[PreparedExample]:
    source_language, target_language = covost_languages(config)
    source_code = config.split("_", 1)[0]
    dataset_root = find_covost_mdc_dataset_root(mdc_root, config, source_code)
    split_path = find_covost_mdc_split(dataset_root, args.split)
    clips_dir = find_covost_mdc_clips_dir(dataset_root)
    records = [record for record in read_tsv(split_path) if record.get("path") and record.get("translation")]

    indices = list(range(len(records)))
    rng.shuffle(indices)
    if args.max_samples:
        indices = indices[: args.max_samples]

    rows: list[PreparedExample] = []
    for index in indices:
        record = records[index]
        source_path = clips_dir / record["path"]
        if not source_path.exists():
            matches = list(dataset_root.rglob(record["path"]))
            if not matches:
                raise SystemExit(f"CoVoST audio file not found for {config}: {record['path']}")
            source_path = matches[0]
        case = f"covost2-{config}-{stable_id(index, record.get('translation', ''))}"
        audio_path = output_root / "corpora" / "covost2" / config / args.split / f"{case}.wav"
        copy_or_convert_audio(source_path, audio_path)
        rows.append(
            PreparedExample(
                case_id=case,
                dataset="covost2",
                split=args.split,
                workload="quality",
                audio_path=audio_path,
                source_language=source_language,
                target_language=target_language,
                source_text=record.get("sentence", ""),
                reference_text=record.get("translation", ""),
                group_id=f"covost2-{config}",
                source_clip_id=record.get("path", ""),
                notes=f"CoVoST 2 MDC sample from {split_path.name}.",
            )
        )
    return rows


def find_covost_mdc_dataset_root(mdc_root: Path, config: str, source_code: str) -> Path:
    for tsv_path in mdc_root.rglob("*.tsv"):
        normalized_name = tsv_path.name.replace(".", "_")
        if config in normalized_name:
            return tsv_path.parent

    candidates = [
        mdc_root / config,
        mdc_root / f"covost2_{config}",
        mdc_root / f"covost-2-{source_code}-english",
    ]
    candidates.extend(path for path in mdc_root.iterdir() if path.is_dir() and source_code in path.name.lower())
    for candidate in candidates:
        if candidate.exists() and list(candidate.rglob("*.tsv")):
            return candidate
    raise SystemExit(f"Could not find extracted MDC CoVoST directory for {config} under {mdc_root}")


def find_covost_mdc_split(dataset_root: Path, split: str) -> Path:
    split_names = {
        "validation": ["dev.tsv", "validation.tsv", "valid.tsv"],
        "dev": ["dev.tsv", "validation.tsv", "valid.tsv"],
        "test": ["test.tsv"],
        "train": ["train.tsv"],
    }.get(split, [f"{split}.tsv"])
    for name in split_names:
        matches = list(dataset_root.rglob(name))
        if matches:
            return matches[0]
        suffix_matches = list(dataset_root.rglob(f"*.{name}"))
        if suffix_matches:
            return suffix_matches[0]
    raise SystemExit(f"Could not find CoVoST split {split!r} under {dataset_root}")


def find_covost_mdc_clips_dir(dataset_root: Path) -> Path:
    for name in ("clips", "audio"):
        matches = [path for path in dataset_root.rglob(name) if path.is_dir()]
        if matches:
            return matches[0]
    return dataset_root


def covost_languages(config: str) -> tuple[str, str]:
    source, target = config.split("_", 1)
    iso = {
        "ro": "ron",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "en": "eng",
    }
    return iso.get(source, source), iso.get(target, target)


def prepare_common_voice(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    source_root = Path(args.source_dir)
    manifest_path = output_root / "manifests" / "common_voice_ro_manifest.csv"
    validated_tsv = source_root / "validated.tsv"
    clips_dir = source_root / "clips"
    if not validated_tsv.exists():
        raise SystemExit(f"validated.tsv not found: {validated_tsv}")
    if not clips_dir.exists():
        raise SystemExit(f"clips directory not found: {clips_dir}")

    records = read_tsv(validated_tsv)
    records = [record for record in records if record.get("path") and record.get("sentence")]
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        normalized = normalize_transcript(record["sentence"])
        if normalized:
            groups.setdefault(normalized, []).append(record)

    repeated_groups = [
        (normalized, group)
        for normalized, group in groups.items()
        if len(group) >= 2
    ]
    repeated_groups.sort(key=lambda item: (-len(item[1]), item[0]))
    if args.max_groups:
        repeated_groups = repeated_groups[: args.max_groups]

    rows: list[PreparedExample] = []
    for group_index, (normalized, group) in enumerate(repeated_groups):
        seed_record = group[0]
        repeat_record = group[1]
        group_id = f"cv-ro-{stable_id(normalized)}"
        rows.append(common_voice_example(output_root, clips_dir, seed_record, group_id, group_index, "seed", "miss", True))
        rows.append(common_voice_example(output_root, clips_dir, repeat_record, group_id, group_index, "natural_repeat", "text_exact", True))
        rows.append(common_voice_example(output_root, clips_dir, seed_record, group_id, group_index, "exact_replay", "audio_exact", True, suffix="replay"))
        rows.append(common_voice_example(output_root, clips_dir, seed_record, group_id, group_index, "speaker_control", "miss", False, speaker_id="1", suffix="speaker"))
        rows.append(common_voice_example(output_root, clips_dir, seed_record, group_id, group_index, "domain_control", "miss", False, domain="medical", suffix="domain"))

    rows.extend(common_voice_negative_controls(output_root, clips_dir, repeated_groups, limit=args.max_negative_controls))

    write_manifest(
        manifest_path,
        rows,
        {
            "dataset": "mozilla-common-voice-ro-25.0",
            "source_dir": str(source_root),
            "split": "validated",
            "groups": len(repeated_groups),
            "negative_controls": args.max_negative_controls,
            "rows": len(rows),
        },
    )
    return manifest_path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file, delimiter="\t"))


def common_voice_example(
    output_root: Path,
    clips_dir: Path,
    record: dict[str, str],
    group_id: str,
    group_index: int,
    workload: str,
    expected_layer: str,
    reuse_allowed: bool,
    speaker_id: str = "0",
    domain: str = "common_voice",
    suffix: str = "",
) -> PreparedExample:
    source_path = clips_dir / record["path"]
    case_suffix = suffix or stable_id(record["path"], workload, length=8)
    case_id = f"{group_id}-{case_suffix}"
    audio_path = output_root / "corpora" / "common_voice_ro" / f"{case_id}.wav"
    copy_or_convert_audio(source_path, audio_path)
    return PreparedExample(
        case_id=case_id,
        dataset="common_voice_ro",
        split="validated",
        workload=workload,
        audio_path=audio_path,
        source_language="ron",
        target_language="eng",
        source_text=record.get("sentence", ""),
        reference_text="",
        group_id=group_id,
        speaker_id=speaker_id,
        domain=domain,
        privacy_level="public",
        expected_cache_layer=expected_layer,
        expected_reuse_allowed=reuse_allowed,
        source_clip_id=record.get("path", ""),
        notes=f"Common Voice repeated-sentence group {group_index}.",
    )


def common_voice_negative_controls(
    output_root: Path,
    clips_dir: Path,
    repeated_groups: list[tuple[str, list[dict[str, str]]]],
    limit: int,
) -> list[PreparedExample]:
    rows: list[PreparedExample] = []
    for left_index, (left_norm, left_group) in enumerate(repeated_groups):
        if len(rows) >= limit:
            break
        match = None
        for right_norm, right_group in repeated_groups[left_index + 1 : left_index + 80]:
            if left_norm != right_norm and near_text(left_norm, right_norm):
                match = (right_norm, right_group)
                break
        if not match:
            continue
        right_norm, right_group = match
        group_id = f"cv-ro-negative-{stable_id(left_norm, right_norm)}"
        rows.append(common_voice_example(output_root, clips_dir, left_group[0], group_id, left_index, "negative_seed", "miss", True, suffix="seed"))
        rows.append(common_voice_example(output_root, clips_dir, right_group[0], group_id, left_index, "negative_control", "miss", False, suffix="near"))
    return rows


def near_text(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 20:
        return False
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return left[:8] == right[:8]
    score = fuzz.ratio(left, right)
    return 65 <= score < 95


def prepare_all(args: argparse.Namespace) -> list[Path]:
    paths = []
    paths.append(prepare_fleurs(args))
    paths.append(prepare_covost2(args))
    if args.source_dir:
        paths.append(prepare_common_voice(args))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Polyglot evaluation datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("fleurs", "covost2", "common-voice", "all"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        sub.add_argument("--seed", type=int, default=DEFAULT_SEED)
        sub.add_argument("--max-samples", type=int, default=0)
        sub.add_argument("--split", default="validation")
        sub.add_argument("--language-pairs", default="ro-en,fr-en,de-en,es-en")
        sub.add_argument("--configs", default="fr_en,de_en,es_en")
        sub.add_argument("--covost-mdc-root", default="")
        sub.add_argument("--covost-common-voice-root", default="")
        sub.add_argument("--source-dir", default="")
        sub.add_argument("--max-groups", type=int, default=0)
        sub.add_argument("--max-negative-controls", type=int, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "fleurs":
        paths = [prepare_fleurs(args)]
    elif args.command == "covost2":
        paths = [prepare_covost2(args)]
    elif args.command == "common-voice":
        if not args.source_dir:
            raise SystemExit("--source-dir is required for Common Voice")
        paths = [prepare_common_voice(args)]
    else:
        paths = prepare_all(args)

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
