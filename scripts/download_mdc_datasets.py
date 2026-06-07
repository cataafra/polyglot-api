"""Download Mozilla Data Collective datasets for evaluation.

The MDC API key is read from MDC_API_KEY. Raw archives are written outside git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_DATASETS = {
    "covost2_fr_en": "cmpbfyxlc002nmj07k67e3ok2",
    "covost2_de_en": "cmou2fdyx015fl307bux4c4gi",
    "covost2_es_en": "cmp704ive02qymp075orb8ok4",
}


def api_request(path: str, method: str, api_key: str) -> dict:
    request = Request(
        f"https://mozilladatacollective.com/api{path}",
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"MDC API request failed for {path}: HTTP {error.code}: {body}") from error


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"exists {output_path}")
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "polyglot-api-evaluation/1.0"})
    with urlopen(request, timeout=120) as response, tmp_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    tmp_path.replace(output_path)
    print(f"downloaded {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MDC evaluation datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", choices=sorted(DEFAULT_DATASETS), help="Dataset alias to download.")
    args = parser.parse_args()

    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        raise SystemExit("MDC_API_KEY is required")

    output_dir = Path(args.output_dir)
    aliases = args.dataset or list(DEFAULT_DATASETS)
    for alias in aliases:
        dataset_id = DEFAULT_DATASETS[alias]
        details = api_request(f"/datasets/{dataset_id}", "GET", api_key)
        session = api_request(f"/datasets/{dataset_id}/download", "POST", api_key)
        filename = session.get("filename") or f"{alias}.tar.gz"
        metadata_path = output_dir / f"{alias}.metadata.json"
        metadata_path.write_text(
            json.dumps({"details": details, "download": {k: v for k, v in session.items() if k != "downloadUrl"}}, indent=2),
            encoding="utf-8",
        )
        download_file(session["downloadUrl"], output_dir / filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
