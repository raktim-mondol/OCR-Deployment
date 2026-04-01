import asyncio
import base64
import io
import pathlib
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import yaml

# ── Load config ──────────────────────────────────────────────────────────
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yml"
with open(CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

MODEL_PATH = CONFIG["model"]["path"]
API_URL = "https://dr-raktim-mondol--chandra-ocr-2-api.modal.run"
TIMEOUT_SECONDS = 600  # 10 min per PDF

_lock = threading.Lock()


def extract_pdf(pdf_path: pathlib.Path, out_path: pathlib.Path, client: httpx.Client) -> dict | None:
    """Send one PDF to the deployed OCR API, write result to disk, return stats."""
    tag = pdf_path.stem[:30]
    t0 = time.time()
    size_kb = pdf_path.stat().st_size / 1024

    with _lock:
        print(f"  [{tag}] Submitting {pdf_path.name} ({size_kb:.0f} KB) ...")

    resp = client.post(
        f"{API_URL}/extract",
        files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
    )
    resp.raise_for_status()
    result = resp.json()

    pages = result.get("pages", [])
    sections = [f"<!-- Page {p['page']} -->\n\n{p['markdown']}" for p in pages]
    markdown = "\n\n---\n\n".join(sections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    elapsed = time.time() - t0
    stats = {"pages": len(pages), "chars": len(markdown), "elapsed": elapsed, "error": None}

    with _lock:
        print(f"  [{tag}] OK  {stats['pages']} pages  {stats['chars']:,} chars  {elapsed:.1f}s")

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch PDF->Markdown via deployed Chandra OCR 2 API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="paper", help="Folder containing input PDFs")
    parser.add_argument("--output-dir", default="output", help="Folder for output .md files")
    parser.add_argument("--glob", default="*.pdf", help="Filename glob filter")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=CONFIG["gpu"]["max_containers"],
        help="Max concurrent PDF requests (default: max_containers from config)",
    )
    parser.add_argument("--force", action="store_true", help="Re-process already-converted PDFs")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return

    all_pdfs = sorted(input_dir.glob(args.glob))
    if not all_pdfs:
        print(f"No PDFs matching '{args.glob}' found in {input_dir}")
        return

    todo = []
    done_count = 0
    for pdf in all_pdfs:
        out = output_dir / (pdf.stem + ".md")
        if out.exists() and not args.force:
            done_count += 1
        else:
            todo.append(pdf)

    print(f"\n{'=' * 64}")
    print(f"  Chandra OCR 2  —  API Batch Extractor")
    print(f"{'=' * 64}")
    print(f"  API         : {API_URL}")
    print(f"  Input       : {input_dir}  ({len(all_pdfs)} PDF(s) found)")
    print(f"  Output      : {output_dir}")
    print(f"  DPI         : {CONFIG['pdf']['dpi']}")
    print(f"  Parallel    : {args.max_parallel}  PDF(s) at once (max_containers={CONFIG['gpu']['max_containers']})")
    print(f"  To do       : {len(todo)}   Already done : {done_count}  (--force to redo)")
    print(f"{'=' * 64}\n")

    if not todo:
        print("All PDFs already converted. Use --force to re-process.")
        return

    session_start = time.time()
    ok_count = 0
    err_count = 0
    total_pages = 0

    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {pool.submit(extract_pdf, pdf, output_dir / (pdf.stem + ".md"), client): pdf for pdf in todo}
            completed = 0
            for future in as_completed(futures):
                pdf = futures[future]
                tag = pdf.stem[:30]
                completed += 1
                try:
                    stats = future.result()
                    if stats:
                        ok_count += 1
                        total_pages += stats["pages"]
                except Exception as e:
                    err_count += 1
                    with _lock:
                        print(f"  [{tag}] FAILED: {e}")

    elapsed = time.time() - session_start
    print(f"\n{'=' * 64}")
    print(f"  Done    : {ok_count} succeeded  |  {err_count} failed  |  {done_count} skipped")
    print(f"  Pages   : {total_pages}")
    print(f"  Time    : {elapsed:.1f}s  ({elapsed / max(ok_count, 1):.1f}s avg/PDF)")
    print(f"  Output  : {output_dir}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
