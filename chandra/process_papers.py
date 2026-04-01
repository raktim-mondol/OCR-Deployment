"""
Batch PDF → Markdown using Chandra OCR 2 on Modal.

For every PDF in --input-dir, produces a matching .md file in --output-dir.
Already-converted files are skipped unless --force is passed.

Architecture
------------
  • The deployed ChandraOCR class (modal deploy chandra/modal_app.py) is
    limited to max_containers=5, so at most 5 L40S containers run at once.
  • This script uses a ThreadPoolExecutor(max_workers=5) to keep the pipeline
    full: each thread calls process_pdf.remote(pdf_bytes, name) which routes
    to a dedicated L40S container.
  • Inside each container all pages of that PDF are processed concurrently
    via asyncio + the local vLLM server (FlashInfer / CUDA-graphs active).

Requires the app to be deployed first:
    modal deploy chandra/modal_app.py

Usage
-----
    python chandra/process_papers.py
    python chandra/process_papers.py --input-dir paper --output-dir out/chandra2
    python chandra/process_papers.py --max-parallel 3   # fewer containers
    python chandra/process_papers.py --force            # re-process done files
    python chandra/process_papers.py --glob "*survey*"  # filename filter
"""

import argparse
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import yaml

# ── Load config ──────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yml"
with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

MAX_CONTAINERS = CONFIG["gpu"]["max_containers"]
DPI = CONFIG["pdf"]["dpi"]

# Thread lock so log lines from parallel PDF jobs don't interleave
_print_lock = threading.Lock()


def tprint(tag: str, msg: str) -> None:
    with _print_lock:
        print(f"  [{tag}] {msg}", flush=True)


def process_one_pdf(
    pdf_path: Path,
    out_path: Path,
    model,  # ChandraOCR stub from modal.Cls.from_name
) -> dict:
    """
    Send one PDF to a dedicated L40S container and write the result.
    Returns a stats dict: {pages, chars, elapsed, error}.
    """
    tag = pdf_path.stem[:30]
    t0 = time.time()

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        return {"pages": 0, "chars": 0, "elapsed": 0.0, "error": str(exc)}

    tprint(tag, f"Submitting {pdf_path.name} ({len(pdf_bytes) / 1024:.0f} KB) ...")

    try:
        result = model.process_pdf.remote(pdf_bytes, pdf_path.name)
    except Exception as exc:
        return {"pages": 0, "chars": 0, "elapsed": time.time() - t0, "error": str(exc)}

    # Build markdown output: one section per page
    sections = [
        f"<!-- Page {p['page']} -->\n\n{p['markdown']}" for p in result["pages"]
    ]
    markdown = "\n\n---\n\n".join(sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    elapsed = time.time() - t0
    stats = {
        "pages": result["total_pages"],
        "chars": len(markdown),
        "elapsed": elapsed,
        "error": None,
    }
    tprint(
        tag,
        f"OK  {stats['pages']} pages  {stats['chars']:,} chars  {elapsed:.1f}s",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PDF → Markdown via Chandra OCR 2 on Modal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="paper",
        help="Folder containing input PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_papers_chandra2",
        help="Folder for output .md files",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Max PDFs processed simultaneously (≤ max_containers=5 on the app)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process PDFs that already have an output file",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Filename glob filter, e.g. '*survey*'",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover PDFs ──────────────────────────────────────────────────────────
    all_pdfs = sorted(input_dir.glob(args.glob))
    if not all_pdfs:
        print(f"No PDFs matching '{args.glob}' found in {input_dir}")
        sys.exit(0)

    todo, done = [], []
    for pdf in all_pdfs:
        out = output_dir / (pdf.stem + ".md")
        if out.exists() and not args.force:
            done.append(pdf)
        else:
            todo.append(pdf)

    print(f"\n{'=' * 64}")
    print(f"  Chandra OCR 2  —  Batch Processor")
    print(f"{'=' * 64}")
    print(f"  Input       : {input_dir}  ({len(all_pdfs)} PDF(s) found)")
    print(f"  Output      : {output_dir}")
    print(f"  DPI         : {DPI}  (set in config.yml)")
    print(f"  Parallel    : {args.max_parallel}  PDF(s) at once  (max_containers={MAX_CONTAINERS})")
    print(f"  Per-container: all pages processed concurrently via asyncio+vLLM")
    print(f"  GPU         : L40S  (FlashInfer / CUDA graphs / BF16)")
    print(
        f"  To do       : {len(todo)}   Already done : {len(done)}  (--force to redo)"
    )
    print(f"{'=' * 64}\n")

    if not todo:
        print("All PDFs already converted. Use --force to re-process.")
        sys.exit(0)

    # ── Connect to the deployed Modal class ────────────────────────────────────
    # Requires: modal deploy chandra/modal_app.py
    ChandraOCR = modal.Cls.from_name("chandra-ocr-2", "ChandraOCR")
    model = ChandraOCR()

    # ── Process PDFs in parallel ───────────────────────────────────────────────
    session_start = time.time()
    results_map = {}

    def run_one(pdf: Path):
        out_path = output_dir / (pdf.stem + ".md")
        try:
            return pdf.name, process_one_pdf(pdf, out_path, model)
        except Exception:
            return pdf.name, {
                "pages": 0,
                "chars": 0,
                "elapsed": 0.0,
                "error": traceback.format_exc(),
            }

    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run_one, pdf): pdf for pdf in todo}
        for future in as_completed(futures):
            pdf_name, stats = future.result()
            results_map[pdf_name] = stats
            completed += 1
            with _print_lock:
                print(f"  [{completed}/{len(todo)}] {pdf_name} complete", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    results = [{"pdf": k, **v} for k, v in results_map.items()]
    total_time = time.time() - session_start
    ok_count = sum(1 for r in results if not r["error"])
    err_count = len(results) - ok_count
    total_pages = sum(r["pages"] for r in results)
    total_chars = sum(r["chars"] for r in results)

    print(f"\n{'=' * 64}")
    print(
        f"  Done    : {ok_count} succeeded  |  {err_count} failed  |  {len(done)} skipped"
    )
    print(f"  Pages   : {total_pages}    Chars : {total_chars:,}")
    print(
        f"  Time    : {total_time:.1f}s  ({total_time / max(ok_count, 1):.1f}s avg/PDF)"
    )
    print(f"  Output  : {output_dir}")
    print(f"{'=' * 64}")

    if err_count:
        print("\nFailed PDFs:")
        for r in results:
            if r["error"]:
                print(f"  - {r['pdf']}: {r['error'][:160]}")


if __name__ == "__main__":
    main()
