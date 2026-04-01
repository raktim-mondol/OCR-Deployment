"""
Chandra OCR 2 — Modal deployment
=================================
Model   : datalab-to/chandra-ocr-2  (weights in 'chandra-weights' volume)
GPU     : L40S  (48 GB VRAM, Ada Lovelace SM 8.9 — full BF16 + FlashInfer)
Backend : vLLM OpenAI-compatible server
          Flash Attention is enabled via FlashInfer (VLLM_ATTENTION_BACKEND=FLASHINFER)
          --enforce-eager is intentionally omitted → CUDA graph capture stays ON
Scale   : max_containers=5  — one container per PDF, up to 5 PDFs in parallel
Pages   : processed concurrently inside each container (asyncio + httpx semaphore)
DPI     : configurable in config.yml (default 150)

First-time setup
----------------
    # 1. Download model weights into the volume (one-time)
    modal run chandra/setup_volume.py

    # 2. Deploy the app
    modal deploy chandra/modal_app.py

Quick single-PDF test
---------------------
    modal run chandra/modal_app.py --pdf paper/test.pdf

Batch processing (up to 5 PDFs in parallel)
--------------------------------------------
    python chandra/process_papers.py --input-dir paper --output-dir extracted_papers_chandra2

HTTP API
--------
    # Health check
    curl https://<your-app>.modal.run/health

    # Extract text from a PDF
    curl -X POST https://<your-app>.modal.run/extract -F "file=@paper/test.pdf"
"""

import modal
import yaml
import pathlib

# ── Load hyperparameters from config.yml ─────────────────────────────────────
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yml"
with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

MODEL_PATH = CONFIG["model"]["path"]
VLLM_PORT = CONFIG["vllm"]["port"]

# GPU tuning
IMAGE_DPI = CONFIG["pdf"]["dpi"]
JPEG_QUALITY = CONFIG["pdf"]["jpeg_quality"]
MAX_CONTAINERS = CONFIG["gpu"]["max_containers"]
MAX_CONCURRENT_PAGES = CONFIG["gpu"]["max_concurrent_pages"]

# Container-level retry
RETRY_CONFIG = CONFIG["modal_retries"]

# ── Persistent volume (weights downloaded once via setup_volume.py) ───────────
volume = modal.Volume.from_name("chandra-weights")

# ── Container image ────────────────────────────────────────────────────────────
# nvidia devel image ensures CUDA headers/toolchain are present for vLLM's
# JIT-compiled kernels.  vLLM ≥ 0.6.0 ships FlashInfer pre-compiled for
# CUDA 12.x; on L40S (SM 8.9) FlashInfer is the default attention backend —
# faster than plain Flash-Attention 2 for batched decoding.
#
# Flash Attention activation summary:
#   - VLLM_ATTENTION_BACKEND=FLASHINFER  (env, set below)
#   - --dtype bfloat16                   (optimal for L40S; enables BF16 tensor cores)
#   - --enforce-eager OMITTED            (keeps CUDA graph capture ON)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "gcc",
        "g++",
        "build-essential",
        "git",
        "libgomp1",
    )
    # vLLM 0.6+ bundles FlashInfer (and optionally Flash-Attention 2) for CUDA 12.x.
    # Install first so its torch pin resolves before the rest of the packages.
    .pip_install("vllm>=0.6.0")
    .pip_install(
        "chandra-ocr",  # Chandra OCR 2 Python SDK
        "pyyaml>=6.0",  # config.yml loader
        "PyMuPDF>=1.24.0",  # fitz — PDF→JPEG, pure Python+C, no Poppler
        "Pillow>=10.0.0",
        "requests",
        "httpx>=0.27.0",  # async HTTP client for vLLM API calls
        "fastapi>=0.115.0",
        "python-multipart",
        "uvicorn",
        "beautifulsoup4>=4.12.0",  # HTML → Markdown post-processing
        "lxml",  # faster BS4 parser
    )
    .env(
        {
            # Explicitly select FlashInfer attention backend in vLLM.
            # On L40S this gives fused prefill/decode kernels with BF16 support.
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

app = modal.App("chandra-ocr-2")


# ── HTML → Markdown post-processor ────────────────────────────────────────────
def html_to_markdown(html: str) -> str:
    """
    Convert HTML output from the model into pure GitHub-Flavoured Markdown.

    Handles:
      • Headings (h1–h6), bold, italic, inline code, superscript, links
      • Ordered and unordered lists
      • Tables — including cells with rowspan / colspan.
        Standard GFM tables cannot express merged cells, so we expand them:
          - colspan N  → repeat the cell text across N columns
          - rowspan N  → repeat the cell text in the same column for N rows
        This faithfully represents the data without silently dropping content.

    If the input contains no HTML tags it is returned unchanged so that
    already-correct Markdown pages pass through with zero cost.
    """
    import re
    from bs4 import BeautifulSoup

    # Fast path: no tags present → already plain Markdown
    if not re.search(r"<[a-zA-Z][^>]*>", html):
        return html

    soup = BeautifulSoup(html, "lxml")

    # ── Tables ────────────────────────────────────────────────────────────────
    def _expand_table(table) -> list[list[str]]:
        """
        Build a 2-D list of strings from an HTML table, fully expanding
        rowspan and colspan.  Uses an 'occupied' dict keyed by (row, col)
        so that cells from a prior rowspan are inserted at the right column.
        """
        rows = table.find_all("tr")
        occupied: dict[tuple[int, int], str] = {}
        max_col = 0

        for r_idx, row in enumerate(rows):
            c_idx = 0
            for cell in row.find_all(["th", "td"]):
                # skip columns already filled by a rowspan above
                while (r_idx, c_idx) in occupied:
                    c_idx += 1

                # inline formatting inside the cell
                for b in cell.find_all(["b", "strong"]):
                    b.replace_with(f"**{b.get_text()}**")
                for i in cell.find_all(["i", "em"]):
                    i.replace_with(f"*{i.get_text()}*")
                for code in cell.find_all("code"):
                    code.replace_with(f"`{code.get_text()}`")

                text = (
                    cell.get_text(" ", strip=True)
                    .replace("|", "\\|")  # escape pipe so it doesn't break columns
                    .replace("\n", " ")
                )
                rs = int(cell.get("rowspan", 1))
                cs = int(cell.get("colspan", 1))

                # stamp every spanned (row, col) position with this text
                for dr in range(rs):
                    for dc in range(cs):
                        occupied[(r_idx + dr, c_idx + dc)] = text

                c_idx += cs
                max_col = max(max_col, c_idx)

        n_rows = len(rows)
        return [
            [occupied.get((r, c), "") for c in range(max_col)] for r in range(n_rows)
        ]

    def _table_to_md(table) -> str:
        grid = _expand_table(table)
        if not grid:
            return ""
        n_cols = max(len(row) for row in grid)
        lines: list[str] = []
        for i, row in enumerate(grid):
            padded = row + [""] * (n_cols - len(row))
            lines.append("| " + " | ".join(c if c else " " for c in padded) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * n_cols) + " |")
        return "\n".join(lines)

    for table in soup.find_all("table"):
        table.replace_with(_table_to_md(table) + "\n")

    # ── Block-level elements ──────────────────────────────────────────────────
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            tag.replace_with(f"\n{'#' * level} {tag.get_text(strip=True)}\n")

    for tag in soup.find_all("p"):
        tag.replace_with(f"\n{tag.get_text()}\n")

    for tag in soup.find_all("br"):
        tag.replace_with("\n")

    for ol in soup.find_all("ol"):
        items = [
            f"{i}. {li.get_text(strip=True)}"
            for i, li in enumerate(ol.find_all("li"), 1)
        ]
        ol.replace_with("\n" + "\n".join(items) + "\n")

    for ul in soup.find_all("ul"):
        items = [f"- {li.get_text(strip=True)}" for li in ul.find_all("li")]
        ul.replace_with("\n" + "\n".join(items) + "\n")

    # ── Inline elements ───────────────────────────────────────────────────────
    for tag in soup.find_all(["b", "strong"]):
        tag.replace_with(f"**{tag.get_text()}**")

    for tag in soup.find_all(["i", "em"]):
        tag.replace_with(f"*{tag.get_text()}*")

    for tag in soup.find_all("code"):
        tag.replace_with(f"`{tag.get_text()}`")

    for tag in soup.find_all("sup"):
        tag.replace_with(f"^{tag.get_text()}^")

    for tag in soup.find_all("a"):
        href = tag.get("href", "")
        text = tag.get_text()
        tag.replace_with(f"[{text}]({href})" if href else text)

    # ── Collect, clean up excess blank lines ─────────────────────────────────
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# max_containers=5 → Modal will run at most 5 instances of this class at once.
# Each instance handles exactly one PDF (no @modal.concurrent used here),
# so the system processes up to 5 PDFs simultaneously.
@app.cls(
    image=image,
    gpu="L40S:1",
    timeout=1800,
    scaledown_window=120,
    volumes={"/model": volume},
    max_containers=MAX_CONTAINERS,
    retries=modal.Retries(
        max_retries=RETRY_CONFIG["max_retries"],
        backoff_coefficient=RETRY_CONFIG["backoff_coefficient"],
    ),
)
class ChandraOCR:
    @modal.enter()
    def start_vllm_server(self):
        """
        Launch the vLLM OpenAI-compatible server once when the container boots.

        Flash Attention optimisations active in this configuration:
          1. VLLM_ATTENTION_BACKEND=FLASHINFER  — env var (set in image .env)
          2. --dtype bfloat16                   — BF16 tensor cores on L40S
          3. --enforce-eager OMITTED            — CUDA graphs ON (kernel fusion)
        """
        import subprocess, time
        import requests as req

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL_PATH,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            # ── Flash Attention / performance ──────────────────────────────
            "--dtype",
            "bfloat16",  # BF16 on L40S (Ada)
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-batched-tokens",
            "65536",
            # ── Model / safety ─────────────────────────────────────────────
            "--trust-remote-code",
            "--max-model-len",
            "8192",
            # ── Multimodal: one image per page ─────────────────────────────
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--mm-processor-kwargs",
            '{"min_pixels": 3136, "max_pixels": 602112}',
            # NOTE: --enforce-eager is intentionally NOT passed here.
            # Omitting it keeps CUDA graph capture active, which combined
            # with VLLM_ATTENTION_BACKEND=FLASHINFER gives maximum throughput.
        ]

        self.vllm_proc = subprocess.Popen(cmd)
        print("Waiting for vLLM server (FlashInfer, CUDA graphs, BF16) ...")

        for attempt in range(120):  # poll every 5 s, up to 10 min
            try:
                req.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
                print(f"vLLM server ready (waited {attempt * 5}s)")
                return
            except Exception:
                time.sleep(5)

        raise RuntimeError("vLLM server did not become healthy within 10 minutes")

    @modal.exit()
    def stop_vllm_server(self):
        self.vllm_proc.terminate()
        self.vllm_proc.wait()

    @modal.method()
    def process_pdf(self, pdf_bytes: bytes, filename: str = "doc.pdf") -> dict:
        """
        Render every page of the PDF at the configured DPI (see config.yml),
        then OCR all pages concurrently against the local vLLM server using asyncio.

        Parameters
        ----------
        pdf_bytes : raw bytes of the input PDF
        filename  : original filename (used for logging and output labelling)

        Returns
        -------
        {
            "filename":    str,
            "total_pages": int,
            "pages": [{"page": int, "markdown": str}, ...]   # sorted by page
        }
        """
        import fitz  # PyMuPDF
        import io, base64, asyncio, httpx
        from PIL import Image

        # ── 1. Render all pages to JPEG ───────────────────────────────
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_images: list[tuple[int, bytes]] = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=IMAGE_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            page_images.append((i + 1, buf.getvalue()))

        doc.close()
        total = len(page_images)
        print(
            f"[{filename}] {total} page(s) rendered at {IMAGE_DPI} DPI — OCR starting"
        )

        # ── 2. OCR all pages concurrently via async httpx → vLLM ─────────
        async def ocr_page(
            sem: asyncio.Semaphore,
            client: httpx.AsyncClient,
            page_num: int,
            jpeg_bytes: bytes,
        ) -> dict:
            """OCR a single page with exponential backoff retry."""
            import asyncio as aio
            attempt = 0
            max_retries = 3
            while True:
                async with sem:
                    b64 = base64.b64encode(jpeg_bytes).decode()
                    payload = {
                        "model": MODEL_PATH,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64}"
                                        },
                                    },
                                    {"type": "text", "text": "ocr"},
                                ],
                            }
                        ],
                        "max_tokens": 4096,
                        "temperature": 0.0,
                    }
                    try:
                        resp = await client.post(
                            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                            json=payload,
                            timeout=300.0,
                        )
                        resp.raise_for_status()
                        content = resp.json()["choices"][0]["message"]["content"]
                        markdown = html_to_markdown(content)
                        return {"page": page_num, "markdown": markdown}
                    except Exception as exc:
                        if attempt >= max_retries:
                            return {"page": page_num, "markdown": f"[OCR failed on page {page_num} after {max_retries} retries: {exc}]"}
                        attempt += 1
                        print(f"[{filename}] Page {page_num} attempt {attempt}/{max_retries} failed: {exc}")
                        await aio.sleep(2 ** attempt)

        async def run_all() -> list:
            # Semaphore caps the number of pages in-flight at once.
            # vLLM batches them internally for maximum GPU utilisation.
            sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
            async with httpx.AsyncClient(timeout=300.0) as client:
                tasks = [
                    ocr_page(sem, client, pnum, jbytes) for pnum, jbytes in page_images
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())
        results.sort(key=lambda r: r["page"])

        print(f"[{filename}] {len(results)}/{total} pages complete")
        return {
            "filename": filename,
            "total_pages": total,
            "pages": results,
        }


# ── FastAPI HTTP endpoint ──────────────────────────────────────────────────────
# CPU-only image for the API web layer — no CUDA / build-essential needed
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn",
        "python-multipart",
        "httpx>=0.27.0",
    )
)


@app.function(image=web_image, timeout=1800)
@modal.asgi_app()
def api():
    """
    FastAPI web layer — lightweight CPU container.
    Routes PDFs to the already-deployed chandra-ocr-2 app via Cls.from_name,
    so Modal reuses pooled GPU containers instead of deploying new ones.
    """
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse

    web = FastAPI(title="Chandra OCR 2 API")
    ChandraOCRRemote = modal.Cls.from_name("chandra-ocr-2", "ChandraOCR")
    worker = ChandraOCRRemote()

    @web.post("/extract")
    async def extract(file: UploadFile = File(...)):
        """Upload a PDF, receive markdown per page."""
        pdf_bytes = await file.read()
        result = worker.process_pdf.remote(pdf_bytes, file.filename or "upload.pdf")
        return JSONResponse(result)

    @web.get("/health")
    def health():
        return {"status": "ok", "model": "chandra-ocr-2", "gpu": "L40S"}

    return web


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    pdf: str = "",
    input_dir: str = "paper",
    output_dir: str = "extracted_papers_chandra2",
):
    """
    Single-PDF mode:
        modal run chandra/modal_app.py --pdf paper/test.pdf

    Batch mode (dispatches up to 5 containers in parallel):
        modal run chandra/modal_app.py --input-dir paper

    Output .md files are written to --output-dir (default: extracted_papers_chandra2/).
    """
    import pathlib

    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    p = pathlib.Path(pdf) if pdf else None
    if p is not None and p.is_dir():
        # user passed a folder to --pdf (e.g. --pdf paper) → treat as input-dir
        pdfs = sorted(p.glob("*.pdf"))
    elif p is not None and p.is_file():
        pdfs = [p]
    elif p is not None:
        print(f"Error: '{pdf}' is not a file or directory.")
        return
    else:
        pdfs = sorted(pathlib.Path(input_dir).glob("*.pdf"))

    if not pdfs:
        src = pdf or input_dir
        print(f"No PDFs found in '{src}'.")
        return

    print(
        f"Found {len(pdfs)} PDF(s).  "
        f"Dispatching to up to {MAX_CONTAINERS} L40S containers …"
    )

    worker = ChandraOCR()
    payloads = [(p.read_bytes(), p.name) for p in pdfs]

    # starmap sends each (pdf_bytes, filename) pair to a separate container.
    # max_containers=5 on the class caps simultaneous containers at 5,
    # so at most 5 PDFs are processed in parallel at any time.
    for result in worker.process_pdf.starmap(payloads, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"ERROR: {result}")
            continue

        fname = result["filename"]
        md_path = out_path / (pathlib.Path(fname).stem + ".md")
        sections = [
            f"<!-- Page {p['page']} -->\n\n{p['markdown']}" for p in result["pages"]
        ]
        md_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
        print(f"Saved {md_path}  ({result['total_pages']} pages)")
