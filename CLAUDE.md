# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR-deploy converts academic PDFs to Markdown using **Chandra OCR 2** (datalab-to/chandra-ocr-2) running on Modal serverless GPUs. A FastAPI HTTP endpoint routes work to a pool of up to 5 L40S containers.

| Backend | Model | GPU | Modal App |
|---------|-------|-----|-----------|
| `chandra/` | datalab-to/chandra-ocr-2 | L40S (48 GB) | `chandra-ocr-2` |

GLM-OCR was removed. The `glm/` directory references and tests are legacy — skip them.

## Common Commands

```bash
# Setup
pip install -r requirements.txt
modal run chandra/setup_volume.py          # one-time: download weights to Volume

# Deploy
modal deploy chandra/modal_app.py

# Single PDF via CLI
modal run chandra/modal_app.py --pdf paper/test.pdf

# Batch via local SDK (uses Modal .remote() directly)
python chandra/process_papers.py --input-dir paper --output-dir output
python chandra/process_papers.py --max-parallel 3
python chandra/process_papers.py --force

# Batch via HTTP API (parallel, up to 5 concurrent)
python chandra/api_batch_extract.py

# Test via API
curl https://<your-app>.modal.run/health
curl -X POST https://<your-app>.modal.run/extract -F "file=@paper/test.pdf"
```

## Architecture

### Two-Tier Deployment

Both components are defined in one file (`chandra/modal_app.py`) and deployed together:

**1. GPU Worker — `ChandraOCR` class** (`@app.cls`)
- `@modal.enter()` — launches local vLLM server on container boot (FlashInfer + BF16 + CUDA graphs, no `--enforce-eager`)
- `process_pdf(pdf_bytes, filename)` — renders pages at 200 DPI via PyMuPDF, OCRs each page concurrently against local vLLM via `asyncio.Semaphore(MAX_CONCURRENT_PAGES)`
- Per-page exponential backoff retry (3 attempts) for transient vLLM failures
- HTML→Markdown post-processing with table rowspan/colspan expansion
- `retries=modal.Retries(max_retries=2)` on the class decorator for container-level failures
- `max_containers=5` — up to 5 L40S containers, each handles one PDF at a time

**2. FastAPI Web Layer — `api()` function** (`@app.function`)
- Lightweight CPU-only container — `web_image` has only fastapi+uvicorn+httpx+python-multipart
- `modal.Cls.from_name("chandra-ocr-2", "ChandraOCR")` — routes to already-deployed GPU pool
- Endpoints: `POST /extract`, `GET /health`
- **Clients must follow redirects** — Modal returns 303 from web tier to GPU workers. Use `httpx.Client(follow_redirects=True)` or `curl -L`.

### Key Configuration (`modal_app.py` lines 46-50)

```python
IMAGE_DPI = 200           # page rendering resolution
MAX_CONTAINERS = 5        # max L40S containers (one per PDF)
MAX_CONCURRENT_PAGES = 2  # pages in-flight per container
```

### Output Directories

| Directory | Populated by |
|-----------|-------------|
| `output/` | `api_batch_extract.py` results |
| `extracted_papers_chandra2/` | `process_papers.py` / local entrypoint results |
| `paper/` | Input PDFs |
