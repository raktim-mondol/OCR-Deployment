# Chandra OCR 2

Academic PDF → Markdown extraction powered by [Chandra OCR 2](https://github.com/datalab-to/chandra-ocr-2) on [Modal](https://modal.com). Converts multi-page PDFs into clean, structured Markdown using a vLLM-served multimodal vision-language model on L40S GPUs.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Web Layer                │
│  CPU-only Modal Function (lightweight, auto-scale)  │
│  POST /extract  —  Accepts PDF, routes to GPU pool  │
│  GET  /health   —  Status check                     │
└─────────────────────────┬───────────────────────────┘
                          │ modal.Cls.from_name
                          │ .remote() call
                          ▼
┌─────────────────────────────────────────────────────┐
│              GPU Worker Pool (up to 5 L40S)         │
│  @app.cls max_containers=5  —  one container ≈ 1 PDF│
│                                                     │
│  Each container:                                    │
│  1. Renders PDF pages to JPEG at 150 DPI (PyMuPDF)  │
│  2. Starts vLLM server on boot (@modal.enter)       │
│  3. OCRs pages concurrently (asyncio + httpx)       │
│     → vLLM /v1/chat/completions local endpoint      │
│  4. Post-processes HTML→Markdown                    │
│     → Table rowspan/colspan expansion               │
│  5. Returns per-page markdown + metadata            │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two-tier deployment** | Web tier on CPU (cheap, fast scale) routes work to GPU pool (expensive, shared). API calls don't spin up new GPU containers. |
| **`modal.Cls.from_name` in API** | Connects to already-deployed `chandra-ocr-2` app so Modal reuses pooled GPU containers instead of deploying fresh ones per call. |
| **vLLM local server** | Each GPU container runs its own vLLM process, eliminating network hops between the OCR logic and inference engine. |
| **Per-page retry with backoff** | Long pages with formulas/tables are prone to vLLM timeout. 3 retries with exponential backoff prevents single-page failures from killing the entire PDF. |
| **Client must follow redirects** | Modal returns 303 from the web tier to GPU workers. Use `httpx.Client(follow_redirects=True)` or `curl -L`. |

### Configuration

All hyperparameters live in [`chandra/config.yml`](chandra/config.yml). Edit values then re-deploy:

```bash
modal deploy chandra/modal_app.py
```

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `path` | `/model/chandra-ocr-2` | Model location inside container |
| `gpu` | `max_containers` | `5` | Parallel PDFs (one container per PDF) |
| `gpu` | `max_concurrent_pages` | `2` | Pages processed simultaneously per container |
| `gpu` | `timeout` | `5000` | Container timeout in seconds |
| `pdf` | `dpi` | `150` | PDF → JPEG resolution per page |
| `pdf` | `jpeg_quality` | `95` | JPEG quality (1–100) |
| `vllm` | `port` | `8000` | vLLM local server port |
| `vllm` | `gpu_memory_utilization` | `0.90` | Fraction of VRAM for vLLM |
| `vllm` | `max_num_batched_tokens` | `65536` | Max token batch size |
| `vllm` | `max_num_seqs` | `4` | Max sequences for batching |
| `vllm` | `max_model_len` | `8192` | Max context length |
| `ocr` | `max_tokens` | `4096` | Max tokens per page |
| `ocr` | `page_timeout` | `500.0` | vLLM request timeout per page (seconds) |
| `ocr` | `max_retries` | `3` | Per-page retry attempts |
| `modal_retries` | `max_retries` | `2` | Container-level retries |
| `modal_retries` | `backoff_coefficient` | `2.0` | Retry backoff multiplier |

## Getting Started

### Prerequisites

- Python 3.10+
- Modal account with API token configured (`modal setup`)
- GPU access on Modal (L40S or similar supported GPU)

### Installation

```bash
pip install -r requirements.txt
```

### First-Time Setup

```bash
# 1. Download model weights into Modal Volume (runs once, ~10-15 min)
modal run chandra/setup_volume.py

# 2. Deploy the app (FastAPI endpoint + GPU worker class)
modal deploy chandra/modal_app.py
```

After deployment, the API is available at:
```
https://<your-account>--<app-name>-api.modal.run
```

## Usage

### Single PDF — via CLI (Modal SDK)

The fastest way to test the system. Dispatches directly to the GPU worker:

```bash
modal run chandra/modal_app.py --pdf paper/mypaper.pdf
modal run chandra/modal_app.py --pdf paper/mypaper.pdf --output-dir my_output
```

### Single PDF — via HTTP API

```bash
# Health check
curl https://<your-app>.modal.run/health

# Extract text from a PDF (follow redirects with -L)
curl -L -X POST https://<your-app>.modal.run/extract \
     -F "file=@paper/mypaper.pdf" | python -m json.tool
```

### Batch — via HTTP API (recommended for production)

Sends up to 5 PDFs concurrently to the deployed API:

```bash
python chandra/api_batch_extract.py
python chandra/api_batch_extract.py --input-dir paper --output-dir output
python chandra/api_batch_extract.py --max-parallel 3      # fewer concurrent
python chandra/api_batch_extract.py --force               # re-process done files
python chandra/api_batch_extract.py --glob "*survey*"     # filename filter
```

Already-converted PDFs are skipped by default (resume-safe).

### Batch — via Modal SDK

Uses `modal.Cls.from_name` to connect to the deployed app and send work:

```bash
python chandra/process_papers.py
python chandra/process_papers.py --input-dir paper --output-dir extracted_papers_chandra2
python chandra/process_papers.py --max-parallel 3
python chandra/process_papers.py --force
python chandra/process_papers.py --glob "*survey*"
```

## API Reference

### `POST /extract`

Upload a PDF file and receive per-page Markdown extraction.

**Request**
```
Content-Type: multipart/form-data
file=<PDF file>
```

**Response**
```json
{
  "filename": "paper.pdf",
  "total_pages": 12,
  "pages": [
    {
      "page": 1,
      "markdown": "# Paper Title\n\nLorem ipsum..."
    },
    {
      "page": 2,
      "markdown": "## Section 2\n\n..."
    }
  ]
}
```

### `GET /health`

**Response**
```json
{
  "status": "ok",
  "model": "chandra-ocr-2",
  "gpu": "L40S"
}
```

## Output Format

Extracted Markdown files contain pages separated by `---` with HTML comments for page markers:

```markdown
<!-- Page 1 -->

# Paper Title

Abstract text here...

---

<!-- Page 2 -->

## Introduction

Content on page 2...
```

Files are written to `output/` (API batch) or `extracted_papers_chandra2/` (SDK batch) by default.

## Deployment Details

### Infrastructure

| Component | Spec |
|-----------|------|
| Model | datalab-to/chandra-ocr-2 |
| GPU | L40S (48 GB VRAM, Ada Lovelace SM 8.9) |
| Attention backend | FlashInfer (`VLLM_ATTENTION_BACKEND=FLASHINFER`) |
| Precision | bfloat16 (BF16 tensor cores) |
| CUDA graphs | Enabled (`--enforce-eager` omitted) |
| PDF rendering | 150 DPI via PyMuPDF |
| Container timeout | 5000s (configurable) |
| Scale-down window | 120s |

### Retry Strategy

Two levels of retry protect against transient failures:

1. **Per-page** (inside `process_pdf`): Exponential backoff, 3 attempts per page with 2^N second delays. Handles vLLM timeouts on long pages.
2. **Per-PDF** (Modal decorator): `modal.Retries(max_retries=2)` retries the entire `process_pdf` call if the container fails. Handles OOM or cold-start issues.

### Storage

- **`chandra-weights` Volume** — Persistent Modal Volume storing model weights. Downloads happen once via `setup_volume.py`.
- **`/model` mount** — Weights mounted read-only in GPU containers.

## Project Structure

```
OCR-deploy/
├── chandra/
│   ├── config.yml              ← Hyperparameters (edit this to tune)
│   ├── modal_app.py            ← Main deployment: GPU class + FastAPI API
│   ├── process_papers.py       ← Batch processor via Modal SDK
│   ├── api_batch_extract.py    ← Batch processor via HTTP API
│   └── setup_volume.py         ← One-time weight downloader
├── modal_doc/
│   ├── SKILL.md                ← Modal SDK reference
│   └── reference.md            ← API quick reference
├── paper/                      ← Input PDFs
├── output/                     ← API batch extraction results
├── extracted_papers_chandra2/  ← SDK batch extraction results
├── requirements.txt
├── CLAUDE.md                   ← Claude Code workspace instructions
└── README.md
```

## Troubleshooting

**303 redirect errors from curl**
```bash
# Use -L to follow redirects
curl -L -X POST https://<your-app>.modal.run/extract -F "file=@paper.pdf"
```

**Container times out on large PDFs**
Increase `gpu.timeout` in `config.yml` and re-deploy.

**Low accuracy on pages with small text or formulas**
Increase `pdf.dpi` in `config.yml` (150 → 200 or 300) and re-deploy. Higher DPI means larger images and slower processing but better OCR quality.

**Out of memory errors**
Reduce `vllm.gpu_memory_utilization` (0.90 → 0.85) or decrease `vllm.max_num_seqs` in `config.yml`.

**vLLM slow to start**
Cold-start is expected — vLLM loads the model at container boot (~2-5 min). The `scaledown_window: 120` keeps containers alive for 2 minutes after the last request to avoid repeated cold starts.

## License

This project is for research and deployment use. The underlying model (Chandra OCR 2) is licensed under its own terms — see the [Chandra OCR 2 repository](https://github.com/datalab-to/chandra-ocr-2) for model licensing details.
