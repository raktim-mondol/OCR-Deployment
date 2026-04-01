---
name: modal
description: >
  Use this skill whenever the user wants to build, deploy, or debug applications on Modal (modal.com) —
  the serverless cloud/GPU platform for Python. Covers: writing Modal Functions and Classes, configuring
  container Images, mounting Volumes and Secrets, building web endpoints (FastAPI/ASGI/WSGI), sandboxed
  execution, scheduling, parallel map/starmap, CLI usage, and common patterns for AI/ML workloads.
  Trigger when the user mentions "Modal", "modal.com", "@app.function", "modal deploy", "modal run",
  GPU serverless, or any Modal SDK imports.
license: MIT
---

# Modal SDK Skill

Modal is a serverless cloud platform for running Python functions on scalable infrastructure, including
GPUs. This skill covers the full Modal Python SDK API and CLI so you can build, deploy, and debug Modal
applications correctly.

**Official docs:** https://modal.com/docs/guide  
**Full API reference:** https://modal.com/docs/reference  
**Examples:** https://modal.com/docs/examples

---

## Core Mental Model

Every Modal program has three layers:

1. **App** — the deployment unit; groups Functions and Classes together
2. **Function / Cls** — Python callables decorated to run in Modal-managed containers
3. **Infrastructure** — Images (container envs), Volumes, Secrets, Queues, Dicts attached to Functions

```python
import modal

app = modal.App("my-app")

@app.function(gpu="A10G", timeout=300)
def my_fn(x):
    return x * 2
```

---

## Application Construction

### `modal.App`

```python
app = modal.App(
    name="my-app",           # optional display name
    image=base_image,        # default Image for all functions
    secrets=[...],           # default Secrets
)
```

| Method | Description |
|---|---|
| `App(name, image, secrets, volumes)` | Construct a new App |
| `app.name` | User-provided name |
| `app.app_id` | ID of running/stopped app |
| `App.from_name(name)` | Look up or create an App by name (useful for Sandboxes) |
| `app.run()` | Context manager — run ephemeral app locally |
| `app.deploy()` | Programmatic equivalent of `modal deploy` |
| `@app.function(...)` | Register a Function with the App |
| `@app.cls(...)` | Register a Class with the App |
| `app.include(other_app)` | Merge another App's objects in |
| `app.set_tags(tags)` | Attach key-value metadata (billing, org context) |
| `app.get_tags()` | Retrieve current tags |

**Running locally:**
```python
# Protect from running on import inside containers
if __name__ == "__main__":
    with app.run():
        result = my_fn.remote(42)
```

---

## Serverless Execution

### `modal.Function`

Constructed via `@app.function()`, never directly.

```python
@app.function(
    image=modal.Image.debian_slim().pip_install("torch"),
    gpu="A10G",              # or "T4", "A100", "H100", etc.
    cpu=2,
    memory=8192,             # MB
    timeout=600,             # seconds
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("my-secret")],
    volumes={"/data": modal.Volume.from_name("my-vol")},
    schedule=modal.Period(hours=1),   # or modal.Cron("0 * * * *")
    concurrency_limit=10,
    allow_concurrent_inputs=5,
)
def my_function(x: int) -> int:
    return x ** 2
```

| Method | Description |
|---|---|
| `.remote(*args)` | Call remotely, block until result |
| `.remote_gen(*args)` | Call as a remote generator |
| `.local(*args)` | Run in caller's environment (no container) |
| `.spawn(*args)` | Fire-and-forget; returns `FunctionCall` |
| `.map(inputs, order_outputs=True, return_exceptions=False)` | Parallel map over iterator |
| `.starmap(inputs)` | Like map but unpacks each item as `*args` |
| `.for_each(inputs)` | Map without collecting results |
| `.spawn_map(inputs)` | Spawn map without waiting |
| `.get_raw_f()` | Return the underlying Python function |
| `.get_current_stats()` | Return `FunctionStats` (queue depth, runner count) |
| `.update_autoscaler(...)` | Override autoscaler at runtime (resets on redeploy) |
| `Function.from_name(app_name, fn_name)` | Reference a deployed Function by name |
| `.get_web_url()` | Get URL for web-endpoint functions |

**Parallel map example:**
```python
results = list(my_function.map([1, 2, 3, 4], order_outputs=True))
# With exception handling:
results = list(my_function.map(inputs, return_exceptions=True))
```

### `modal.FunctionCall`

```python
call = my_function.spawn(42)
result = call.get(timeout=30)   # block up to 30s
```

---

## Class-based Functions (modal.Cls)

Use `@app.cls()` when you need shared state across calls (e.g., loaded model).

```python
@app.cls(gpu="A10G", image=image)
class Predictor:
    @modal.enter()
    def load(self):
        import torch
        self.model = torch.load("model.pt")   # runs once at container start

    @modal.method()
    def predict(self, x):
        return self.model(x)

    @modal.exit()
    def cleanup(self):
        del self.model                         # runs at container shutdown
```

### Parametrized Classes

```python
@app.cls()
class MyService:
    threshold: float = modal.parameter(default=0.5)

    @modal.method()
    def run(self, x):
        return x > self.threshold

# Instantiate with different parameters
svc = MyService(threshold=0.8)
svc.run.remote(0.9)
```

---

## Container Images

### `modal.Image` Factory Methods

```python
# Debian slim (default Python image)
image = modal.Image.debian_slim(python_version="3.11")

# From a Docker Hub or registry image
image = modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04")

# Micromamba (conda-compatible)
image = modal.Image.micromamba(python_version="3.10")
```

### Image Build Methods (chainable)

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy")
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("ffmpeg", "git")
    .env({"HF_HOME": "/cache/huggingface"})
    .workdir("/app")
    .add_local_file("config.json", "/app/config.json", copy=True)
    .run_function(preload_weights, secrets=[secret])  # arbitrary build step
)
```

| Method | Description |
|---|---|
| `.pip_install(*pkgs)` | Install Python packages |
| `.pip_install_from_requirements(path)` | From `requirements.txt` |
| `.pip_install_from_pyproject(path)` | From `pyproject.toml` |
| `.pip_install_private_repos(*repos, secret)` | Private GitHub/GitLab repos |
| `.apt_install(*pkgs)` | Install Debian/apt packages |
| `.run_function(fn, ...)` | Run arbitrary Python as a build step; filesystem changes captured |
| `.run_commands(*cmds)` | Run shell commands as build steps |
| `.add_local_file(local, remote, copy=False)` | Add a file; `copy=True` bakes into layer |
| `.add_local_python_source(module)` | Include a local Python package (required in v1.0+) |
| `.env(dict)` | Set environment variables |
| `.workdir(path)` | Set working directory |
| `.uv_pip_install(*pkgs)` | Install with `uv` (faster) |

> **Note (v1.0+):** Local Python packages are NOT automatically mounted. Use `.add_local_python_source("mypackage")` explicitly.

---

## Secrets

```python
# From a dict
secret = modal.Secret.from_dict({"API_KEY": "abc123"})

# From local environment variables
secret = modal.Secret.from_local_environ(["OPENAI_API_KEY", "HF_TOKEN"])

# From .env file
secret = modal.Secret.from_dotenv()           # finds .env from CWD
secret = modal.Secret.from_dotenv(__file__)   # finds .env relative to this file

# From a named Modal Secret (created in the dashboard)
secret = modal.Secret.from_name("my-openai-secret")
```

Pass to functions via `secrets=[...]` in `@app.function()` — they appear as environment variables inside the container.

---

## Storage

### `modal.Volume` — Persistent high-throughput storage

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/data": vol})
def write_data():
    with open("/data/output.txt", "w") as f:
        f.write("hello")
    vol.commit()     # flush writes to Modal servers

def read_data():
    vol.reload()     # pull latest from Modal servers
    with open("/data/output.txt") as f:
        return f.read()
```

| Method | Description |
|---|---|
| `Volume.from_name(name, create_if_missing=True)` | Get or create Volume |
| `Volume.from_id(id)` | Reference by object ID |
| `vol.commit()` | Flush pending writes (call inside container) |
| `vol.reload()` | Pull latest state (call inside container) |
| `vol.read_only()` | Configure Volume to disallow writes |
| `vol.info()` | Metadata about the Volume |
| `vol.name` | Volume name |

### `modal.Dict` — Distributed key-value store

```python
d = modal.Dict.from_name("my-dict", create_if_missing=True)
d["key"] = "value"
val = d.get("key", default=None)
d.put("key2", 99, skip_if_exists=True)
d.contains("key")   # → bool
d.len()             # expensive; max 100,000
d.update({"a": 1, "b": 2})
d.clear()
d.info()
```

### `modal.Queue` — Distributed FIFO queue

```python
q = modal.Queue.from_name("my-queue", create_if_missing=True)
q.put("item")
q.put_many(["a", "b", "c"])
item = q.get(timeout=5)
items = q.get_many(10, timeout=5)
q.len()
```

### `modal.CloudBucketMount` — S3/GCS bucket mount

```python
mount = modal.CloudBucketMount(
    "my-s3-bucket",
    secret=modal.Secret.from_name("aws-secret"),
    read_only=False,
)

@app.function(volumes={"/s3": mount})
def read_s3():
    import os
    return os.listdir("/s3")
```

---

## Web Endpoints

### FastAPI endpoint (simple)

```python
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def my_endpoint(name: str = "world") -> dict:
    return {"message": f"Hello {name}"}
```

### Full ASGI app (FastAPI, Starlette, etc.)

```python
from fastapi import FastAPI

web_app = FastAPI()

@web_app.get("/")
async def root():
    return {"status": "ok"}

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
```

### WSGI app (Flask, Django, etc.)

```python
from flask import Flask
flask_app = Flask(__name__)

@app.function()
@modal.wsgi_app()
def flask_wrapper():
    return flask_app
```

### Raw HTTP server

```python
@app.function()
@modal.web_server(port=8000)
def start_server():
    import subprocess
    subprocess.Popen(["python", "-m", "http.server", "8000"])
```

---

## Scheduling

```python
# Run every hour
@app.function(schedule=modal.Period(hours=1))
def hourly_job():
    ...

# Run on a cron schedule
@app.function(schedule=modal.Cron("0 9 * * 1-5"))  # 9am weekdays
def daily_report():
    ...
```

---

## Sandboxes (Arbitrary Code Execution)

```python
sandbox = modal.Sandbox.create(
    "bash", "-c", "echo hello",
    image=modal.Image.debian_slim(),
    app=app,
    timeout=120,
    secrets=[modal.Secret.from_name("my-secret")],
    volumes={"/data": modal.Volume.from_name("my-vol")},
)
sandbox.wait()
print(sandbox.stdout.read())
print(sandbox.returncode)
```

### Named Sandboxes

```python
# Create with a name
sb = modal.Sandbox.create("sleep", "300", name="my-worker", app=app)

# Retrieve later
sb = modal.Sandbox.from_name("my-worker")
```

### Executing commands inside a running Sandbox

```python
proc = sandbox.exec("python", "-c", "print(1+1)")
proc.wait()
print(proc.stdout.read())
```

### Filesystem operations

```python
# Read/write files
sandbox.filesystem.write_text("/tmp/hello.txt", "world")
content = sandbox.filesystem.read_text("/tmp/hello.txt")

# Copy to/from local
sandbox.filesystem.copy_to_local("/remote/path", "local_path")
sandbox.filesystem.copy_from_local("local_path", "/remote/path")

# Directory operations
sandbox.filesystem.ls("/tmp")
sandbox.filesystem.mkdir("/tmp/mydir")
sandbox.filesystem.rm("/tmp/mydir")
```

---

## Retries & Error Handling

```python
@app.function(
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,   # seconds
    )
)
def flaky_fn():
    ...
```

Custom exceptions live in `modal.exception`:
- `modal.exception.TimeoutError`
- `modal.exception.NotFoundError`
- `modal.exception.InvalidError`

---

## Networking

```python
# Static outbound IP (for IP allowlisting)
proxy = modal.Proxy.from_name("my-proxy")

@app.function(proxy=proxy)
def call_external_api():
    ...

# Expose a port from within a container
@app.function()
def expose():
    with modal.forward(port=8080) as tunnel:
        print(tunnel.url)   # public URL
        import time; time.sleep(300)
```

---

## Configuration (`modal.config`)

Configuration is set via `~/.modal.toml` or environment variables:

| Config key | Env var | Description |
|---|---|---|
| `token_id` / `token_secret` | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Auth credentials |
| `environment` | `MODAL_ENVIRONMENT` | Target environment (default or custom) |
| `traceback` | `MODAL_TRACEBACK` | Full tracebacks on CLI errors |
| `log_pattern` | `MODAL_LOG_PATTERN` | Log formatting pattern |
| `async_warnings` | `MODAL_ASYNC_WARNINGS` | Warn on blocking API use in async (default: enabled) |
| `force_build` | `MODAL_FORCE_BUILD` | Skip Image cache; rebuild all layers |

```python
# Programmatic config access
from modal.config import config
val = config.get("environment")
```

---

## CLI Reference

| Command | Usage |
|---|---|
| `modal run file.py::fn_name` | Run a function or local entrypoint |
| `modal deploy file.py` | Deploy an App persistently |
| `modal serve file.py` | Serve a web endpoint with hot reload |
| `modal shell file.py::fn_name` | Interactive shell with same image/volumes as function |
| `modal container list` | List running containers |
| `modal container logs <id>` | Stream container logs |
| `modal volume list` | List Volumes |
| `modal volume get <vol> <path>` | Download a file from a Volume |
| `modal volume put <vol> <local> <remote>` | Upload a file to a Volume |
| `modal dict list` | List Dicts |
| `modal queue list` | List Queues |
| `modal secret list` | List Secrets |
| `modal secret create <name> KEY=val ...` | Create a Secret |
| `modal deploy --env <env> file.py` | Deploy to a specific environment |
| `modal token new` | Create a new auth token |
| `modal token set --token-id X --token-secret Y` | Set credentials |
| `modal setup` | Interactive first-time setup |
| `modal app list` | List deployed Apps |
| `modal app stop <id>` | Stop a deployed App |

**Referring to functions in CLI:**
```bash
modal run myfile.py::my_function          # explicit function
modal run myfile.py::app.my_function      # via app variable
modal run myfile.py                        # if only one entrypoint
modal run mypackage.mymodule::my_function  # via module path
```

---

## Common Patterns

### Loading a model once (cold start optimization)

```python
@app.cls(gpu="A10G", image=image)
class Model:
    @modal.enter()
    def load(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model="gpt2")

    @modal.method()
    def generate(self, prompt: str) -> str:
        return self.pipe(prompt)[0]["generated_text"]
```

### Batch processing with map

```python
@app.function()
def process_item(item):
    return item.upper()

@app.local_entrypoint()
def main():
    items = ["a", "b", "c", "d"]
    results = list(process_item.map(items))
```

### Local entrypoint

```python
@app.local_entrypoint()
def main():
    # This runs locally; calls .remote() to execute on Modal
    output = my_function.remote(10)
    print(output)
```
Run with: `modal run file.py`

### Storing model weights in a Volume

```python
vol = modal.Volume.from_name("model-weights", create_if_missing=True)

@app.function(volumes={"/weights": vol}, image=image)
def download_weights():
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-3-8B", local_dir="/weights/llama3")
    vol.commit()

@app.cls(volumes={"/weights": vol}, gpu="A10G", image=image)
class LlamaModel:
    @modal.enter()
    def load(self):
        self.model = load_from_path("/weights/llama3")
```

### Async functions

Modal supports async Python natively:

```python
@app.function()
async def async_fn(url: str):
    import httpx
    async with httpx.AsyncClient() as client:
        return (await client.get(url)).text
```

---

## Gotchas & Best Practices

- **v1.0+ automounting removed:** Always use `.add_local_python_source("mypkg")` to include local packages — they are no longer auto-mounted.
- **`vol.commit()` is required:** After writing to a Volume inside a container, call `vol.commit()` or changes may be lost.
- **Protect `app.run()` from import:** Always wrap in `if __name__ == "__main__":` or use `@app.local_entrypoint()` to avoid running on import inside containers.
- **`modal.deploy()` is non-streaming:** Unlike `app.run()`, `app.deploy()` does not stream function logs back.
- **Pinning SDK version:** Pin Modal to a minor version (`modal~=1.0.0`) to avoid breaking changes across `1.Y.0` releases.
- **Async warnings:** If blocking Modal APIs are used in async code, you'll see runtime warnings. Address them — they often indicate real performance issues.
- **Image caching:** Images are cached by recipe. Use `force_build=True` (or `MODAL_FORCE_BUILD=1`) to force a rebuild without polluting other apps' caches.
- **`Dict.len()` is expensive:** It scans the entire Dict and caps at 100,000 — avoid calling in hot paths.
- **Secrets in build steps:** Secrets can be passed to `.run_function()` image build steps; they are not baked into the image layer.
