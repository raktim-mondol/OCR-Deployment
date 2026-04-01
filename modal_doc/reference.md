
## 🏗️ Application Construction

### `modal.App`
The main unit of deployment for code on Modal. Key methods include:
- **`App(name, ...)`** — Construct a new app, optionally with default image, mounts, secrets, or volumes
- **`.name`** — The user-provided name of the App
- **`.app_id`** — Returns the `app_id` of a running or stopped app
- **`.from_name(name)`** — Look up an App with a given name, creating a new one if necessary (results in a deployed-state App)
- **`.run()`** — Context manager that runs an ephemeral app on Modal; the main entry point
- **`.deploy()`** — Deploy the App so it's available persistently; programmatic equivalent of `modal deploy`
- **`@app.function()`** — Decorator to register a new Modal Function with this App
- **`@app.cls()`** — Decorator to register a new Modal Cls with this App
- **`.include(other_app)`** — Include another App's objects in this one
- **`.set_tags(tags)`** / **`.get_tags()`** — Attach/retrieve key-value metadata on the App

---

## ⚡ Serverless Execution

### `modal.Function`
A serverless function backed by an autoscaling container pool. Generally constructed via `@app.function()`, not directly. Key methods:
- **`.hydrate()`** — Sync local object with server identity
- **`.update_autoscaler(...)`** — Override current autoscaler behavior at runtime
- **`.from_name(name)`** — Reference a Function from a deployed App by name
- **`.remote(*args)`** — Call the function remotely and wait for result
- **`.remote_gen(*args)`** — Call as a generator remotely
- **`.local(*args)`** — Execute in the same environment as the caller
- **`.spawn(*args)`** — Call without waiting; returns a `FunctionCall` handle
- **`.get_raw_f()`** — Return the inner Python function
- **`.get_current_stats()`** — Return a `FunctionStats` object with queue/runner counts
- **`.map(inputs, ...)`** — Parallel map; output order matches input order by default; set `order_outputs=False` for completion order; `return_exceptions=True` treats exceptions as results
- **`.starmap(inputs)`** — Like `map` but spreads arguments over function parameters
- **`.for_each(inputs)`** — Map without collecting results
- **`.spawn_map(inputs)`** — Spawn map without waiting

### `modal.Cls`
A serverless class supporting parametrization and lifecycle hooks. Registered via `@app.cls()`.

### `modal.FunctionCall`
A handle returned by `.spawn()` that can be polled or awaited. Conceptually similar to `multiprocessing.pool.apply_async` or a Future/Promise.

---

## 🔧 Extended Function Configuration

### Lifecycle Hooks
- **`@modal.enter`** — Decorator for a method executed during container startup
- **`@modal.exit`** — Decorator for a method executed during container shutdown
- **`@modal.method`** — Decorator for exposing a class method as an invokable function

### Class Parametrization
- **`modal.parameter`** — Defines class parameters, akin to a `dataclass` field

### Web Integrations
- **`@modal.fastapi_endpoint`** — Decorator for exposing a simple FastAPI-based endpoint
- **`@modal.asgi_app`** — Decorator for registering an ASGI app. Supports all popular Python web libraries; gives full flexibility for defining one or more web endpoints.
- **`@modal.wsgi_app`** — Decorator for WSGI apps. Note: WSGI has been superseded by ASGI, which is preferred.
- **`@modal.web_server`** — Decorator for functions that construct an HTTP web server

### Function Semantics
- **`@modal.batched`** — Enables dynamic input batching
- **`@modal.concurrent`** — Enables input concurrency per container

### Scheduling
- **`modal.Cron(expr)`** — Schedule using cron syntax
- **`modal.Period(seconds=...)`** — Schedule at a fixed interval

### Exception Handling
- **`modal.Retries`** — Define retry policy for input failures

---

## 📦 Container Configuration

### `modal.Image`
Base class for container images. Do not construct directly — use static factory methods:
- **`Image.debian_slim()`** — Default image based on official Python Docker images
- **`Image.from_registry(tag)`** — Use an existing container image
- **`Image.micromamba()`** — Micromamba-based image
- **`.pip_install(*packages)`** — Install Python packages via pip
- **`.pip_install_from_requirements(path)`** — Install from a local `requirements.txt`
- **`.pip_install_from_pyproject(path)`** — Install from `pyproject.toml`
- **`.pip_install_private_repos(*repos, secret)`** — Install from private GitHub/GitLab repos
- **`.apt_install(*packages)`** — Install Debian packages
- **`.run_function(raw_f, ...)`** — Run a user-defined function as an image build step; filesystem changes are captured as a new Image layer
- **`.add_local_file(local_path, remote_path, copy=False)`** — Add a local file; `copy=True` bakes it into the image layer
- **`.env(vars)`** — Set environment variables
- **`.workdir(path)`** — Set working directory

### `modal.Secret`
A pointer to secrets exposed as environment variables. Methods:
- **`Secret.from_dict(d)`** — Create a secret from a `str→str` dictionary
- **`Secret.from_local_environ(keys)`** — Create from local environment variables
- **`Secret.from_dotenv(path)`** — Create from a `.env` file
- **`Secret.from_name(name)`** — Reference a secret by name
- **`.info()`** — Return information about the Secret
- **`.update(...)`** — Add or overwrite key-value pairs

---

## 🗄️ Data Primitives

### Persistent Storage

**`modal.Volume`**
Distributed storage for high-throughput parallel reads. Key methods include `.from_name()`, `.from_id()`, `.read_only()`, `.info()`, and `.name`.

**`modal.CloudBucketMount`**
Storage backed by a third-party cloud bucket (S3, GCS, etc.).

**`modal.NetworkFileSystem`**
Shared, writeable cloud storage — superseded by `modal.Volume`.

### In-Memory Storage

**`modal.Dict`**
A distributed key-value store. Key methods:
- **`Dict.from_name(name, create_if_missing=True)`** — Reference (and optionally create) a named Dict
- **`.info()`** — Return metadata
- **`.clear()`** — Remove all items
- **`.get(key, default=None)`** — Get value by key
- **`.contains(key)`** — Check if key exists
- **`.len()`** — Return length (expensive; max 100,000)
- **`.update(other)`** — Update with additional items
- **`.put(key, value, skip_if_exists=False)`** — Insert a value

**`modal.Queue`**
A distributed FIFO queue. Supports `.from_name()`, `.from_id()`, `.info()`, and `.name`.

---

## 🏖️ Sandboxed Execution

### `modal.Sandbox`
An interface for restricted code execution. Key methods:
- **`Sandbox.create(...)`** — Create a new Sandbox (container created asynchronously); supports `name=` for named sandboxes
- **`Sandbox.from_name(name)`** — Retrieve a named Sandbox
- **`Sandbox.from_id(id)`** — Reference a Sandbox by ID
- **`.hydrate()`** — Sync with server
- **`.wait()`** — Block until Sandbox finishes
- **`.poll()`** — Check if finished; returns `None` if running, else exit code
- **`.exec(cmd)`** — Execute a command, returns `ContainerProcess`
- **`.terminate()`** — Stop the Sandbox
- **`.tunnels()`** — Network tunnels into the Sandbox
- **`.create_connect_token(...)`** — Auth token for HTTP/WebSocket access
- **`.reload_volumes()`** — Reload mounted Volumes
- **`.detach()`** — Disconnect client from Sandbox
- **`.snapshot_filesystem()`** — Snapshot the filesystem
- **`.filesystem`** — Namespace for filesystem APIs: `.read_bytes()`, `.read_text()`, `.write_bytes()`, `.write_text()`, `.copy_to_local()`, `.copy_from_local()`, `.open()`, `.ls()`, `.mkdir()`, `.rm()`, `.watch()`
- **`.stdout`** / **`.stderr`** / **`.stdin`** / **`.returncode`**

### `modal.SandboxSnapshot`
A snapshot of a Sandbox filesystem state.

### `modal.container_process.ContainerProcess`
An object representing a sandboxed process.

### `modal.file_io.FileIO`
A handle for a file in the Sandbox filesystem.

---

## 🌐 Networking

- **`modal.Proxy`** — Provides a static outbound IP address for containers
- **`modal.Tunnel`** — Expose a port from a container publicly
- **`modal.forward(port)`** — Context manager for publicly exposing a port

---

## ⚙️ Configuration

### `modal.config`
Key config options (set via `.modal.toml` or environment variables):
- **`MODAL_TRACEBACK`** — Enable full tracebacks on CLI errors
- **`MODAL_LOG_PATTERN`** — Customize log formatting
- **`MODAL_ASYNC_WARNINGS`** — Warn when blocking Modal APIs are used in async contexts
- **`force_build`** / `MODAL_FORCE_BUILD` — Ignore Image cache and rebuild all layers

Class: `modal.config.Config` with methods `.get()`, `.override_locally()`, `.to_dict()`
Functions: `config_profiles()`, `config_set_active_profile()`

---

## 🖥️ CLI Reference

| Command | Purpose |
|---|---|
| `modal run` | Run a Modal function or local entrypoint |
| `modal deploy` | Deploy an App persistently |
| `modal serve` | Serve a web endpoint |
| `modal shell` | Start an interactive shell using the spec of a function (same image, volumes, mounts) |
| `modal container` | Manage running containers |
| `modal volume` | Read and edit Volumes |
| `modal dict` | Manage Dicts |
| `modal queue` | Manage Queues |
| `modal secret` | Manage Secrets |
| `modal token` | Manage auth tokens |
| `modal environment` | Manage environments |
| `modal profile` | Manage config profiles |
| `modal config` | View/edit config |
| `modal app` | Manage Apps |
| `modal dashboard` | Open the web dashboard |
| `modal setup` | Initial setup wizard |

---

The full live reference is at **[modal.com/docs/reference](https://modal.com/docs/reference)**. Each class and decorator has its own detailed page linked from there.