"""Make a model worker exit when the API server that started it goes away.

Workers (MLX voice cloning, Indic-Speak, the local language model) hold gigabytes of memory. If the API is killed
hard, they would otherwise keep running as orphans. `watch_parent()` starts a daemon thread that exits the process
as soon as it is re-parented (macOS/Linux: the parent died). Windows does not re-parent, so it is a no-op there.

Also usable as a launcher for a third-party module:
    python parent_watch.py --run-module mlx_lm server --model ... --port ...

For `mlx_lm` it also bounds memory, which otherwise grows to several GB over a long session (measured in the 45-minute
soak, see docs/RESULTS.md): MLX keeps every freed GPU buffer in a pool for reuse (limit_mlx_cache), and `mlx_lm server`
keeps up to ten whole KV caches without a byte limit (default_mlx_lm_args). Both are overridable with the
MLX_CACHE_LIMIT_MB, MLX_LM_PROMPT_CACHE_SIZE and MLX_LM_PROMPT_CACHE_BYTES environment variables, or by passing the
server flags yourself.
"""
import os
import runpy
import sys
import threading
import time


def watch_parent(poll_s: float = 2.0) -> None:
    parent = os.getppid()

    if os.name == "nt":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(parent))
            if handle:
                def win_loop():
                    # Blocks with 0% CPU until parent process exits
                    ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                    ctypes.windll.kernel32.CloseHandle(handle)
                    os._exit(0)

                threading.Thread(target=win_loop, name="parent-watch-win", daemon=True).start()
                return
        except Exception:
            pass

    def loop():
        while True:
            time.sleep(poll_s)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


DEFAULT_MLX_CACHE_LIMIT_MB = 256


def limit_mlx_cache(default_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB) -> None:
    """Cap MLX's pool of freed GPU buffers (MLX_CACHE_LIMIT_MB, default 256 MB; 0 turns the pool off).

    Without a cap the pool is only emptied under memory pressure, so a process serving requests of varying size keeps
    growing. A freed buffer that is not in the pool is simply allocated again next time, which costs microseconds."""
    try:
        import mlx.core as mx
        mx.set_cache_limit(max(0, int(os.getenv("MLX_CACHE_LIMIT_MB", default_mb))) * 2 ** 20)
    except Exception:  # noqa: BLE001  (no MLX here, or a bad value: keep MLX's own default)
        pass


def default_mlx_lm_args(args: list) -> list:
    """`mlx_lm server` arguments with a bounded prompt cache unless the caller chose one.

    The server keeps KV caches of past prompts (default: ten, no byte limit) so a repeated prefix is not computed again.
    A 3B model uses about 110 KB per token, so ten 1200-token prompts are over 1.3 GB. A few are enough for an agent's
    shared instructions, and the byte limit keeps long conversations from growing it.

    How the limit works in mlx_lm 0.32 (read in mlx_lm/server.py): `--prompt-cache-bytes` takes sizes such as 512M
    (decimal megabytes). The server builds its cache with the entry count only (LRUPromptCache(prompt_cache_size)) and
    applies the byte limit by trimming (`trim_to`) every time it admits a request, for models that batch (Llama does; not
    a draft-model or --kv-bits setup). So the cache can exceed the limit by the newest entry until the next request, and
    a server that is idle keeps what it has. Measured with the old 3,700-token prompts (330 MB of KV each) and no limit:
    the server's footprint grew about 320 MB per distinct prompt."""
    if not args or args[0] != "server":
        return args
    out = list(args)
    for flag, env, default in (("--prompt-cache-size", "MLX_LM_PROMPT_CACHE_SIZE", "4"),
                               ("--prompt-cache-bytes", "MLX_LM_PROMPT_CACHE_BYTES", "512M")):
        if not any(a == flag or a.startswith(flag + "=") for a in out):
            value = os.getenv(env, default).strip()
            if value:
                out += [flag, value]
    # Gemma 4 thinks silently by default: hundreds of hidden tokens before the first spoken word (measured 11 s to
    # first token, many empty replies at a 300-token cap). Voice replies need it off; MLX_LM_THINKING=on keeps it.
    model = next((out[i + 1] for i, a in enumerate(out[:-1]) if a == "--model"), "")
    if ("gemma-4" in model.lower() and os.getenv("MLX_LM_THINKING", "off").strip().lower() != "on"
            and not any(a == "--chat-template-args" or a.startswith("--chat-template-args=") for a in out)):
        out += ["--chat-template-args", '{"enable_thinking": false}']
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--run-module":
        sys.exit("usage: parent_watch.py --run-module <module> [args...]")
    module = sys.argv[2]
    module_args = sys.argv[3:]
    if module == "mlx_lm":
        module_args = default_mlx_lm_args(module_args)
        limit_mlx_cache()
    sys.argv = [module, *module_args]
    watch_parent()
    runpy.run_module(module, run_name="__main__", alter_sys=True)
