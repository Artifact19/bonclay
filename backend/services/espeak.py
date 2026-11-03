"""Point Kokoro's phonemizer at a working espeak-ng before Kokoro loads.

misaki (Kokoro's G2P) forces the espeak-ng library bundled in the `espeakng-loader` wheel. On macOS
arm64 that build reads its data from a path baked in at build time
(/Users/runner/work/espeakng-loader/...), and espeak-ng exits the whole process when the file is
missing, which takes the API server down with it. When a system espeak-ng is available
(Homebrew, apt, or ESPEAK_NG_LIBRARY / ESPEAK_NG_DATA), use it instead.
"""
import logging
import os

log = logging.getLogger("bonclay.espeak")

_CANDIDATES = [
    ("/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"),
    ("/usr/local/lib/libespeak-ng.dylib", "/usr/local/share/espeak-ng-data"),
    ("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
    ("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1", "/usr/lib/aarch64-linux-gnu/espeak-ng-data"),
]

# Standard Windows installation candidates
_prog = os.environ.get("ProgramFiles", r"C:\Program Files")
_prog86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
_WINDOWS_CANDIDATES = [
    (os.path.join(_prog, "eSpeak NG", "libespeak-ng.dll"), os.path.join(_prog, "eSpeak NG", "espeak-ng-data")),
    (os.path.join(_prog, "eSpeak NG", "espeak-ng.dll"), os.path.join(_prog, "eSpeak NG", "espeak-ng-data")),
    (os.path.join(_prog86, "eSpeak NG", "libespeak-ng.dll"), os.path.join(_prog86, "eSpeak NG", "espeak-ng-data")),
    (os.path.join(_prog86, "eSpeak NG", "espeak-ng.dll"), os.path.join(_prog86, "eSpeak NG", "espeak-ng-data")),
]

_configured = False


def configure_espeak() -> None:
    """Idempotent. Call before `from kokoro import KPipeline`."""
    global _configured
    if _configured:
        return
    _configured = True
    candidates = list(_CANDIDATES)
    if os.name == "nt":
        candidates.extend(_WINDOWS_CANDIDATES)
        import shutil
        found = shutil.which("espeak-ng")
        if found:
            bin_dir = os.path.dirname(found)
            for lib_name in ("libespeak-ng.dll", "espeak-ng.dll"):
                lib_path = os.path.join(bin_dir, lib_name)
                data_path = os.path.join(bin_dir, "espeak-ng-data")
                if not os.path.isdir(data_path):
                    data_path = os.path.join(os.path.dirname(bin_dir), "espeak-ng-data")
                candidates.insert(0, (lib_path, data_path))
    pairs = [(os.environ.get("ESPEAK_NG_LIBRARY"), os.environ.get("ESPEAK_NG_DATA"))] + candidates
    for lib, data in pairs:
        if lib and data and os.path.isfile(lib) and os.path.isdir(data):
            try:
                import misaki.espeak  # noqa: F401  (sets the bundled library; overridden below)
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
            except Exception as e:  # noqa: BLE001  kokoro/misaki not installed: nothing to configure
                log.debug("espeak configuration skipped: %s", e)
                return
            EspeakWrapper.set_library(lib)
            EspeakWrapper.set_data_path(data)
            log.info("Using system espeak-ng: %s", lib)
            return
    log.info("No system espeak-ng found; using the espeakng-loader bundle")
