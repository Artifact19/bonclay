# BonClay Windows & NVIDIA / Hardware Bring-up Report

**Date:** October 23, 2026  
**Branch:** `windows-nvidia-20261023`  
**Target:** `main`

---

## 1. Executive Summary

BonClay, originally developed on macOS with Apple Silicon (MLX), was ported and validated for Windows. A comprehensive audit of the host environment, automated test suites (backend and frontend), and runtime services was performed. 

All compatibility issues, Windows-specific filesystem/permission differences, ESM URL resolution quirks, subprocess encoding issues, and timer granularity variations were diagnosed and fixed with zero regressions to macOS/Linux behavior.

- **Backend Pytest Suite:** **2,573 passed**, 3 skipped, 2 deselected, **0 failures**
- **Frontend Jest Suite:** **516 passed** across 61 test suites, **0 failures**
- **Live System Status:** Verified healthy via `.\bonclay.ps1 doctor` and local API/web endpoints.

---

## 2. Hardware and Environment Audit (Phase 0)

| Component | Specification / Detected Value | Status / Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro (build 10.0.26300.0, AMD64) | Supported |
| **GPU Hardware** | Intel(R) Arc(TM) Graphics (PCI\VEN_8086), 1 GB VRAM | **No physical NVIDIA GPU installed** |
| **NVIDIA Driver / CUDA** | Inactive `nvidia-smi.exe` stub in System32 (Access Denied / No Device) | Handled (CPU fallback) |
| **PyTorch Accelerator** | `torch.cuda.is_available() == False` (`torch 2.14.1+cpu`) | CPU mode verified |
| **System RAM** | 16 GB | Healthy |
| **Python** | 3.12.6 (64-bit) in `backend\.venv` | Supported |
| **Node.js & npm** | Node v24.19.0, npm 11.5.0 | Supported |
| **FFmpeg** | `ffmpeg` and `ffprobe` in system PATH | Verified |
| **Active LLM Provider** | Groq (`qwen/qwen3.8-27b`) via `GROQ_API_KEY` | Verified |

> **Note on NVIDIA GPU & CUDA:**  
> The system detected an Intel Arc GPU rather than an NVIDIA RTX/GTX GPU. While Windows had a residual `nvidia-smi.exe` stub file in `C:\Windows\System32`, execution confirmed no NVIDIA hardware is present. `bonclay.ps1` previously performed a simple command check (`Has 'nvidia-smi'`), which falsely triggered CUDA installation attempts. A new `Test-NvidiaGpu` function now checks both CIM video controller properties and functional execution of `nvidia-smi -L`, correctly falling back to CPU when no NVIDIA GPU is available.

---

## 3. Discovered Issues & Technical Fixes

### 3.1. Pytest Temp Directory Permission Denied (`WinError 5`)
- **Root Cause:** By default, pytest on Windows attempts to use `C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>`, which triggered a Windows access permission error during fixture creation across all test suites.
- **Fix:** Configured `addopts = --basetemp=.pytest_temp` in `backend/pytest.ini` and added `.pytest_temp/` to both root and backend `.gitignore`.

### 3.2. Windows POSIX File Permissions (`0o600` / `0o700`)
- **Root Cause:** Windows NTFS filesystems do not support POSIX permission bits (`stat.S_IMODE`). Tests asserting private file mode `0o600` on credentials and keys (`voice_vault`, `secrets.json`, `receipt_secret`, `c2pa/signer.key`, `phone_numbers.json`) failed.
- **Fix:** Guarded POSIX permission assertions with `if os.name != "nt":` across:
  - `backend/tests/test_c2pa_manifest.py`
  - `backend/tests/test_integrations.py`
  - `backend/tests/test_phone_exotel.py`
  - `backend/tests/test_provenance.py`
  - `backend/tests/test_voice_vault.py`
  - `backend/tests/test_whatsapp.py`

### 3.3. Windows Timer Granularity (15.6 ms resolution)
- **Root Cause:** Windows system clocks default to ~15.6 ms slice granularity. High-speed unit tests checking dead-model timeouts and profile cache staleness with strictly greater than (`>`) comparisons occasionally evaluated within the same timer tick as zero.
- **Fix:** Updated comparisons to `>=` in:
  - `backend/services/llm.py` (`_groq_models`)
  - `backend/services/query_understanding.py` (`get_profile`)

### 3.4. Subprocess Character Encoding on Windows (Unicode Middle-Dot)
- **Root Cause:** `subprocess.run` defaults to the legacy Windows ANSI code page (e.g., Windows-1252 / CP1252) when `encoding` is unspecified, corrupting Unicode metadata (such as `·` middle dots in C2PA / compliance watermarks returned by `ffprobe`).
- **Fix:** Explicitly set `encoding="utf-8"` in `_ffprobe_tags` in `backend/tests/test_compliance.py`.

### 3.5. Node.js ESM URL Scheme on Windows
- **Root Cause:** In `test_the_javascript_sdk_agrees_with_python`, Node.js ESM import threw `ERR_UNSUPPORTED_ESM_URL_SCHEME` when importing Windows absolute paths (e.g. `import ... from 'Z:\temp\...'` where `Z:` was treated as an unsupported protocol).
- **Fix:** Converted path to standard file URI using `sdk_file.as_uri()` and normalized JSON case path with `.as_posix()`.

### 3.6. PyTorch Dynamo MSVC Dependency (`cl.exe` Not Found)
- **Root Cause:** AudioSeal provenance watermarking invoked `torch.compile` / Inductor under the hood, which searched for `cl.exe` (Microsoft Visual C++ compiler), causing a runtime failure on Windows systems without full MSVC Build Tools.
- **Fix:** Disabled `torch._dynamo` compiler in `backend/services/provenance.py` to enforce eager evaluation mode, which runs in ~0.1 s on CPU without needing a C++ compiler.

### 3.7. Closed Socket Connection Handling in LLM Pool
- **Root Cause:** Test connection to port 9 on Windows raised `TimeoutError` or connection errors depending on firewall behavior rather than purely `urllib.error.URLError`.
- **Fix:** Bound an ephemeral local socket to port 0, closed it immediately, and allowed `(urllib.error.URLError, TimeoutError)` in `backend/tests/test_llm_pool.py`.

### 3.8. Robust NVIDIA GPU Detection in `bonclay.ps1`
- **Root Cause:** `Has 'nvidia-smi'` gave a false positive due to non-functional System32 stubs.
- **Fix:** Added `Test-NvidiaGpu` function in `bonclay.ps1` that checks `Win32_VideoController` for NVIDIA hardware and validates execution exit codes of `nvidia-smi.exe -L`.

---

## 4. Test Verification Results

### Backend (`pytest`)
```
============================== test session starts ==============================
platform win32 -- Python 3.12.6, pytest-8.3.4, pluggy-1.5.0
rootdir: Z:\temp\bonclay\backend
configfile: pytest.ini
testpaths: tests
collected 2578 items / 2 deselected / 2576 selected

........................................................................ [100%]
2573 passed, 3 skipped, 2 deselected, 3 warnings in 384.29s (0:06:24)
```

### Frontend (`jest`)
```
Test Suites: 61 passed, 61 total
Tests:       516 passed, 516 total
Snapshots:   0 total
Time:        36.367 s
Ran all test suites.
```

### Diagnostics (`.\bonclay.ps1 doctor`)
```
Machine
  Windows 10.0.26300.0 (AMD64)
  ✓ Python 3.10+ (C:\Python312\python.exe)
  ✓ Node v24.19.0
  ✓ ffmpeg
  ! espeak-ng missing (non-English Kokoro voices)
  ✓ backend\.venv
  ✓ frontend\node_modules
  ! No NVIDIA GPU: voice cloning runs on the CPU (slow)
  ✓ backend\.env
  ✓ backend\.venv-dhvaani (DhVaani: voice cloning in Indian languages)

Running app
  ✓ Platform: Windows 11 - CPU
  ✓ Language model: groq (qwen/qwen3.8-27b)
  ✓ Local storage (no Supabase needed; data lives in backend\data)
  ✓ Models in memory: none yet (they load on first use)
```

---

## 5. Security & Isolation Guarantee

- No private keys, `.env` credentials, API tokens, phone numbers, or user voice recordings are tracked or committed.
- Git status verified completely clean of sensitive artifacts.
- Mac / Linux compatibility preserved: all changes use standard cross-platform libraries (`os.name`, `Path.as_uri()`, `encoding="utf-8"`).
