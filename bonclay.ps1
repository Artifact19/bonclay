# BonClay for Windows 10/11: one command to set up, start, stop and check the whole studio.
# Works in Windows PowerShell 5.1 and PowerShell 7. macOS and Linux use ./bonclay instead.
#
#   powershell -ExecutionPolicy Bypass -File .\bonclay.ps1 setup [-Indic] [-Chatterbox] [-Dhvaani]   install everything;
#   -Indic adds Indic-Speak, -Chatterbox adds Hindi voice cloning, -Dhvaani adds voice cloning in Indian languages
#   ... start | stop | restart | status | doctor | logs [api|web] | test | warm [--clone]
#
# Ports: $env:BONCLAY_API_PORT (default 8000) and $env:BONCLAY_WEB_PORT (default 3000).
# This file is ASCII-only on purpose, so Windows PowerShell 5.1 reads it correctly.

param(
    [Parameter(Position = 0)]
    [string]$Command = 'help',
    [Parameter(Position = 1)]
    [string]$Target = '',
    [switch]$Indic,
    [switch]$Chatterbox,
    [switch]$Dhvaani,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ProgressPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root     = $PSScriptRoot
$Backend  = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$VenvDir  = Join-Path $Backend '.venv'
$VenvPy   = Join-Path $VenvDir 'Scripts\python.exe'
$IndicDir = Join-Path $Backend '.venv-indic'
$ChatterboxDir = Join-Path $Backend '.venv-chatterbox'
$DhvaaniDir = Join-Path $Backend '.venv-dhvaani'
$EnvFile  = Join-Path $Backend '.env'
$RunDir   = Join-Path $Root '.bonclay'
$LogDir   = Join-Path $RunDir 'logs'

$ApiPort = if ($env:BONCLAY_API_PORT) { [int]$env:BONCLAY_API_PORT } else { 8000 }
$WebPort = if ($env:BONCLAY_WEB_PORT) { [int]$env:BONCLAY_WEB_PORT } else { 3000 }
$ApiUrl  = "http://localhost:$ApiPort"
$WebUrl  = "http://localhost:$WebPort"

$Tick  = [string][char]0x2713
$Cross = [string][char]0x2717
$Dot   = [string][char]0x00B7

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ------------------------------------------------------------------------------------------- output helpers
function Ok   { param([string]$Text) Write-Host ("  {0} {1}" -f $Tick, $Text) -ForegroundColor Green }
function Warn { param([string]$Text) Write-Host ("  ! {0}" -f $Text) -ForegroundColor Yellow }
function Bad  { param([string]$Text) Write-Host ("  {0} {1}" -f $Cross, $Text) -ForegroundColor Red }
function Fix  { param([string]$Text) Write-Host ("    fix: {0}" -f $Text) -ForegroundColor DarkGray }
function Step { param([string]$Text) Write-Host ''; Write-Host $Text -ForegroundColor White }

function Has {
    param([string]$Name)
    [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Espeak {
    (Has 'espeak-ng') -or (Test-Path (Join-Path $env:ProgramFiles 'eSpeak NG\espeak-ng.exe'))
}

function Test-NvidiaGpu {
    if (-not (Has 'nvidia-smi')) { return $false }
    try {
        $p = Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '-L' -NoNewWindow -Wait -PassThru -ErrorAction Stop
        if ($p.ExitCode -ne 0) { return $false }
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like '*NVIDIA*' }
        if ($gpus) { return $true }
        $out = (& 'nvidia-smi.exe' '-L' 2>$null | Out-String)
        return ($out -like '*NVIDIA*' -or $out -like '*GeForce*' -or $out -like '*RTX*')
    } catch {
        return $false
    }
}

function Get-DotEnv {
    # Reads KEY=value from backend\.env as plain text. The file is never executed.
    param([string]$Key)
    if (-not (Test-Path $EnvFile)) { return '' }
    $pattern = '^\s*' + [regex]::Escape($Key) + '\s*=\s*(.*?)\s*$'
    foreach ($line in (Get-Content -Path $EnvFile)) {
        if ($line -match $pattern) { return $Matches[1].Trim([char]'"', [char]"'") }
    }
    return ''
}

# ------------------------------------------------------------------------------------------- python and processes
function Test-PythonCommand {
    # Asks an interpreter for its full path. Returns that path only when it is Python 3.10 or newer.
    param([string]$Exe, [string]$Version = '')
    $code = "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')"
    if ($Version) { $out = & $Exe $Version -c $code 2>$null } else { $out = & $Exe -c $code 2>$null }
    if ($LASTEXITCODE -ne 0) { return $null }
    $found = @($out) | Select-Object -First 1
    if ($found -and (Test-Path -LiteralPath ([string]$found))) { return [string]$found }
    return $null
}

function Find-Python {
    # Order: py -3.12, py -3.11, py -3.10, then python. Returns the full path of the interpreter, or $null.
    if (Has 'py') {
        foreach ($version in @('-3.12', '-3.11', '-3.10')) {
            $found = Test-PythonCommand -Exe 'py' -Version $version
            if ($found) { return $found }
        }
    }
    if (Has 'python') {
        $found = Test-PythonCommand -Exe 'python'
        if ($found) { return $found }
    }
    return $null
}

function Get-PidOf {
    param([string]$Name)
    $file = Join-Path $RunDir "$Name.pid"
    if (-not (Test-Path $file)) { return $null }
    $raw = Get-Content -Path $file -TotalCount 1
    $number = 0
    if ([int]::TryParse(("$raw").Trim(), [ref]$number)) { return $number }
    return $null
}

function Test-Alive {
    param([string]$Name)
    $savedPid = Get-PidOf $Name
    [bool]($savedPid -and (Get-Process -Id $savedPid -ErrorAction SilentlyContinue))
}

function Test-PortBusy {
    param([int]$Port)
    try { [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop) } catch { $false }
}

function Start-Hidden {
    # Runs a command through cmd.exe in a hidden window. cmd.exe is the root of the process tree,
    # so taskkill /T can stop the API or the web app together with its children.
    param([string]$Command, [string]$WorkDir, [hashtable]$Vars = @{})
    $saved = @{}
    foreach ($key in $Vars.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, [string]$Vars[$key], 'Process')
    }
    try {
        return Start-Process -FilePath 'cmd.exe' -ArgumentList ('/d /c "{0}"' -f $Command) -WorkingDirectory $WorkDir -WindowStyle Hidden -PassThru
    } finally {
        foreach ($key in $Vars.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process')
        }
    }
}

function Wait-Http {
    param([string]$Url, [int]$Seconds, $Proc)
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $Seconds) {
        if ($Proc -and $Proc.HasExited) { return $false }
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 15 | Out-Null
            return $true
        } catch {
            Start-Sleep -Milliseconds 700
        }
    }
    return $false
}

function Save-Pid {
    param([string]$Name, $Proc)
    Set-Content -Path (Join-Path $RunDir "$Name.pid") -Value "$($Proc.Id)" -Encoding ASCII
}

# ------------------------------------------------------------------------------------------- setup
function Invoke-IndicSetup {
    param([string]$BasePy)
    Step 'Indic-Speak (22 Indian languages, needs HF_TOKEN with access granted)'

    $hfToken = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { Get-DotEnv 'HF_TOKEN' }
    if (-not $hfToken) {
        Warn 'HF_TOKEN is not set, so Indic-Speak was skipped (Hindi still works through Kokoro)'
        Fix 'request access on https://huggingface.co/bodhan-ai/indic-speak, put HF_TOKEN=hf_... in backend\.env, then run setup -Indic again'
        return
    }

    $indicPy = Join-Path $IndicDir 'Scripts\python.exe'
    if (-not (Test-Path $indicPy)) {
        Write-Host "  Creating virtualenv at $IndicDir"
        & $BasePy -m venv $IndicDir
        if ($LASTEXITCODE -ne 0) { Warn 'Could not create backend\.venv-indic'; return }
    }

    Write-Host '  Installing Indic-Speak packages (this can take a few minutes)'
    & $indicPy -m pip install -q --upgrade pip
    & $indicPy -m pip install -q 'transformers>=5' torch snac soundfile numpy huggingface_hub
    if ($LASTEXITCODE -ne 0) { Warn 'Indic-Speak packages did not install (Hindi still works through Kokoro)'; return }

    $env:HF_TOKEN = $hfToken
    $model  = if ($env:INDIC_SPEAK_MODEL) { $env:INDIC_SPEAK_MODEL } else { 'bodhan-ai/indic-speak' }
    $worker = Join-Path $Backend 'services\indic_speak_worker.py'

    Write-Host "  Downloading $model (about 7 GB; resumes if interrupted)"
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        & $indicPy $worker --download --model $model
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            $downloaded = $true
        } elseif ($code -eq 2) {
            Warn 'Access to bodhan-ai/indic-speak is not granted yet'
            Fix 'request access on the model page, then run setup -Indic again'
            return
        } elseif ($attempt -lt 3) {
            Warn "Download failed (attempt $attempt of 3). Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
    if (-not $downloaded) {
        Warn "Could not download $model after 3 attempts"
        Fix 'check your network, then run setup -Indic again'
        return
    }

    Ok 'Indic-Speak is ready'
    Write-Host '  Licence: Indic Open Model License v1.0. Show "Built with Indic-Speak from Bodhan AI / AI4Bharat." wherever its output reaches users.'
}

function Invoke-ChatterboxSetup {
    # The PowerShell twin of backend/scripts/setup_chatterbox.sh: its own virtualenv, then the model download.
    param([string]$BasePy)
    Step 'Chatterbox (Hindi voice cloning, about 3 GB, no token needed)'

    $chatterboxPy = Join-Path $ChatterboxDir 'Scripts\python.exe'
    if (-not (Test-Path $chatterboxPy)) {
        Write-Host "  Creating virtualenv at $ChatterboxDir"
        & $BasePy -m venv $ChatterboxDir
        if ($LASTEXITCODE -ne 0) { Warn 'Could not create backend\.venv-chatterbox; cloned voices in Hindi stay unavailable'; return }
    }

    Write-Host '  Installing Chatterbox packages (this can take several minutes)'
    & $chatterboxPy -m pip install -q --upgrade pip
    # setuptools<81: newer releases drop pkg_resources, which the Perth watermarker needs.
    & $chatterboxPy -m pip install -q 'chatterbox-tts==0.1.7' 'setuptools<81' soundfile numpy hf_transfer
    if ($LASTEXITCODE -ne 0) { Warn 'Chatterbox packages did not install; cloned voices in Hindi stay unavailable'; return }

    $model  = if ($env:CHATTERBOX_MODEL) { $env:CHATTERBOX_MODEL } else { 'ResembleAI/chatterbox' }
    $worker = Join-Path $Backend 'services\chatterbox_worker.py'
    $env:HF_HUB_ENABLE_HF_TRANSFER = '1'
    $env:HF_HUB_DISABLE_XET = '1'

    Write-Host "  Downloading $model (about 3 GB; resumes if interrupted)"
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        & $chatterboxPy $worker --download --model $model
        if ($LASTEXITCODE -eq 0) {
            $downloaded = $true
        } elseif ($attempt -lt 3) {
            Warn "Download failed (attempt $attempt of 3). Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
    if (-not $downloaded) {
        Warn "Could not download $model after 3 attempts; cloned voices in Hindi stay unavailable"
        Fix 'check your network, then run setup -Chatterbox again'
        return
    }

    Ok 'Chatterbox is ready (the backend finds backend\.venv-chatterbox on its own)'
    Write-Host '  It runs on the CPU, about 4-6 s of compute per second of speech: use it for Studio renders, not live calls. Licence: MIT (Resemble AI).'
}

function Invoke-DhvaaniSetup {
    # The PowerShell twin of backend/scripts/setup_dhvaani.sh: its own virtualenv, the pinned packages, the gated model
    # download and a one-sentence check.
    param([string]$BasePy)
    Step 'DhVaani (voice cloning in Indian languages, about 0.5 GB, needs HF_TOKEN with access granted by ARTPARK-IISc)'

    $hfToken = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { Get-DotEnv 'HF_TOKEN' }
    if (-not $hfToken) {
        Warn 'HF_TOKEN is not set, so DhVaani was skipped (Hindi can still be cloned with Chatterbox)'
        Fix 'ask ARTPARK-IISc for access to https://huggingface.co/ARTPARK-IISc/DhVaani-0.5, put HF_TOKEN=hf_... in backend\.env, then run setup -Dhvaani again'
        return
    }

    $dhvaaniPy = Join-Path $DhvaaniDir 'Scripts\python.exe'
    $useUv = Has 'uv'
    if (-not (Test-Path $dhvaaniPy)) {
        Write-Host "  Creating virtualenv at $DhvaaniDir"
        if ($useUv) { & uv venv --python 3.11 $DhvaaniDir } else { & $BasePy -m venv $DhvaaniDir }
        if ($LASTEXITCODE -ne 0) { Warn 'Could not create backend\.venv-dhvaani; cloned voices in Indian languages stay unavailable'; return }
    }

    Write-Host '  Installing DhVaani packages (this can take several minutes)'
    # The model card lists neither tensorboard nor torchcodec; it needs both.
    if (Test-NvidiaGpu) {
        $torchArgs = @('torch', 'torchaudio', 'torchcodec', '--index-url', 'https://download.pytorch.org/whl/cu128', '--extra-index-url', 'https://pypi.org/simple')
    } else {
        $torchArgs = @('torch==2.14.1', 'torchaudio==2.11.0', 'torchcodec==0.17.0')
    }
    $otherArgs = @('numpy==2.4.6', 'soundfile==0.14.0', 'safetensors==0.8.0', 'einops==0.8.2', 'vocos==0.1.0', 'tensorboard==2.21.0', 'pydub==0.25.1', 'scipy', 'huggingface_hub==2.2.0', 'hf_transfer')
    if ($useUv) {
        & uv pip install --python $dhvaaniPy @torchArgs
        if ($LASTEXITCODE -eq 0) { & uv pip install --python $dhvaaniPy @otherArgs }
    } else {
        & $dhvaaniPy -m pip install -q --upgrade pip
        & $dhvaaniPy -m pip install -q @torchArgs
        if ($LASTEXITCODE -eq 0) { & $dhvaaniPy -m pip install -q @otherArgs }
    }
    if ($LASTEXITCODE -ne 0) { Warn 'DhVaani packages did not install; cloned voices in Indian languages stay unavailable'; return }

    $env:HF_TOKEN = $hfToken
    $model  = if ($env:DHVAANI_MODEL) { $env:DHVAANI_MODEL } else { 'ARTPARK-IISc/DhVaani-0.5' }
    $worker = Join-Path $Backend 'services\dhvaani_worker.py'
    $env:HF_HUB_ENABLE_HF_TRANSFER = '1'
    $env:HF_HUB_DISABLE_XET = '1'

    Write-Host "  Downloading $model (about 0.5 GB, plus the vocoder; resumes if interrupted)"
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        & $dhvaaniPy $worker --download --model $model
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            $downloaded = $true
        } elseif ($code -eq 2) {
            Warn "Access to $model is not granted for this token"
            Fix 'ask ARTPARK-IISc for access on the model page, then run setup -Dhvaani again'
            return
        } elseif ($attempt -lt 3) {
            Warn "Download failed (attempt $attempt of 3). Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
    if (-not $downloaded) {
        Warn "Could not download $model after 3 attempts; cloned voices in Indian languages stay unavailable"
        Fix 'check your network, then run setup -Dhvaani again'
        return
    }

    Write-Host '  Checking that it speaks (one Hindi sentence)'
    & $dhvaaniPy $worker --selftest --model $model
    if ($LASTEXITCODE -ne 0) { Warn 'The model downloaded but did not produce speech; see the messages above'; return }

    Ok 'DhVaani is ready (the backend finds backend\.venv-dhvaani on its own)'
    Write-Host '  Credit: Voice cloning in Indian languages by DhVaani from ARTPARK-IISc. Licence notes: docs/decisions/indian-language-cloning.md.'
}

function Invoke-Setup {
    param([switch]$WithIndic, [switch]$WithChatterbox, [switch]$WithDhvaani)

    Step ("Checking the machine (Windows {0}, {1})" -f [Environment]::OSVersion.Version, $env:PROCESSOR_ARCHITECTURE)
    $basePy = Find-Python
    if (-not $basePy) {
        Bad 'Python 3.10 or newer is required'
        Fix 'winget install Python.Python.3.12   (then open a new PowerShell window)'
        exit 1
    }
    $pyVersion = (& $basePy -c 'import platform; print(platform.python_version())' 2>$null | Select-Object -First 1)
    Ok ("Python {0}  ({1})" -f $pyVersion, $basePy)

    if (Has 'node') {
        Ok ("Node {0}" -f (node -v))
    } else {
        Bad 'Node.js 20+ is required'
        Fix 'winget install OpenJS.NodeJS.LTS'
        exit 1
    }

    if (Has 'ffmpeg') { Ok 'ffmpeg' } else { Warn 'ffmpeg is missing (needed to read uploaded audio)'; Fix 'winget install Gyan.FFmpeg' }
    if (Test-Espeak) { Ok 'espeak-ng' } else { Warn 'espeak-ng is missing (needed for non-English Kokoro voices)'; Fix 'download the Windows installer from https://github.com/espeak-ng/espeak-ng/releases' }

    Step 'Backend environment (backend\.venv)'
    if (-not (Test-Path $VenvPy)) {
        & $basePy -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Bad 'Could not create backend\.venv'; exit 1 }
    }
    & $VenvPy -m pip install -q --upgrade pip

    if (Test-NvidiaGpu) {
        Write-Host '  NVIDIA GPU found: installing PyTorch with CUDA'
        & $VenvPy -m pip install -q torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
        if ($LASTEXITCODE -ne 0) { Warn 'CUDA PyTorch did not install; continuing with the default PyTorch' }
    }

    $reqFile        = Join-Path $Backend 'requirements-dev.txt'
    $constraintFile = Join-Path $Backend 'constraints.txt'
    $pipArgs = @('-q', '-r', $reqFile, '-c', $constraintFile)
    if (Test-NvidiaGpu) {
        $pipArgs += @('--extra-index-url', 'https://download.pytorch.org/whl/cu121')
    }
    & $VenvPy -m pip install @pipArgs
    if ($LASTEXITCODE -eq 0) { Ok 'Python packages installed' } else { Bad 'pip install failed (see the output above)'; exit 1 }

    if (-not (Test-Path $EnvFile)) {
        $example = Join-Path $Backend '.env.example'
        if (Test-Path $example) {
            Copy-Item -Path $example -Destination $EnvFile
            Ok 'Created backend\.env from the example (add keys there when you have them)'
        } else {
            Warn 'backend\.env.example is missing; create backend\.env by hand'
        }
    } else {
        Ok 'backend\.env exists (left unchanged)'
    }

    Step 'Web app (frontend\node_modules)'
    Push-Location $Frontend
    & npm.cmd install --no-audit --no-fund --loglevel=error
    $npmCode = $LASTEXITCODE
    Pop-Location
    if ($npmCode -eq 0) { Ok 'npm packages installed' } else { Bad 'npm install failed'; exit 1 }

    Step 'Voice cloning and the language model'
    if (Test-NvidiaGpu) { Ok 'Voice cloning uses the NVIDIA GPU (CUDA)' } else { Warn 'No NVIDIA GPU found: voice cloning runs on the CPU (slow)' }
    Warn 'There is no MLX on Windows, so agents need a language model from one of these:'
    Fix 'set GROQ_API_KEY in backend\.env, or run Ollama or LM Studio and set LLM_BASE_URL=http://localhost:11434/v1 and LLM_MODEL=llama3.2 in backend\.env'

    if ($WithIndic) { Invoke-IndicSetup -BasePy $basePy }
    if ($WithChatterbox) { Invoke-ChatterboxSetup -BasePy $basePy }
    if ($WithDhvaani) { Invoke-DhvaaniSetup -BasePy $basePy }

    Step 'Done. Start everything with: .\bonclay.ps1 start'
}

# ------------------------------------------------------------------------------------------- start / stop / status
# next dev keeps its build in frontend\.next. A cache built from older code can serve old pages after many merges, so a start
# removes it when HEAD is not the commit the last web start recorded (.bonclay\web.head). Same rule as ./bonclay.
function Clear-StaleWebCache {
    if (-not (Has 'git')) { return }
    $head = (& git -C $Root rev-parse HEAD 2>$null | Select-Object -First 1)
    if (-not $head) { return }
    $headFile = Join-Path $RunDir 'web.head'
    $recorded = ''
    if (Test-Path $headFile) { $recorded = (Get-Content -Path $headFile -TotalCount 1).Trim() }
    $cache = Join-Path $Frontend '.next'
    if ((Test-Path $cache) -and ($head -ne $recorded)) {
        try {
            Remove-Item -Path $cache -Recurse -Force -ErrorAction Stop
            Ok 'Cleared the web build cache (code changed since the last start)'
        } catch { Warn 'Could not remove frontend\.next' }
    }
    Set-Content -Path $headFile -Value $head
}

function Invoke-CleanWeb {
    if (Test-Alive 'web') {
        Bad ("The web app is running (pid {0}), so its build cache stays" -f (Get-PidOf 'web'))
        Fix '.\bonclay.ps1 stop, then .\bonclay.ps1 clean-web'
        exit 1
    }
    $cache = Join-Path $Frontend '.next'
    if (Test-Path $cache) {
        Remove-Item -Path $cache -Recurse -Force
        Ok 'Deleted frontend\.next (the web build cache; the next start rebuilds it)'
    } else {
        Write-Host '  No web build cache (frontend\.next) to delete'
    }
}

function Start-BonClay {
    if (-not (Test-Path $VenvPy)) { Bad 'Not set up yet'; Fix '.\bonclay.ps1 setup'; exit 1 }
    Step 'Starting BonClay'

    if (Test-Alive 'api') {
        Ok ("API already running (pid {0})" -f (Get-PidOf 'api'))
    } elseif (Test-PortBusy $ApiPort) {
        Warn "Port $ApiPort is already in use by another program"
        Fix 'stop that program, or set $env:BONCLAY_API_PORT to another port (for example 8001) and start again'
    } else {
        $apiLog = Join-Path $LogDir 'api.log'
        $apiCmd = '"{0}" -m uvicorn backend:app --host 127.0.0.1 --port {1} > "{2}" 2>&1' -f $VenvPy, $ApiPort, $apiLog
        $proc = Start-Hidden -Command $apiCmd -WorkDir $Backend -Vars @{ HF_HUB_DISABLE_XET = '1' }
        Save-Pid -Name 'api' -Proc $proc
        if (Wait-Http -Url "http://127.0.0.1:$ApiPort/api/health" -Seconds 90 -Proc $proc) {
            Ok "API     $ApiUrl"
        } else {
            Bad 'API did not start'
            Fix '.\bonclay.ps1 logs api'
        }
    }

    if (Test-Alive 'web') {
        Ok ("Web app already running (pid {0})" -f (Get-PidOf 'web'))
    } elseif (Test-PortBusy $WebPort) {
        Warn "Port $WebPort is already in use by another program"
        Fix 'stop that program, or set $env:BONCLAY_WEB_PORT to another port and start again'
    } else {
        Clear-StaleWebCache
        $webLog = Join-Path $LogDir 'web.log'
        # --webpack: Turbopack on Windows fails to resolve next/font/google ("queries have exactly one entry")
        $webCmd = 'npx.cmd next dev --webpack -p {0} > "{1}" 2>&1' -f $WebPort, $webLog
        $proc = Start-Hidden -Command $webCmd -WorkDir $Frontend -Vars @{ NEXT_PUBLIC_API_URL = $ApiUrl }
        Save-Pid -Name 'web' -Proc $proc
        if (Wait-Http -Url "http://127.0.0.1:$WebPort" -Seconds 150 -Proc $proc) {
            Ok "Web app $WebUrl"
        } else {
            Bad 'Web app did not start'
            Fix '.\bonclay.ps1 logs web'
        }
    }

    Write-Host ''
    Write-Host "  Open $WebUrl  $Dot  check everything with .\bonclay.ps1 doctor  $Dot  stop with .\bonclay.ps1 stop"
    Write-Host "  Run .\bonclay.ps1 warm before a demo: models load on first use, so the first spoken reply is slow otherwise." -ForegroundColor DarkGray
}

function Invoke-Warm {
    param([switch]$WithClone)
    if (-not (Test-Path $VenvPy)) { Bad 'Not set up yet'; Fix '.\bonclay.ps1 setup'; exit 1 }
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5 | Out-Null }
    catch { Bad "The API is not running on port $ApiPort"; Fix '.\bonclay.ps1 start'; exit 1 }
    Step 'Warming up everything for a live demo'
    $url = "http://127.0.0.1:$ApiPort/api/warmup"
    if ($WithClone) { $url += '?include=clone' }
    $headers = @{ 'storage-mode' = 'local' }
    try { $state = Invoke-RestMethod -Method Post -Uri $url -Headers $headers -TimeoutSec 15 }
    catch { Bad ("The API refused the request: {0}" -f $_.Exception.Message); exit 1 }

    $shown = @{}
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        foreach ($c in $state.components) {
            if ($shown.ContainsKey($c.id) -or $c.state -eq 'pending' -or $c.state -eq 'warming') { continue }
            $shown[$c.id] = $true
            $secs = if ($null -ne $c.seconds) { ('{0:N1}s' -f [double]$c.seconds) } else { '' }
            $line = '{0,-32} {1,7}  {2}' -f $c.label, $secs, $c.detail
            if ($c.state -eq 'ready') { Ok $line }
            elseif ($c.state -eq 'skipped') { Warn $line }
            else { Bad $line }
        }
        if ($state.state -ne 'running') { break }
        Start-Sleep -Seconds 2
        try { $state = Invoke-RestMethod -Uri $url.Split('?')[0] -Headers $headers -TimeoutSec 15 }
        catch { Bad ("Lost the API while warming up: {0}" -f $_.Exception.Message); exit 1 }
    }
    $failed = @($state.components | Where-Object { $_.state -eq 'failed' })
    Write-Host ''
    if ($failed.Count -gt 0) {
        Bad ("Partly ready: {0} component(s) failed" -f $failed.Count)
        Fix '.\bonclay.ps1 logs api'
        exit 1
    }
    Ok ("Ready in {0:N0}s. First replies will be fast." -f $watch.Elapsed.TotalSeconds)
}

function Stop-Named {
    param([string]$Name)
    if (Test-Alive $Name) {
        $targetPid = Get-PidOf $Name
        & taskkill.exe /T /F /PID $targetPid 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Ok "Stopped $Name"
        } else {
            Bad "Could not stop $Name (pid $targetPid)"
            Fix 'end that process in Task Manager, then run .\bonclay.ps1 stop again'
        }
    } else {
        Write-Host "  $Name was not running"
    }
    Remove-Item -Path (Join-Path $RunDir "$Name.pid") -Force -ErrorAction SilentlyContinue
}

function Stop-BonClay {
    Step 'Stopping BonClay'
    Stop-Named -Name 'web'
    Stop-Named -Name 'api'
}

function Show-Status {
    Step 'Status'
    if (Test-Alive 'api') { Ok ("API running (pid {0}) {1}" -f (Get-PidOf 'api'), $ApiUrl) } else { Write-Host '  API stopped' }
    if (Test-Alive 'web') { Ok ("Web app running (pid {0}) {1}" -f (Get-PidOf 'web'), $WebUrl) } else { Write-Host '  Web app stopped' }
}

function Show-Logs {
    param([string]$Which)
    if (-not $Which) { $Which = 'api' }
    if ($Which -notin @('api', 'web')) {
        Write-Host 'usage: .\bonclay.ps1 logs [api|web]'
        exit 1
    }
    $log = Join-Path $LogDir "$Which.log"
    if (-not (Test-Path $log)) {
        Warn "No log yet: $log"
        Fix 'start BonClay first with .\bonclay.ps1 start'
        return
    }
    Get-Content -Path $log -Tail 80 -Wait
}

# ------------------------------------------------------------------------------------------- test
function Invoke-Tests {
    if (-not (Test-Path $VenvPy)) { Bad 'Not set up yet'; Fix '.\bonclay.ps1 setup'; exit 1 }

    Step 'Backend tests'
    Push-Location $Backend
    & $VenvPy -m pytest -q
    $backendCode = $LASTEXITCODE
    Pop-Location
    if ($backendCode -ne 0) { exit 1 }

    Step 'Frontend tests'
    Push-Location $Frontend
    & npx.cmd jest
    $frontendCode = $LASTEXITCODE
    Pop-Location
    if ($frontendCode -ne 0) { exit 1 }
}

# ------------------------------------------------------------------------------------------- doctor
function Invoke-Doctor {
    Step 'Machine'
    Write-Host ("  Windows {0} ({1})" -f [Environment]::OSVersion.Version, $env:PROCESSOR_ARCHITECTURE)

    $basePy = Find-Python
    if ($basePy) { Ok "Python 3.10+ ($basePy)" } else { Bad 'Python 3.10+ missing'; Fix 'winget install Python.Python.3.12' }
    if (Has 'node') { Ok ("Node {0}" -f (node -v)) } else { Bad 'Node.js missing'; Fix 'winget install OpenJS.NodeJS.LTS' }
    if (Has 'ffmpeg') { Ok 'ffmpeg' } else { Bad 'ffmpeg missing'; Fix 'winget install Gyan.FFmpeg' }
    if (Test-Espeak) { Ok 'espeak-ng' } else { Warn 'espeak-ng missing (non-English Kokoro voices)'; Fix 'download the Windows installer from https://github.com/espeak-ng/espeak-ng/releases' }
    if (Test-Path $VenvPy) { Ok 'backend\.venv' } else { Bad 'backend\.venv missing'; Fix '.\bonclay.ps1 setup' }
    if (Test-Path (Join-Path $Frontend 'node_modules')) { Ok 'frontend\node_modules' } else { Bad 'frontend packages missing'; Fix '.\bonclay.ps1 setup' }
    if (Test-NvidiaGpu) { Ok 'NVIDIA GPU (CUDA for voice cloning)' } else { Warn 'No NVIDIA GPU: voice cloning runs on the CPU (slow)' }
    if (Test-Path $EnvFile) { Ok 'backend\.env' } else { Warn 'backend\.env missing'; Fix 'copy backend\.env.example backend\.env' }
    if (Test-Path (Join-Path $DhvaaniDir 'Scripts\python.exe')) { Ok 'backend\.venv-dhvaani (DhVaani: voice cloning in Indian languages)' } else { Warn 'DhVaani is not installed (optional): your voice cannot speak Marathi, Tamil, Bengali, Telugu and the other Indian languages'; Fix '.\bonclay.ps1 setup -Dhvaani   (needs HF_TOKEN in backend\.env, with access granted by ARTPARK-IISc)' }

    Step 'Running app'
    $health = $null
    try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5 } catch { $health = $null }
    if (-not $health) {
        Bad "API is not running on port $ApiPort"
        Fix '.\bonclay.ps1 start'
        return
    }

    $plat  = $health.platform
    $accel = if ($plat -and $plat.accelerator_name) { $plat.accelerator_name } else { 'CPU' }
    Ok ("Platform: {0} {1} - {2}" -f $plat.os, $plat.os_version, $accel)

    $llm = $health.llm
    if ($llm) {
        if ($llm.active) {
            Ok ("Language model: {0} ({1})" -f $llm.active, $llm.model)
        } else {
            Warn 'No language model: agents, the Playground and the Evaluation Lab cannot answer'
            Fix 'add GROQ_API_KEY to backend\.env, or run Ollama and set LLM_BASE_URL'
        }
    } elseif ((Get-DotEnv 'GROQ_API_KEY') -or (Get-DotEnv 'LLM_BASE_URL')) {
        Ok 'Language model configured in backend\.env (the API does not report which one is active)'
    } else {
        Warn 'No language model: agents, the Playground and the Evaluation Lab cannot answer'
        Fix 'add GROQ_API_KEY to backend\.env, or run Ollama and set LLM_BASE_URL'
    }

    $sb = $health.supabase
    if ($sb -and $sb.configured) {
        if ($sb.reachable) {
            Ok 'Supabase connected'
        } else {
            Warn 'Supabase configured but unreachable'
            Fix 'check NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in backend\.env'
        }
    } else {
        Ok 'Local storage (no Supabase needed; data lives in backend\data)'
    }

    $loaded = @($health.models_loaded | Where-Object { $_ })
    if ($loaded.Count -gt 0) {
        Ok ("Models in memory: {0}" -f ($loaded -join ', '))
    } else {
        Ok 'Models in memory: none yet (they load on first use)'
    }
}

# ------------------------------------------------------------------------------------------- help and dispatch
function Show-Help {
    Write-Host @'
BonClay for Windows: one command to set up, start, stop and check the studio.

  .\bonclay.ps1 setup [-Indic] [-Chatterbox] [-Dhvaani]
                                 install everything (Python env, PyTorch, npm packages); -Indic adds Indic-Speak,
                                 -Chatterbox adds Hindi voice cloning (about 3 GB), -Dhvaani adds voice cloning in
                                 Indian languages (DhVaani, ARTPARK-IISc; about 0.5 GB, needs HF_TOKEN)
  .\bonclay.ps1 start            start the API (port 8000) and the web app (port 3000) in the background
  .\bonclay.ps1 stop             stop both
  .\bonclay.ps1 restart          stop, then start
  .\bonclay.ps1 status           what is running
  .\bonclay.ps1 doctor           check the machine and the running app, with a fix for every problem
  .\bonclay.ps1 logs [api|web]   follow the logs (Ctrl+C to stop)
  .\bonclay.ps1 test             run the backend and frontend test suites
  .\bonclay.ps1 warm [--clone]   load every model a live demo needs now (first reply ~3 s instead of ~18 s)
  .\bonclay.ps1 clean-web        delete frontend\.next (the web build cache) when the web app is not running

If PowerShell blocks the script, run it like this:
  powershell -ExecutionPolicy Bypass -File .\bonclay.ps1 setup

Ports: set $env:BONCLAY_API_PORT and $env:BONCLAY_WEB_PORT before start.
macOS and Linux use ./bonclay instead.
'@
}

# "--indic" / "--chatterbox" / "--dhvaani" work like the bash script's flags, in any order and combination.
$extra = @($Target) + @($Rest)
if ($extra -contains '--indic') { $Indic = $true }
if ($extra -contains '--chatterbox') { $Chatterbox = $true }
if ($extra -contains '--dhvaani') { $Dhvaani = $true }
if ($Command.ToLowerInvariant() -eq 'setup') {
    $unknown = @($extra | Where-Object { $_ -and $_ -ne '--indic' -and $_ -ne '--chatterbox' -and $_ -ne '--dhvaani' })
    if ($unknown.Count -gt 0) { Bad ("Unknown option for setup: {0} (use -Indic, -Chatterbox and/or -Dhvaani)" -f ($unknown -join ' ')); exit 1 }
    $Target = ''
} elseif ($Target -eq '--indic' -or $Target -eq '--chatterbox' -or $Target -eq '--dhvaani') {
    $Target = ''
}

switch ($Command.ToLowerInvariant()) {
    'setup'   { Invoke-Setup -WithIndic:$Indic -WithChatterbox:$Chatterbox -WithDhvaani:$Dhvaani }
    'start'   { Start-BonClay }
    'stop'    { Stop-BonClay }
    'restart' { Stop-BonClay; Start-BonClay }
    'status'  { Show-Status }
    'doctor'  { Invoke-Doctor }
    'logs'    { Show-Logs -Which $Target }
    'test'    { Invoke-Tests }
    'warm'    { Invoke-Warm -WithClone:($Target -eq '--clone') }
    'clean-web' { Invoke-CleanWeb }
    'help'    { Show-Help }
    default   { Show-Help; exit 1 }
}
