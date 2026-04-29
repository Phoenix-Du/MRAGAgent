$ErrorActionPreference = "Stop"

$ragDir = "d:\multimodal-rag-agent\third_party\RAG-ANYTHING"
if (!(Test-Path $ragDir)) {
    throw "RAG-ANYTHING not found at $ragDir"
}

Set-Location $ragDir

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip --proxy=""
& ".\.venv\Scripts\python.exe" -m pip install raganything --proxy=""
# Optional: HTML / Office routing in bridge uses Docling CLI (large dependency).
& ".\.venv\Scripts\python.exe" -m pip install docling --proxy=""
& ".\.venv\Scripts\python.exe" -m pip install fastapi uvicorn python-dotenv --proxy=""

# Run bridge API from workspace module path.
& ".\.venv\Scripts\python.exe" -m uvicorn app.integrations.raganything_bridge:app --host 0.0.0.0 --port 9002 --app-dir "d:\multimodal-rag-agent"
