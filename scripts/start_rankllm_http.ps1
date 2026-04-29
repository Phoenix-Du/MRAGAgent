$ErrorActionPreference = "Stop"

$rankDir = "d:\multimodal-rag-agent\third_party\rank_llm"
if (!(Test-Path $rankDir)) {
    throw "rank_llm not found at $rankDir"
}

Set-Location $rankDir

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip --proxy=""
& ".\.venv\Scripts\python.exe" -m pip install rank-llm fastapi uvicorn --proxy=""

# Use local HTTP bridge compatible with /v1/rerank contract expected by orchestrator.
& ".\.venv\Scripts\python.exe" -m uvicorn app.integrations.rankllm_bridge:app --host 0.0.0.0 --port 8082 --app-dir "d:\multimodal-rag-agent"
