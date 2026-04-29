$ErrorActionPreference = "Stop"

$rasaDir = "d:\multimodal-rag-agent\third_party\rasa_main"
if (!(Test-Path $rasaDir)) {
    throw "rasa_main not found at $rasaDir"
}
$projectDir = "d:\multimodal-rag-agent\rasa_project"
if (!(Test-Path $projectDir)) {
    throw "rasa_project not found at $projectDir"
}

Set-Location $rasaDir

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

$py = Join-Path $rasaDir ".venv\Scripts\python.exe"

# Ensure rasa CLI is available in this venv.
& $py -m pip install --upgrade pip --proxy=""
& $py -m pip install rasa==3.6.21 --proxy=""
& $py -m pip install jieba --proxy=""

$modelsDir = Join-Path $projectDir "models"
if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

# Train NLU model if no model exists yet.
$existingModels = Get-ChildItem -Path $modelsDir -Filter "*.tar.gz" -ErrorAction SilentlyContinue
if (-not $existingModels) {
    & $py -m rasa train nlu `
        --config (Join-Path $projectDir "config.yml") `
        --nlu (Join-Path $projectDir "data\nlu.yml") `
        --domain (Join-Path $projectDir "domain.yml") `
        --out $modelsDir
}

$trainedModels = Get-ChildItem -Path $modelsDir -Filter "*.tar.gz" -ErrorAction SilentlyContinue
if (-not $trainedModels) {
    throw "Rasa NLU training did not produce a model in $modelsDir"
}

# Start real Rasa parse endpoint (/model/parse) on port 5005.
& $py -m rasa run `
    --enable-api `
    --cors "*" `
    --port 5005 `
    --model $modelsDir
