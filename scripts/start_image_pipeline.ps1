$ErrorActionPreference = "Stop"

$root = "d:\multimodal-rag-agent"
Set-Location $root

# IMAGE_SEARCH_PROVIDER: wikimedia | serpapi
# Optional:
#   SERPAPI_API_KEY=xxx
#   QWEN_VLM_RERANK_ENDPOINT=http://127.0.0.1:9011/rerank

python -m uvicorn app.integrations.image_pipeline_bridge:app --host 0.0.0.0 --port 9010
