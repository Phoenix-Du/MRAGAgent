from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BridgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Shared model / API config
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    qwen_api_key: str | None = None
    qwen_base_url: str | None = None
    qwen_model: str = "qwen-plus"
    qwen_parser_model: str = "qwen3.5-plus"
    qwen_vlm_model: str = "qwen3.5-omni-plus-2026-03-15"
    request_timeout_seconds: int = 30
    parser_read_timeout_seconds: int = 120

    # RAGAnything bridge
    raganything_crawl_hybrid: bool = True
    raganything_max_html_chars: int = 2_000_000
    raganything_llm_model: str = "gpt-4o-mini"
    raganything_vision_model: str = "qwen3.5-omni-plus-2026-03-15"
    raganything_embedding_model: str = "text-embedding-3-large"
    raganything_embedding_dim: int = 3072
    raganything_working_dir: str = "./raganything_storage"
    raganything_parser: str = "mineru"
    raganything_query_mode: str = "hybrid"
    raganything_add_scripts_dir_to_path: bool = True

    # Image pipeline bridge
    serpapi_api_keys: str | None = None
    serpapi_api_key: str | None = None
    chinese_clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    chinese_clip_local_files_only: bool = True
    image_cache_dir: str = "./raganything_storage/image_pipeline_cache"
    image_search_provider: str = "serpapi"
    image_vlm_rank_enabled: bool = True
    image_top_k_default: int = 5
    image_cache_ttl_hours: int = 72
    image_cache_cleanup_interval_seconds: int = 1800
    image_retrieval_k: int | None = None
    image_source_min_accessible: int | None = None
    image_source_max_check: int | None = None
    image_access_check_concurrency: int = 8
    image_access_check_timeout: int = 10
    image_clip_eval: int | None = None
    image_clip_download_concurrency: int = 6
    image_clip_keep: int | None = None
    image_clip_min_score: float = 0.18
    image_final_max_check: int | None = None
    image_vlm_rank_pool: int | None = None


bridge_settings = BridgeSettings()
