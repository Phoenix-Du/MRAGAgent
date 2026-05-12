from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Multimodal RAG Agent"
    app_version: str = "0.2.0"
    log_level: str = "INFO"
    cors_allow_origins: str = "*"

    # External connectors (replace with real endpoints)
    crawl4ai_endpoint: str | None = None
    crawl4ai_local_enabled: bool = True
    crawl4ai_local_timeout_seconds: int = 12
    serpapi_endpoint: str | None = None
    serpapi_api_keys: str | None = None
    serpapi_api_key: str | None = None
    bge_reranker_model: str = "BAAI/bge-reranker-base"
    bge_reranker_local_files_only: bool = False
    rasa_endpoint: str | None = None
    rag_anything_endpoint: str | None = None
    image_pipeline_endpoint: str | None = "http://127.0.0.1:9010/search-rank"
    web_search_candidates_n: int = 12
    web_url_select_m: int = 5
    web_crawl_concurrency: int = 3
    image_proxy_max_redirects: int = 5

    request_timeout_seconds: int = 30
    image_pipeline_timeout_seconds: int = 120
    rag_ingest_timeout_seconds: int = 120
    rag_query_timeout_seconds: int = 120
    parser_connect_timeout_seconds: int = 10
    parser_read_timeout_seconds: int = 120
    memory_backend: str = "memory"
    memory_max_turns: int = 10
    image_search_ingest_enabled: bool = False
    general_qa_body_rerank_enabled: bool = True
    allow_placeholder_fallback: bool = True
    local_rag_store_max_docs: int = 1000

    redis_url: str | None = None
    redis_prefix: str = "mmrag"

    mysql_host: str | None = None
    mysql_port: int = 3306
    mysql_user: str | None = None
    mysql_password: str | None = None
    mysql_database: str | None = None


settings = AppSettings()
