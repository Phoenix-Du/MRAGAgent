from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsStore:
    requests_total: int = 0
    requests_success_total: int = 0
    requests_failed_total: int = 0
    intent_fallback_total: int = 0
    rag_ingest_fallback_total: int = 0
    rag_query_fallback_total: int = 0
    search_fallback_total: int = 0
    crawl_fallback_total: int = 0
    bge_rerank_fallback_total: int = 0
    image_pipeline_fallback_total: int = 0
    clarification_needed_total: int = 0
    total_latency_ms_sum: int = 0
    rag_latency_ms_sum: int = 0
    _lock: Lock = field(default_factory=Lock)

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            setattr(self, name, int(getattr(self, name, 0)) + value)

    def add_latency(self, total_latency_ms: int, rag_latency_ms: int) -> None:
        with self._lock:
            self.total_latency_ms_sum += int(total_latency_ms)
            self.rag_latency_ms_sum += int(rag_latency_ms)

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP mmrag_requests_total Total chat requests.",
                "# TYPE mmrag_requests_total counter",
                f"mmrag_requests_total {self.requests_total}",
                "# HELP mmrag_requests_success_total Successful chat requests.",
                "# TYPE mmrag_requests_success_total counter",
                f"mmrag_requests_success_total {self.requests_success_total}",
                "# HELP mmrag_requests_failed_total Failed chat requests.",
                "# TYPE mmrag_requests_failed_total counter",
                f"mmrag_requests_failed_total {self.requests_failed_total}",
                "# HELP mmrag_intent_fallback_total Intent fallback count.",
                "# TYPE mmrag_intent_fallback_total counter",
                f"mmrag_intent_fallback_total {self.intent_fallback_total}",
                "# HELP mmrag_rag_ingest_fallback_total RAG ingest fallback count.",
                "# TYPE mmrag_rag_ingest_fallback_total counter",
                f"mmrag_rag_ingest_fallback_total {self.rag_ingest_fallback_total}",
                "# HELP mmrag_rag_query_fallback_total RAG query fallback count.",
                "# TYPE mmrag_rag_query_fallback_total counter",
                f"mmrag_rag_query_fallback_total {self.rag_query_fallback_total}",
                "# HELP mmrag_crawl_fallback_total Crawl fallback count.",
                "# TYPE mmrag_crawl_fallback_total counter",
                f"mmrag_crawl_fallback_total {self.crawl_fallback_total}",
                "# HELP mmrag_search_fallback_total Search fallback count.",
                "# TYPE mmrag_search_fallback_total counter",
                f"mmrag_search_fallback_total {self.search_fallback_total}",
                "# HELP mmrag_bge_rerank_fallback_total BGE rerank fallback count.",
                "# TYPE mmrag_bge_rerank_fallback_total counter",
                f"mmrag_bge_rerank_fallback_total {self.bge_rerank_fallback_total}",
                "# HELP mmrag_image_pipeline_fallback_total Image pipeline fallback count.",
                "# TYPE mmrag_image_pipeline_fallback_total counter",
                f"mmrag_image_pipeline_fallback_total {self.image_pipeline_fallback_total}",
                "# HELP mmrag_clarification_needed_total Clarification prompts returned to users.",
                "# TYPE mmrag_clarification_needed_total counter",
                f"mmrag_clarification_needed_total {self.clarification_needed_total}",
                "# HELP mmrag_total_latency_ms_sum Sum of end-to-end latency in ms.",
                "# TYPE mmrag_total_latency_ms_sum counter",
                f"mmrag_total_latency_ms_sum {self.total_latency_ms_sum}",
                "# HELP mmrag_rag_latency_ms_sum Sum of adapter RAG latency in ms.",
                "# TYPE mmrag_rag_latency_ms_sum counter",
                f"mmrag_rag_latency_ms_sum {self.rag_latency_ms_sum}",
            ]
        return "\n".join(lines) + "\n"


metrics = MetricsStore()
