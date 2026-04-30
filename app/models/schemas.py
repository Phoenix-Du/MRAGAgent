from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from app.core.url_safety import is_safe_public_http_url
from pydantic import BaseModel, Field, field_validator


IntentType = Literal["image_search", "general_qa"]
ModalType = Literal["image", "table", "equation", "generic"]


class ModalElement(BaseModel):
    type: ModalType
    url: str | None = None
    desc: str | None = None
    local_path: str | None = None


class SourceDoc(BaseModel):
    doc_id: str
    text_content: str = ""
    modal_elements: list[ModalElement] = Field(default_factory=list)
    structure: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    uid: str = Field(min_length=1, max_length=128)
    intent: IntentType | None = None
    query: str = Field(min_length=1, max_length=4000)
    image_search_query: str | None = None
    original_query: str | None = None
    url: str | None = Field(default=None, max_length=2048)
    source_docs: list[SourceDoc] = Field(default_factory=list)
    images: list[ModalElement] = Field(default_factory=list)
    image_constraints: "ImageSearchConstraints | None" = None
    general_constraints: "GeneralQueryConstraints | None" = None
    use_rasa_intent: bool = True
    intent_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    device_info: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    max_images: int = Field(default=5, ge=1, le=12)
    max_web_docs: int = Field(default=5, ge=1, le=10)
    max_web_candidates: int | None = Field(default=None, ge=1, le=50)

    @field_validator("url")
    @classmethod
    def validate_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not is_safe_public_http_url(value):
            raise ValueError("url must be a public http or https URL")
        return value


class EvidenceItem(BaseModel):
    doc_id: str
    score: float
    snippet: str


class ImageItem(BaseModel):
    url: str
    desc: str | None = None
    local_path: str | None = None


class QueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    images: list[ImageItem] = Field(default_factory=list)
    trace_id: str
    latency_ms: int
    route: IntentType | None = None
    runtime_flags: list[str] = Field(default_factory=list)


class SpatialRelationConstraint(BaseModel):
    relation: Literal[
        "left_right",
        "right_left",
        "front_back",
        "back_front",
        "above_below",
        "below_above",
        "left_of",
        "right_of",
        "in_front_of",
        "behind",
        "on",
        "under",
        "inside",
        "next_to",
    ]
    primary_subject: str
    secondary_subject: str


class ObjectRelationConstraint(BaseModel):
    subject: str
    relation: str
    object: str


class ActionRelationConstraint(BaseModel):
    subject: str
    verb: str
    object: str


class ImageSearchConstraints(BaseModel):
    raw_query: str
    search_rewrite: str | None = None
    subjects: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    subject_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    style_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    count: int | None = None
    landmark: str | None = None
    time_of_day: str | None = None
    must_have_all_subjects: bool = True
    spatial_relations: list[SpatialRelationConstraint] = Field(default_factory=list)
    action_relations: list[ActionRelationConstraint] = Field(default_factory=list)
    object_relations: list[ObjectRelationConstraint] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    parser_source: str = "heuristic"


class GeneralQueryConstraints(BaseModel):
    raw_query: str
    search_rewrite: str | None = None
    city: str | None = None
    attributes: list[str] = Field(default_factory=list)
    compare_targets: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    parser_source: str = "heuristic"


class NormalizedDocument(BaseModel):
    doc_id: str
    text: str
    modal_elements: list[ModalElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedPayload(BaseModel):
    uid: str
    intent: IntentType
    query: str
    image_search_query: str | None = None
    original_query: str | None = None
    max_images: int = 5
    image_constraints: ImageSearchConstraints | None = None
    general_constraints: GeneralQueryConstraints | None = None
    documents: list[NormalizedDocument] = Field(default_factory=list)
