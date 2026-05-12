from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from app.core.url_safety import is_safe_public_http_url
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    model_config = ConfigDict(extra="forbid")

    uid: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, min_length=6, max_length=128)
    intent: IntentType | None = None
    query: str = Field(min_length=1, max_length=4000)
    url: str | None = Field(default=None, max_length=2048)
    source_docs: list[SourceDoc] = Field(default_factory=list)
    images: list[ModalElement] = Field(default_factory=list)
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


class PendingClarification(BaseModel):
    type: Literal["clarification"] = "clarification"
    route: IntentType
    original_query: str
    question: str
    missing: list[str] = Field(default_factory=list)
    created_at: int


class UserPreferences(BaseModel):
    profile: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    pending_clarification: PendingClarification | dict[str, Any] | None = None


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


class ConstraintRelation(BaseModel):
    type: Literal["spatial", "action", "object"] = "object"
    relation: str
    subject: str
    object: str


class ImageSearchConstraints(BaseModel):
    raw_query: str
    search_rewrite: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    visual_attributes: list[str] = Field(default_factory=list)
    relations: list[ConstraintRelation] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    count: int | None = None
    must_have_all_subjects: bool = True
    needs_clarification: bool = False
    clarification_question: str | None = None
    parser_source: str = "heuristic"

    @property
    def subjects(self) -> list[str]:
        value = self.entities.get("subjects") or []
        return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []

    @property
    def subject_synonyms(self) -> dict[str, list[str]]:
        value = self.entities.get("subject_synonyms") or {}
        if not isinstance(value, dict):
            return {}
        return {
            str(k): [str(v).strip() for v in vals if str(v).strip()]
            for k, vals in value.items()
            if isinstance(vals, list)
        }

    @property
    def landmark(self) -> str | None:
        value = str(self.entities.get("landmark") or self.entities.get("location") or "").strip()
        return value or None

    @property
    def time_of_day(self) -> str | None:
        value = str(self.entities.get("time_of_day") or "").strip()
        return value or None

    @property
    def attributes(self) -> list[str]:
        return self.visual_attributes

    @property
    def style_terms(self) -> list[str]:
        value = self.entities.get("style_terms") or []
        return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []

    @property
    def exclude_terms(self) -> list[str]:
        return self.negative_constraints

    @property
    def spatial_relations(self) -> list[SpatialRelationConstraint]:
        out: list[SpatialRelationConstraint] = []
        for rel in self.relations:
            if rel.type != "spatial":
                continue
            try:
                out.append(
                    SpatialRelationConstraint(
                        relation=rel.relation,
                        primary_subject=rel.subject,
                        secondary_subject=rel.object,
                    )
                )
            except Exception:
                continue
        return out

    @property
    def action_relations(self) -> list[ActionRelationConstraint]:
        return [
            ActionRelationConstraint(subject=rel.subject, verb=rel.relation, object=rel.object)
            for rel in self.relations
            if rel.type == "action"
        ]

    @property
    def object_relations(self) -> list[ObjectRelationConstraint]:
        return [
            ObjectRelationConstraint(subject=rel.subject, relation=rel.relation, object=rel.object)
            for rel in self.relations
            if rel.type == "object"
        ]


class GeneralQueryConstraints(BaseModel):
    raw_query: str
    search_rewrite: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    attributes: list[str] = Field(default_factory=list)
    compare_targets: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    parser_source: str = "heuristic"

    @property
    def city(self) -> str | None:
        value = str(self.entities.get("location") or self.entities.get("city") or "").strip()
        return value or None


class QueryExecutionContext(BaseModel):
    uid: str
    request_id: str | None = None
    intent: IntentType
    query: str
    original_query: str
    url: str | None = None
    source_docs: list[SourceDoc] = Field(default_factory=list)
    images: list[ModalElement] = Field(default_factory=list)
    image_search_query: str | None = None
    image_constraints: ImageSearchConstraints | None = None
    general_constraints: GeneralQueryConstraints | None = None
    max_images: int = 5
    max_web_docs: int = 5
    max_web_candidates: int | None = None

    @classmethod
    def from_request(
        cls,
        req: QueryRequest,
        *,
        intent: IntentType,
        query: str | None = None,
        original_query: str | None = None,
    ) -> "QueryExecutionContext":
        current_query = query or req.query
        return cls(
            uid=req.uid,
            request_id=req.request_id,
            intent=intent,
            query=current_query,
            original_query=original_query or current_query,
            url=req.url,
            source_docs=req.source_docs,
            images=req.images,
            max_images=req.max_images,
            max_web_docs=req.max_web_docs,
            max_web_candidates=req.max_web_candidates,
        )


class NormalizedDocument(BaseModel):
    doc_id: str
    text: str
    modal_elements: list[ModalElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedPayload(BaseModel):
    uid: str
    request_id: str | None = None
    intent: IntentType
    query: str
    image_search_query: str | None = None
    original_query: str | None = None
    max_images: int = 5
    image_constraints: ImageSearchConstraints | None = None
    general_constraints: GeneralQueryConstraints | None = None
    documents: list[NormalizedDocument] = Field(default_factory=list)
