"""
Vector Store V2 — driven by IntentRegistry (V3.0 sheet)
─────────────────────────────────────────────────────────
Each sample query from the sheet becomes one embedded document.
Metadata stored per document:
  - intent_id     (Col B derived)
  - intent_name   (Col B raw)
  - sub_intent    (Col C — which numbered sub-intent this query maps to)
  - allowed_personas  (Col D)
  - query_index   (the "1)", "1.1)" prefix)
  - entity_types  (derived from intent semantics)
  - bot_response  (Col F — the expected response for this query index)
  - source        "v3_sheet"

At search time, the top-K results return all this metadata so the classifier
prompt is enriched with: similar query text + its known intent + sub-intent +
the expected response template — giving Opus maximum context.
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from src.intent_registry import IntentRegistry, IntentRecord, SampleQuery

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedded document
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddedDocument:
    """One embedded unit — a single sample query with full metadata."""

    def __init__(
        self,
        text: str,
        intent_id: str,
        intent_name: str,
        sub_intent_label: str,
        allowed_personas: List[str],
        query_index: str,
        entity_types: List[str],
        bot_response: str = "",
        source: str = "v3_sheet",
    ):
        self.text = text
        self.intent_id = intent_id
        self.intent_name = intent_name
        self.sub_intent_label = sub_intent_label
        self.allowed_personas = allowed_personas
        self.query_index = query_index
        self.entity_types = entity_types
        self.bot_response = bot_response
        self.source = source

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "sub_intent_label": self.sub_intent_label,
            "allowed_personas": self.allowed_personas,
            "query_index": self.query_index,
            "entity_types": self.entity_types,
            "bot_response": self.bot_response[:120] + "..." if len(self.bot_response) > 120 else self.bot_response,
            "source": self.source,
        }

    def __repr__(self):
        return f"EmbeddedDoc(intent={self.intent_id}, idx={self.query_index}, text={self.text[:55]!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Build documents from registry
# ─────────────────────────────────────────────────────────────────────────────

def build_documents_from_registry(registry: IntentRegistry) -> List[EmbeddedDocument]:
    """
    Convert every sample query in the registry into an EmbeddedDocument.
    Also adds paraphrased synthetic variants to improve recall.
    """
    documents: List[EmbeddedDocument] = []

    for record in registry.records:
        for sq in record.sample_queries:
            if not sq.text or len(sq.text) < 5:
                continue

            # Find the sub-intent label for this query's index
            sub_label = _find_sub_intent_label(record, sq.sub_intent_ref)

            # Find the bot response for this query index
            bot_resp = _find_bot_response(record, sq.index)

            doc = EmbeddedDocument(
                text=sq.text,
                intent_id=record.intent_id,
                intent_name=record.intent_name,
                sub_intent_label=sub_label,
                allowed_personas=record.allowed_personas,
                query_index=sq.index,
                entity_types=record.entity_types,
                bot_response=bot_resp,
                source="v3_sheet",
            )
            documents.append(doc)

    # Add synthetic paraphrases for better coverage
    documents.extend(_build_synthetic_documents(registry))

    logger.info(f"Built {len(documents)} embedded documents from registry")
    return documents


def _find_sub_intent_label(record: IntentRecord, sub_intent_ref: Optional[int]) -> str:
    if sub_intent_ref is None:
        return record.intent_name
    for si in record.sub_intents:
        if si.index == sub_intent_ref:
            return si.label
    return record.intent_name


def _find_bot_response(record: IntentRecord, query_index: str) -> str:
    for br in record.bot_responses:
        if br.index == query_index:
            return br.text
    # Try top-level match (e.g. "1.1" → look for "1")
    top = query_index.split(".")[0]
    for br in record.bot_responses:
        if br.index == top:
            return br.text
    return ""


def _build_synthetic_documents(registry: IntentRegistry) -> List[EmbeddedDocument]:
    """
    Synthetic paraphrases covering natural Teams phrasings not in the sheet.
    Mapped to intent_ids from the V3.0 sheet.
    """
    # intent_id → from _to_intent_id() applied to Col B names
    synth = [
        # VALIDATE_USER_REQUEST
        ("Am I set up in SAP?",                     "VALIDATE_USER_REQUEST",         "Access Request",                       ["END_USER"]),
        ("I just joined the finance team",           "VALIDATE_USER_REQUEST",         "Access Request",                       ["END_USER"]),
        ("I need access to SAP for my new role",     "VALIDATE_USER_REQUEST",         "Access Request (for user creation)",    ["END_USER"]),
        ("Please revoke all my SAP access",          "VALIDATE_USER_REQUEST",         "Leavers Access Revoke",                 ["END_USER"]),
        ("I am leaving the company",                 "VALIDATE_USER_REQUEST",         "Leavers Access Revoke",                 ["END_USER"]),
        ("Can I see a summary of my current access?","VALIDATE_USER_REQUEST",         "Access Review/Modification (Self)",     ["END_USER"]),

        # DETECT_ROLES_REQUESTED
        ("What roles do people in my team have?",    "DETECT_ROLES_REQUESTED",        "Access Request",                       ["END_USER"]),
        ("Can I get the PO creator role?",           "DETECT_ROLES_REQUESTED",        "Access Request (with preventive SoD)", ["END_USER"]),
        ("I need to be able to create purchase orders","DETECT_ROLES_REQUESTED",      "Access Request",                       ["END_USER"]),
        ("Who approves the PO-Releaser role?",       "DETECT_ROLES_REQUESTED",        "Access Request (with preventive SoD)", ["END_USER"]),

        # CHECK_FOR_SOD
        ("Do I have any SoD violations?",            "CHECK_FOR_SOD",                 "Access Request (with preventive SoD)", ["END_USER"]),
        ("Does adding this role create a conflict?", "CHECK_FOR_SOD",                 "Access Request (with preventive SoD)", ["END_USER"]),
        ("Why is my access flagged as a risk?",      "CHECK_FOR_SOD",                 "SoD (Remediation - Auditor)",          ["END_USER", "AUDITOR"]),
        ("Can you help me raise these remediations?","CHECK_FOR_SOD",                 "Access Review/Modification (Process/Role Owner)", ["PROCESS_OWNER"]),

        # ELABORATE_ROLE_FUNCTIONS
        ("What does the PO-Creator role allow me to do?", "ELABORATE_ROLE_FUNCTIONS", "Access Request",                      ["END_USER"]),
        ("Explain the transactions in this role",    "ELABORATE_ROLE_FUNCTIONS",      "Access Request",                       ["END_USER"]),
        ("I need a new GR processor role for UK",    "ELABORATE_ROLE_FUNCTIONS",      "Role Design Assistance (New)",         ["PROCESS_OWNER", "APP_OWNER"]),
        ("Design a role that only posts goods receipts","ELABORATE_ROLE_FUNCTIONS",   "Role Design Assistance (New)",         ["PROCESS_OWNER", "APP_OWNER"]),

        # KNOWLEDGE_GRAPH_BASED_SCENARIOS
        ("Show me a graphical view of my team's access","KNOWLEDGE_GRAPH_BASED_SCENARIOS","Access Review/Modification (Line Manager)",["END_USER","PROCESS_OWNER"]),
        ("I want to do my annual access review",     "KNOWLEDGE_GRAPH_BASED_SCENARIOS","Access Review/Modification (Process/Role Owner)",["PROCESS_OWNER"]),
        ("Show all access risks in SAP",             "KNOWLEDGE_GRAPH_BASED_SCENARIOS","SoD (Remediation - Auditor)",         ["AUDITOR"]),

        # RUN_A_WHATIF_SCENARIO
        ("If I remove ME28 from this role will the conflict go away?","RUN_A_WHATIF_SCENARIO","Role Design Assistance (Existing)",["PROCESS_OWNER","APP_OWNER"]),
        ("What happens to SoD if I add this auth object?","RUN_A_WHATIF_SCENARIO",    "Role Design Assistance (New)",         ["PROCESS_OWNER","APP_OWNER"]),
        ("Simulate removing the release auth objects","RUN_A_WHATIF_SCENARIO",        "Role Design Assistance (Existing)",    ["PROCESS_OWNER","APP_OWNER"]),
    ]

    docs = []
    for text, intent_id, sub_label, personas in synth:
        # Get entity types from registry if record exists
        rec = registry.get_by_id(intent_id)
        entity_types = rec.entity_types if rec else []

        docs.append(EmbeddedDocument(
            text=text,
            intent_id=intent_id,
            intent_name=intent_id.replace("_", " ").title(),
            sub_intent_label=sub_label,
            allowed_personas=personas,
            query_index="synth",
            entity_types=entity_types,
            bot_response="",
            source="synthetic",
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────

class IntentVectorStore:
    """
    FAISS vector store for intent example retrieval.
    Embeds using sentence-transformers (local, no API cost).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._documents: List[EmbeddedDocument] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, documents: List[EmbeddedDocument]) -> None:
        import faiss
        self._documents = documents
        model = self._get_model()
        texts = [doc.text for doc in documents]
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        k: int = 5,
        persona_filter: Optional[str] = None,
    ) -> List[Tuple[EmbeddedDocument, float]]:
        """
        Return top-k similar documents.
        Optionally filter by persona so examples are always relevant to the user.
        """
        model = self._get_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Over-fetch if filtering, then trim
        fetch_k = k * 4 if persona_filter else k
        scores, indices = self._index.search(query_emb, min(fetch_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            if persona_filter and persona_filter not in doc.allowed_personas:
                # Still include if persona list is wide (END_USER covers all)
                if "END_USER" not in doc.allowed_personas:
                    continue
            results.append((doc, float(score)))
            if len(results) >= k:
                break

        return results

    def save(self, path: str) -> None:
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self._documents, f)
        logger.info(f"Vector store saved → {path} ({self._index.ntotal} vectors)")

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "IntentVectorStore":
        import faiss
        store = cls(model_name=model_name)
        store._index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            store._documents = pickle.load(f)
        logger.info(f"Vector store loaded ← {path} ({store._index.ntotal} vectors)")
        return store

    @property
    def total(self) -> int:
        return self._index.ntotal if self._index else 0


def build_vector_store(
    registry: IntentRegistry,
    save_path: str = "./data/vector_store",
    model_name: str = "all-MiniLM-L6-v2",
) -> IntentVectorStore:
    """Full pipeline: registry → documents → embed → FAISS index → save."""
    documents = build_documents_from_registry(registry)
    store = IntentVectorStore(model_name=model_name)
    store.build(documents)
    store.save(save_path)
    return store
