"""
Intent Registry — V3.0 Sheet Parser
─────────────────────────────────────
Parses intent_definitions_v3.xlsx (V3.0 sheet) as the MASTER intent source.

Sheet structure:
  Col A — Sl No.
  Col B — Common Intent Name          → intent label (our "intent_id")
  Col C — Action Intended             → sub-intent types (list)
  Col D — Persona % Coverage          → allowed personas
  Col E — Sample Query                → numbered examples for embedding
  Col F — Bot Responses               → expected response templates
  Col G — Constraints                 → business rules / constraints
  Col H — Comments                    → notes

Each row = one top-level intent with multiple sub-intents and sample queries.
The sample queries are the corpus for our FAISS vector store.
The action-intended list and persona are the metadata attached to each query.
"""

import re
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubIntent:
    """One entry from the Action Intended column (e.g. 'Access Request')."""
    index: int          # numeric prefix parsed from the cell (1, 2, 3…)
    label: str          # cleaned label text


@dataclass
class SampleQuery:
    """One numbered sample query from the Sample Query column."""
    index: str          # e.g. "1", "1.1", "2", "2.1"
    text: str           # the query text
    sub_intent_ref: Optional[int] = None  # which sub-intent index this maps to


@dataclass
class BotResponse:
    """One numbered bot response from the Bot Responses column."""
    index: str
    text: str


@dataclass
class IntentRecord:
    """
    Full parsed record for one row of the V3.0 sheet.
    This is the master object consumed by the classifier and vector store.
    """
    sl_no: int
    intent_name: str            # Col B — e.g. "Validate user request"
    intent_id: str              # machine-safe ID derived from intent_name
    sub_intents: List[SubIntent]         # Col C — action intended list
    allowed_personas: List[str]          # Col D — parsed persona strings
    coverage_pct: Optional[str]          # Col D — e.g. "~20%"
    sample_queries: List[SampleQuery]    # Col E — numbered queries
    bot_responses: List[BotResponse]     # Col F — numbered responses
    constraints: str                     # Col G
    comments: str                        # Col H

    # Derived: entity types to extract for this intent (from sub-intents + queries)
    entity_types: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent_registry(excel_path: str) -> List[IntentRecord]:
    """
    Parse the V3.0 sheet and return a list of IntentRecord objects.
    One record per row (6 rows = 6 intent groups).
    """
    df = pd.read_excel(excel_path, sheet_name="V3.0", header=0)

    # Rename columns positionally — sheet has 9 cols:
    # Sl No | Intent Name | Action Intended | Persona | % Coverage | Sample Query | Bot Responses | Constraints | Comments
    col_names = ["sl_no", "intent_name", "action_intended",
                 "persona", "coverage", "sample_queries",
                 "bot_responses", "constraints", "comments"]
    df.columns = col_names[:len(df.columns)]

    records: List[IntentRecord] = []

    for _, row in df.iterrows():
        sl_no = _safe_int(row.get("sl_no"))
        if sl_no is None:
            continue

        intent_name = _clean(row.get("intent_name", ""))
        if not intent_name:
            continue

        sub_intents = _parse_sub_intents(_clean(row.get("action_intended", "")))
        # Persona and coverage are now separate columns
        persona_raw = _clean(row.get("persona", ""))
        coverage_raw = _clean(row.get("coverage", ""))
        personas, coverage = _parse_personas(persona_raw + " " + coverage_raw)
        sample_queries = _parse_numbered_items(_clean(row.get("sample_queries", "")))
        bot_responses_raw = _parse_numbered_items(_clean(row.get("bot_responses", "")))

        sq_objects = [SampleQuery(index=idx, text=text) for idx, text in sample_queries]
        br_objects = [BotResponse(index=idx, text=text) for idx, text in bot_responses_raw]

        # Map query index → sub-intent index (e.g. query "1" → sub-intent 1)
        for sq in sq_objects:
            try:
                top_level = int(sq.index.split(".")[0])
                sq.sub_intent_ref = top_level
            except (ValueError, IndexError):
                pass

        record = IntentRecord(
            sl_no=sl_no,
            intent_name=intent_name,
            intent_id=_to_intent_id(intent_name),
            sub_intents=sub_intents,
            allowed_personas=personas,
            coverage_pct=coverage,
            sample_queries=sq_objects,
            bot_responses=br_objects,
            constraints=_clean(row.get("constraints", "")),
            comments=_clean(row.get("comments", "")),
            entity_types=_derive_entity_types(intent_name, sub_intents),
        )
        records.append(record)

    logger.info(f"Parsed {len(records)} intent records from V3.0 sheet")
    for r in records:
        logger.info(f"  [{r.sl_no}] {r.intent_id}: {len(r.sample_queries)} queries, "
                    f"{len(r.sub_intents)} sub-intents, personas={r.allowed_personas}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Column parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sub_intents(raw: str) -> List[SubIntent]:
    """
    Parse the Action Intended column.
    Pattern: "1) Label\n2) Label\n3) Label"
    """
    if not raw:
        return []
    results = []
    # Match numbered items: "1) ...", "2) ..."
    pattern = re.compile(r"(\d+)\)\s*(.+?)(?=\n\d+\)|$)", re.DOTALL)
    for m in pattern.finditer(raw):
        idx = int(m.group(1))
        label = m.group(2).strip().replace("\n", " ")
        results.append(SubIntent(index=idx, label=label))

    # Fallback: split by newline if no numbered pattern
    if not results:
        for line in raw.split("\n"):
            line = line.strip()
            if line:
                results.append(SubIntent(index=len(results) + 1, label=line))

    return results


def _parse_personas(raw: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse the Persona % Coverage column.
    Returns (list_of_personas, coverage_pct_string).
    """
    # Extract coverage percentage
    coverage_match = re.search(r"~?\d+%", raw)
    coverage = coverage_match.group(0) if coverage_match else None

    personas = []
    raw_clean = re.sub(r"~?\d+%", "", raw).strip()

    PERSONA_MAP = {
        "end user": "END_USER",
        "all users": "END_USER",
        "process owner": "PROCESS_OWNER",
        "role owner": "PROCESS_OWNER",
        "data owner": "DATA_OWNER",
        "application owner": "APP_OWNER",
        "app owner": "APP_OWNER",
        "sap security analyst": "APP_OWNER",
        "auditor": "AUDITOR",
        "line manager": "END_USER",
    }

    raw_lower = raw_clean.lower()
    seen = set()
    for key, val in PERSONA_MAP.items():
        if key in raw_lower and val not in seen:
            personas.append(val)
            seen.add(val)

    if not personas:
        personas = ["END_USER"]

    return personas, coverage


def _parse_numbered_items(raw: str) -> List[Tuple[str, str]]:
    """
    Parse numbered multi-line cells.
    Handles: "1) text\n1.1) text\n2) text"
    Returns list of (index_string, text_string) tuples.
    """
    if not raw:
        return []

    results = []
    # Match: "1)", "1.1)", "2)", "2.1)", etc.
    pattern = re.compile(r"(\d+(?:\.\d+)*)\)\s*(.+?)(?=\n\s*\d+(?:\.\d+)*\)|$)", re.DOTALL)

    for m in pattern.finditer(raw):
        idx = m.group(1).strip()
        text = m.group(2).strip().replace("\n", " ")
        # Remove excess whitespace
        text = re.sub(r"\s{2,}", " ", text)
        if text:
            results.append((idx, text))

    # Fallback: plain lines
    if not results:
        for i, line in enumerate(raw.split("\n"), 1):
            line = line.strip()
            if len(line) > 5:
                results.append((str(i), line))

    return results


def _derive_entity_types(intent_name: str, sub_intents: List[SubIntent]) -> List[str]:
    """
    Derive the entity types that should be extracted for this intent.
    Based on intent semantics from the V3.0 sheet.
    """
    name_lower = intent_name.lower()
    sub_labels = " ".join(s.label.lower() for s in sub_intents)
    combined = f"{name_lower} {sub_labels}"

    entities = []

    if any(k in combined for k in ["role", "access request", "detect role"]):
        entities.append("SAP_ROLE")
        entities.append("TCODE")

    if "sod" in combined or "segregation" in combined or "conflict" in combined:
        entities.append("SOD_RULE_ID")
        entities.append("SAP_ROLE")
        entities.append("TCODE")

    if "user" in combined or "validate" in combined or "leaver" in combined:
        entities.append("USERNAME")
        entities.append("COUNTRY")
        entities.append("DEPARTMENT")

    if "what-if" in combined or "scenario" in combined or "design" in combined:
        entities.append("SAP_ROLE")
        entities.append("TCODE")
        entities.append("AUTH_OBJECT")

    if "knowledge graph" in combined or "graph" in combined:
        entities.append("USERNAME")
        entities.append("DEPARTMENT")

    if "elaborate" in combined or "function" in combined:
        entities.append("SAP_ROLE")

    # Always include these fundamentals
    for e in ["USERNAME", "COUNTRY", "DEPARTMENT"]:
        if e not in entities:
            entities.append(e)

    # Deduplicate preserving order
    seen = set()
    return [e for e in entities if not (e in seen or seen.add(e))]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_intent_id(name: str) -> str:
    """Convert intent name to a machine-safe snake_case ID."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Registry accessor
# ─────────────────────────────────────────────────────────────────────────────

class IntentRegistry:
    """
    Singleton-style accessor for the parsed intent definitions.
    Loaded once at startup and reused throughout the pipeline.
    """

    def __init__(self, excel_path: str):
        self.records: List[IntentRecord] = parse_intent_registry(excel_path)
        self._by_id: Dict[str, IntentRecord] = {r.intent_id: r for r in self.records}
        self._by_name: Dict[str, IntentRecord] = {r.intent_name: r for r in self.records}

    def get_by_id(self, intent_id: str) -> Optional[IntentRecord]:
        return self._by_id.get(intent_id)

    def get_by_name(self, name: str) -> Optional[IntentRecord]:
        return self._by_name.get(name)

    def all_intent_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def get_sub_intents(self, intent_id: str) -> List[SubIntent]:
        r = self._by_id.get(intent_id)
        return r.sub_intents if r else []

    def get_entity_types(self, intent_id: str) -> List[str]:
        r = self._by_id.get(intent_id)
        return r.entity_types if r else []

    def get_allowed_personas(self, intent_id: str) -> List[str]:
        r = self._by_id.get(intent_id)
        return r.allowed_personas if r else []

    def get_matching_bot_response(
        self, intent_id: str, query_index: str
    ) -> Optional[str]:
        """Return the bot response template for a specific query index."""
        r = self._by_id.get(intent_id)
        if not r:
            return None
        for br in r.bot_responses:
            if br.index == query_index:
                return br.text
        return None

    def summary(self) -> str:
        lines = ["Intent Registry (V3.0):"]
        for r in self.records:
            lines.append(
                f"  [{r.sl_no}] {r.intent_id}\n"
                f"      Sub-intents : {[s.label for s in r.sub_intents]}\n"
                f"      Personas    : {r.allowed_personas}\n"
                f"      Coverage    : {r.coverage_pct}\n"
                f"      Queries     : {len(r.sample_queries)}\n"
                f"      Entities    : {r.entity_types}"
            )
        return "\n".join(lines)
