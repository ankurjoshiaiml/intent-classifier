"""
Prompt Builder V2 — driven by IntentRegistry
─────────────────────────────────────────────
Builds the Opus classification prompt using data directly from the V3.0 sheet:
  • Col B intent names as the valid output labels
  • Col C sub-intents injected per intent
  • Col D personas injected as access gates
  • Col E sample queries as few-shot examples (retrieved dynamically)
  • Col F bot response templates shown as expected output examples
  • Col G entity types to extract
"""

import json
from typing import List, Tuple, Optional

from src.intent_registry import IntentRegistry, IntentRecord
from src.vector_store_v2 import EmbeddedDocument


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_OUTPUT_SCHEMA = {
    "intent_id": "string — exact intent_id from the taxonomy (e.g. VALIDATE_USER_REQUEST)",
    "intent_name": "string — human-readable name (e.g. 'Validate user request')",
    "sub_intent": "string — the most specific Action Intended label from Col C that matches (e.g. 'Access Request (with preventive SoD)')",
    "confidence": "float 0.0–1.0",
    "extracted_entities": {
        "SAP_ROLE": "list of strings or null",
        "TCODE": "list of strings or null",
        "AUTH_OBJECT": "list of strings or null",
        "SOD_RULE_ID": "list of strings or null",
        "USERNAME": "list of strings or null",
        "COUNTRY": "list of strings or null",
        "DEPARTMENT": "list of strings or null",
        "ACTIVITY": "list of strings or null",
    },
    "persona_gate_passed": "boolean",
    "requires_clarification": "boolean",
    "clarification_question": "string or null",
    "reasoning": "string — 1–2 sentences",
    "suggested_bot_response_template": "string or null — closest matching Col F response template",
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    registry: IntentRegistry,
    allowed_records: List[IntentRecord],
) -> str:
    intent_block = _format_intent_taxonomy(allowed_records)
    schema_text = json.dumps(CLASSIFIER_OUTPUT_SCHEMA, indent=2)

    return f"""You are the Intent Classifier for the PAX Identity EKG system — an enterprise knowledge graph that analyses SAP and Microsoft Entra ID data to detect Segregation of Duties (SoD) violations.

Your ONLY job is to:
1. Classify the user's message into one of the defined intents
2. Identify the specific sub-intent (Action Intended) from Column C
3. Extract all relevant entities (SAP roles, transaction codes, usernames, etc.)

Return a single valid JSON object — no prose, no markdown, no explanation outside the JSON.

════════════════════════════════════════════
INTENT TAXONOMY (from V3.0 sheet)
════════════════════════════════════════════
{intent_block}

════════════════════════════════════════════
OUTPUT SCHEMA
════════════════════════════════════════════
{schema_text}

════════════════════════════════════════════
RULES
════════════════════════════════════════════
1. PERSONA GATE: If the user's persona is not in an intent's allowed_personas, set persona_gate_passed=false, intent_id="OUT_OF_SCOPE".
2. CONFIDENCE < 0.70: Set requires_clarification=true with a targeted follow-up question.
3. SUB-INTENT: Always select the most specific sub-intent from the Action Intended list for the matched intent.
4. ENTITIES: Extract only entities explicitly stated or clearly implied. Use null for absent entities.
5. BOT RESPONSE TEMPLATE: Match the closest response template from Col F for the detected sub-intent.
6. STRICT JSON ONLY: First character must be {{ and last must be }}.
"""


# ─────────────────────────────────────────────────────────────────────────────
# User prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    message: str,
    authenticated_user: str,
    persona: str,
    user_level: str,
    department: str,
    country: str,
    retrieved_examples: List[Tuple[EmbeddedDocument, float]],
) -> str:
    examples_block = _format_retrieved_examples(retrieved_examples)

    return f"""════════════════════════════════════════════
USER CONTEXT
════════════════════════════════════════════
Username   : {authenticated_user}
Persona    : {persona}
Level      : {user_level}
Department : {department}
Country    : {country}

════════════════════════════════════════════
SIMILAR EXAMPLES (retrieved from V3.0 sheet)
════════════════════════════════════════════
{examples_block}

════════════════════════════════════════════
MESSAGE TO CLASSIFY
════════════════════════════════════════════
{message}

Return ONLY the JSON classification object."""


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _format_intent_taxonomy(records: List[IntentRecord]) -> str:
    lines = []
    for record in records:
        lines.append(f"\n▸ intent_id    : {record.intent_id}")
        lines.append(f"  intent_name  : {record.intent_name}")
        lines.append(f"  coverage     : {record.coverage_pct}")
        lines.append(f"  allowed      : {record.allowed_personas}")
        lines.append(f"  sub-intents (Action Intended — Col C):")
        for si in record.sub_intents:
            lines.append(f"    {si.index}) {si.label}")
        lines.append(f"  entity_types : {record.entity_types}")

    lines.append(f"\n▸ intent_id    : OUT_OF_SCOPE")
    lines.append(f"  intent_name  : Out of Scope")
    lines.append(f"  description  : Query has nothing to do with SAP access, SoD, or identity governance.")
    return "\n".join(lines)


def _format_retrieved_examples(
    examples: List[Tuple[EmbeddedDocument, float]]
) -> str:
    if not examples:
        return "No similar examples found."

    lines = []
    for i, (doc, score) in enumerate(examples, 1):
        lines.append(f"Example {i} (similarity={score:.2f}):")
        lines.append(f"  Query       : {doc.text}")
        lines.append(f"  Intent      : {doc.intent_id}")
        lines.append(f"  Sub-intent  : {doc.sub_intent_label}")
        lines.append(f"  Personas    : {doc.allowed_personas}")
        if doc.entity_types:
            lines.append(f"  Entity types: {doc.entity_types}")
        if doc.bot_response:
            preview = doc.bot_response[:100].replace("\n", " ")
            lines.append(f"  Bot response: {preview}...")
        lines.append("")

    return "\n".join(lines)


def get_allowed_records_for_persona(
    registry: IntentRegistry, persona: str
) -> List[IntentRecord]:
    """Return only intent records this persona can access."""
    return [r for r in registry.records if persona in r.allowed_personas]
