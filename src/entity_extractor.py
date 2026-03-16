"""
Entity Extractor
─────────────────
Extracts structured entities from user queries based on the entity_types
defined in the IntentRegistry (Col C + Col G of the V3.0 sheet).

Entity types extracted:
  SAP_ROLE      — ZC:P2P:PO_CREATOR________:1000 style technical names
  TCODE         — ME21N, ME28, MIGO, MIRO etc.
  AUTH_OBJECT   — M_BEST_EKG, M_EINK_FRG etc.
  SOD_RULE_ID   — SoD001, SoD_0001, Rule 0001 etc.
  USERNAME      — SAP usernames or UPNs
  COUNTRY       — UK, Germany, United States etc.
  DEPARTMENT    — Procurement, Finance, Manufacturing etc.
  ACTIVITY      — ACTVT values (01, 02, 03 etc.)

Extraction uses a combination of:
  1. Regex patterns for structured entities (roles, tcodes, auth objects)
  2. Keyword lookup for countries, departments
  3. Claude Opus for free-text entity extraction when regex is insufficient
     (only invoked when structured patterns don't cover everything)
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────────────────────────────────────

# SAP role technical names: ZC:P2P:PO_CREATOR________:1000
SAP_ROLE_PATTERN = re.compile(
    r"\b(ZC:[A-Z0-9_]+(?::[A-Z0-9_]+)*(?::\d{4})?)\b",
    re.IGNORECASE
)

# Transaction codes: ME21N, ME28, MIGO, MIRO, FB60, etc.
TCODE_PATTERN = re.compile(
    r"\b(ME\d+[A-Z]?|MIGO|MIRO|MIRO|FB\d+[A-Z]?|VF\d+[A-Z]?|VA\d+[A-Z]?|SU\d+[A-Z]?|PFCG|SM\d+[A-Z]?)\b",
    re.IGNORECASE
)

# Authorization objects: M_BEST_EKG, M_EINK_FRG, Z_XXXXX etc.
AUTH_OBJECT_PATTERN = re.compile(
    r"\b([MZ]_[A-Z0-9]{2,5}_[A-Z0-9]{2,5}(?:_[A-Z0-9]+)?)\b",
    re.IGNORECASE
)

# SoD rule IDs: SoD001, SoD_001, SoD_0001, Rule 0001, Rule001
SOD_RULE_PATTERN = re.compile(
    r"\b(?:SoD[_\s]?\d{3,4}|Rule[_\s]?\d{3,4}|SoD\s*Rule[_\s]?\d{3,4})\b",
    re.IGNORECASE
)

# SAP usernames — typically short alphanumeric (pjunker, pklasson, etc.)
# Only extract if explicitly mentioned as "user X" or "for X"
USERNAME_CONTEXT_PATTERN = re.compile(
    r"(?:user|for|about|check)\s+([a-zA-Z][a-zA-Z0-9_.@-]{3,30})",
    re.IGNORECASE
)

# Email/UPN pattern
UPN_PATTERN = re.compile(
    r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
)

# ACTVT values
ACTIVITY_PATTERN = re.compile(r"\bACTVT\s*[=:]\s*(\d{2})\b", re.IGNORECASE)

# Countries
COUNTRIES = {
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",
    "germany": "Germany",
    "deutschland": "Germany",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "france": "France",
    "india": "India",
    "australia": "Australia",
    "canada": "Canada",
    "netherlands": "Netherlands",
    "spain": "Spain",
    "italy": "Italy",
}

# Departments / Functions
DEPARTMENTS = {
    "procurement": "Procurement & Supply Chain",
    "supply chain": "Procurement & Supply Chain",
    "finance": "Finance",
    "financial": "Finance",
    "accounts payable": "Finance",
    "ap": "Finance",
    "manufacturing": "Manufacturing",
    "warehouse": "Manufacturing",
    "goods receipt": "Manufacturing",
    "sales": "Sales & Marketing",
    "marketing": "Sales & Marketing",
    "hr": "Human Resources",
    "human resources": "Human Resources",
    "it": "IT",
    "information technology": "IT",
    "audit": "Audit & Compliance",
    "compliance": "Audit & Compliance",
}


# ─────────────────────────────────────────────────────────────────────────────
# Extractor class
# ─────────────────────────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts named entities from user queries.

    Two modes:
      1. Regex-only (fast, deterministic) — default for structured entities
      2. LLM-assisted (Opus via Bedrock) — for free-text descriptions where
         regex can't catch "create purchase orders" → TCODE: ME21N
    """

    def __init__(self, bedrock_client=None, model_id: str = ""):
        self._bedrock_client = bedrock_client
        self._model_id = model_id

    def extract(
        self,
        query: str,
        intent_id: str,
        entity_types: List[str],
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract entities from query.

        Args:
            query: The raw user message
            intent_id: Classified intent (used to focus extraction)
            entity_types: Which entity types to extract (from IntentRegistry)
            use_llm: If True, fall back to LLM for entities regex missed

        Returns:
            Dict with entity_type → value/list
        """
        results: Dict[str, Any] = {et: None for et in entity_types}

        text = query.strip()

        # ── Regex extraction ──────────────────────────────────────────────
        if "SAP_ROLE" in entity_types:
            roles = SAP_ROLE_PATTERN.findall(text)
            results["SAP_ROLE"] = roles if roles else None

        if "TCODE" in entity_types:
            tcodes = [t.upper() for t in TCODE_PATTERN.findall(text)]
            results["TCODE"] = tcodes if tcodes else None

        if "AUTH_OBJECT" in entity_types:
            auth_objs = [a.upper() for a in AUTH_OBJECT_PATTERN.findall(text)]
            results["AUTH_OBJECT"] = auth_objs if auth_objs else None

        if "SOD_RULE_ID" in entity_types:
            rule_ids = SOD_RULE_PATTERN.findall(text)
            results["SOD_RULE_ID"] = rule_ids if rule_ids else None

        if "USERNAME" in entity_types:
            upns = UPN_PATTERN.findall(text)
            ctx_users = USERNAME_CONTEXT_PATTERN.findall(text)
            all_users = list(dict.fromkeys(upns + ctx_users))  # deduplicate, preserve order
            results["USERNAME"] = all_users if all_users else None

        if "COUNTRY" in entity_types:
            found_countries = _extract_countries(text)
            results["COUNTRY"] = found_countries if found_countries else None

        if "DEPARTMENT" in entity_types:
            found_depts = _extract_departments(text)
            results["DEPARTMENT"] = found_depts if found_depts else None

        if "ACTIVITY" in entity_types:
            activities = ACTIVITY_PATTERN.findall(text)
            results["ACTIVITY"] = activities if activities else None

        # ── LLM-assisted extraction for missed entities ───────────────────
        # Only invoke if there are entity types that regex missed AND
        # the intent suggests free-text descriptions (role by task, what-if)
        needs_llm = self._needs_llm_extraction(intent_id, results, entity_types)

        if use_llm and needs_llm and self._bedrock_client:
            llm_extras = self._llm_extract(query, intent_id, entity_types, results)
            # Merge: LLM fills gaps, doesn't override regex findings
            for key, val in llm_extras.items():
                if val and results.get(key) is None:
                    results[key] = val

        # Clean up: remove None keys for cleaner output, but keep empty lists
        return {k: v for k, v in results.items() if v is not None}

    def _needs_llm_extraction(
        self,
        intent_id: str,
        current_results: Dict,
        entity_types: List[str],
    ) -> bool:
        """
        Decide if LLM extraction is needed.
        True if: role/tcode entities are expected but regex found nothing,
        AND the intent is task-based (user described a task, not a role name).
        """
        task_based_intents = {
            "DETECT_ROLES_REQUESTED",
            "ELABORATE_ROLE_FUNCTIONS",
            "CHECK_FOR_SOD",
            "RUN_A_WHATIF_SCENARIO",
        }
        if intent_id not in task_based_intents:
            return False

        # If we expect SAP_ROLE or TCODE but found nothing, use LLM
        for et in ["SAP_ROLE", "TCODE"]:
            if et in entity_types and current_results.get(et) is None:
                return True
        return False

    def _llm_extract(
        self,
        query: str,
        intent_id: str,
        entity_types: List[str],
        already_found: Dict,
    ) -> Dict[str, Any]:
        """
        Use Claude Opus to extract entities that regex couldn't find.
        E.g. "I need to create purchase orders" → TCODE: ["ME21N", "ME22N"]
        """
        already_str = json.dumps({k: v for k, v in already_found.items() if v}, indent=2)
        types_str = ", ".join(entity_types)

        prompt = f"""You are an SAP entity extractor for a Segregation of Duties system.

Extract the following entity types from the user query: {types_str}

User query: "{query}"
Intent: {intent_id}
Already extracted by regex: {already_str}

SAP knowledge to apply:
- "create purchase orders" → TCODE: ME21N, ME22N
- "approve/release purchase orders" → TCODE: ME28, ME29N
- "goods receipt" or "GR" → TCODE: MIGO
- "invoice posting" or "park invoice" → TCODE: MIRO
- "vendor master" → TCODE: XK01, XK02
- "payment" → TCODE: F110, F-53

Return ONLY a JSON object with entity_type as keys and extracted values as arrays.
Only include entities you are confident about. Use null for not found.
Example: {{"TCODE": ["ME21N"], "SAP_ROLE": null, "DEPARTMENT": ["Procurement"]}}"""

        try:
            import json as json_mod
            body = json_mod.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            })
            response = self._bedrock_client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            raw = json_mod.loads(response["body"].read())["content"][0]["text"].strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            return json_mod.loads(raw)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_countries(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for keyword, canonical in COUNTRIES.items():
        if keyword in text_lower and canonical not in found:
            found.append(canonical)
    return found


def _extract_departments(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for keyword, canonical in DEPARTMENTS.items():
        if keyword in text_lower and canonical not in found:
            found.append(canonical)
    return found
