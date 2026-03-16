"""
Intent Classifier Node V2
─────────────────────────
Updated LangGraph node that uses:
  - IntentRegistry (V3.0 sheet as master intent source)
  - IntentVectorStore V2 (sample queries from V3.0 sheet embedded)
  - EntityExtractor (regex + LLM-assisted)
  - PromptBuilder V2 (injects sub-intents, entity types, response templates)

Output per query:
  intent_id           ← Col B (machine ID)
  intent_name         ← Col B (human name)
  sub_intent          ← Col C (most specific Action Intended)
  confidence          ← 0.0–1.0
  extracted_entities  ← SAP_ROLE, TCODE, AUTH_OBJECT, USERNAME, COUNTRY, DEPT...
  persona_gate_passed ← Col D validation
  allowed_personas    ← Col D
  entity_types        ← Col C/G derived
  bot_response_template ← Col F closest match
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any
import boto3

from src.intent_registry import IntentRegistry
from src.vector_store_v2 import IntentVectorStore
from src.entity_extractor import EntityExtractor
from src.prompt_builder_v2 import (
    build_system_prompt,
    build_user_prompt,
    get_allowed_records_for_persona,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output model (plain dataclass — no Pydantic dependency needed here)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationResult:
    def __init__(self, data: Dict[str, Any], processing_time_ms: float, retrieval_info: list):
        self.intent_id: str = data.get("intent_id") or "UNKNOWN"
        self.intent_name: str = data.get("intent_name") or ""
        self.sub_intent: str = data.get("sub_intent") or ""
        self.confidence: float = float(data.get("confidence") or 0.0)
        self.extracted_entities: Dict = data.get("extracted_entities") or {}
        self.persona_gate_passed: bool = bool(data.get("persona_gate_passed", True))
        self.requires_clarification: bool = bool(data.get("requires_clarification", False))
        self.clarification_question: Optional[str] = data.get("clarification_question")
        self.reasoning: str = data.get("reasoning") or ""
        self.bot_response_template: Optional[str] = data.get("suggested_bot_response_template")
        self.processing_time_ms: float = processing_time_ms
        self.retrieval_info: list = retrieval_info

    def to_dict(self) -> Dict:
        return {
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "sub_intent": self.sub_intent,
            "confidence": self.confidence,
            "extracted_entities": self.extracted_entities,
            "persona_gate_passed": self.persona_gate_passed,
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
            "reasoning": self.reasoning,
            "bot_response_template": self.bot_response_template,
            "processing_time_ms": self.processing_time_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifierV2:
    """
    Main classifier. Can be used standalone or as a LangGraph node.

    Standalone usage:
        classifier = IntentClassifierV2(registry, vector_store)
        result = classifier.classify(
            message="Please give me access to ZC:P2P:PO_CREATOR________:1000",
            authenticated_user="pjunker",
            persona="END_USER",
            user_level="L4",
            department="Procurement & Supply Chain",
            country="United Kingdom",
        )
        print(result.intent_id)           # DETECT_ROLES_REQUESTED
        print(result.sub_intent)          # Access Request (with preventive SoD)
        print(result.extracted_entities)  # {"SAP_ROLE": ["ZC:P2P:PO_CREATOR________:1000"]}
    """

    def __init__(
        self,
        registry: IntentRegistry,
        vector_store: IntentVectorStore,
        few_shot_k: int = 5,
        confidence_threshold: float = 0.70,
        bedrock_model_id: Optional[str] = None,
        aws_region: Optional[str] = None,
    ):
        self.registry = registry
        self.vector_store = vector_store
        self.few_shot_k = few_shot_k
        self.confidence_threshold = confidence_threshold
        self.model_id = bedrock_model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "arn:aws:bedrock:eu-west-2:613564179512:inference-profile/eu.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self.aws_region = aws_region or os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")
        self._bedrock = None

    def _get_bedrock(self):
        if self._bedrock is None:
            self._bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
        return self._bedrock

    def classify(
        self,
        message: str,
        authenticated_user: str,
        persona: str = "END_USER",
        user_level: str = "L4",
        department: str = "",
        country: str = "",
    ) -> ClassificationResult:
        start = time.time()

        # 1. Get allowed intents for this persona
        allowed_records = get_allowed_records_for_persona(self.registry, persona)

        # 2. Retrieve similar examples from vector store
        retrieved = self.vector_store.search(
            query=message,
            k=self.few_shot_k,
            persona_filter=persona,
        )
        retrieval_info = [
            {"text": doc.text, "intent_id": doc.intent_id,
             "sub_intent": doc.sub_intent_label, "score": round(score, 3)}
            for doc, score in retrieved
        ]

        # 3. Build prompts
        system_prompt = build_system_prompt(self.registry, allowed_records)
        user_prompt = build_user_prompt(
            message=message,
            authenticated_user=authenticated_user,
            persona=persona,
            user_level=user_level,
            department=department,
            country=country,
            retrieved_examples=retrieved,
        )

        # 4. Call Bedrock
        raw = self._call_bedrock(system_prompt, user_prompt)

        # 5. Parse response
        data = self._parse_json(raw)

        # 6. Post-process: entity extraction (regex layer on top of LLM)
        intent_id = data.get("intent_id", "UNKNOWN")
        record = self.registry.get_by_id(intent_id)
        entity_types = record.entity_types if record else []

        extractor = EntityExtractor(
            bedrock_client=self._get_bedrock() if self._needs_llm_entities(intent_id) else None,
            model_id=self.model_id,
        )
        regex_entities = extractor.extract(
            query=message,
            intent_id=intent_id,
            entity_types=entity_types,
            use_llm=False,  # regex only — LLM already extracted in main call
        )

        # Merge: LLM entities + regex entities (regex is more precise for patterns)
        llm_entities = data.get("extracted_entities", {}) or {}
        merged_entities = _merge_entities(llm_entities, regex_entities)
        data["extracted_entities"] = merged_entities

        # 7. Hard persona gate
        data = self._validate_persona_gate(data, persona)

        # 8. Confidence gate
        confidence = float(data.get("confidence", 0.0))
        if confidence < self.confidence_threshold and not data.get("requires_clarification"):
            data["requires_clarification"] = True
            if not data.get("clarification_question"):
                data["clarification_question"] = (
                    "Could you give me a bit more detail? Are you asking about your own access, "
                    "a specific user, a role name, or a compliance report?"
                )

        elapsed = round((time.time() - start) * 1000, 1)
        result = ClassificationResult(data, elapsed, retrieval_info)

        logger.info(
            f"[{result.intent_id}] '{message[:50]}' | "
            f"sub={result.sub_intent[:40]!r} | "
            f"conf={result.confidence:.2f} | {elapsed}ms"
        )
        return result

    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_bedrock()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        response = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["content"][0]["text"].strip()

    def _parse_json(self, raw: str) -> Dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error. Raw: {raw[:200]}")
            return {
                "intent_id": "UNKNOWN",
                "intent_name": "Unknown",
                "sub_intent": "",
                "confidence": 0.0,
                "extracted_entities": {},
                "persona_gate_passed": True,
                "requires_clarification": True,
                "clarification_question": "I couldn't understand your request. Could you rephrase?",
                "reasoning": "JSON parse failure.",
            }

    def _validate_persona_gate(self, data: Dict, persona: str) -> Dict:
        intent_id = data.get("intent_id", "")
        record = self.registry.get_by_id(intent_id)
        if record and persona not in record.allowed_personas:
            logger.warning(f"Persona gate BLOCKED: {persona} → {intent_id}")
            data["intent_id"] = "OUT_OF_SCOPE"
            data["persona_gate_passed"] = False
            data["requires_clarification"] = False
            data["clarification_question"] = (
                f"This type of query requires elevated access. "
                f"Your role ({persona}) does not have permission. "
                f"Please contact your Process Owner or Auditor."
            )
        return data

    def _needs_llm_entities(self, intent_id: str) -> bool:
        return intent_id in {
            "DETECT_ROLES_REQUESTED",
            "CHECK_FOR_SOD",
            "ELABORATE_ROLE_FUNCTIONS",
            "RUN_A_WHATIF_SCENARIO",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _merge_entities(llm: Dict, regex: Dict) -> Dict:
    """
    Merge LLM-extracted and regex-extracted entities.
    Regex wins for structured patterns (roles, tcodes, auth objects).
    LLM fills gaps for semantic entities (departments, task descriptions).
    """
    merged = {}
    all_keys = set(list(llm.keys()) + list(regex.keys()))

    # Regex-preferred keys
    regex_preferred = {"SAP_ROLE", "TCODE", "AUTH_OBJECT", "SOD_RULE_ID", "ACTIVITY"}

    for key in all_keys:
        regex_val = regex.get(key)
        llm_val = llm.get(key)

        if key in regex_preferred:
            val = regex_val if regex_val else llm_val
        else:
            val = llm_val if llm_val else regex_val

        if val:
            merged[key] = val

    return merged