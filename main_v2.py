"""
PAX Intent Classifier V2 — Main Entrypoint
───────────────────────────────────────────
Driven entirely by the V3.0 intent sheet.

Commands:
  python main_v2.py setup   — parse V3.0 sheet, build vector store
  python main_v2.py run     — interactive REPL
  python main_v2.py test    — run test suite against all 6 intents
  python main_v2.py show    — print the full parsed intent registry
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent
EXCEL_V3        = BASE_DIR / "data" / "intent_definitions_v3.xlsx"
VECTOR_STORE    = BASE_DIR / "data" / "vector_store_v2"


# ─────────────────────────────────────────────────────────────────────────────

def cmd_setup():
    from src.intent_registry import IntentRegistry
    from src.vector_store_v2 import build_vector_store

    if not EXCEL_V3.exists():
        print(f"❌ File not found: {EXCEL_V3}")
        sys.exit(1)

    print(f"📊 Parsing intent registry from: {EXCEL_V3}")
    registry = IntentRegistry(str(EXCEL_V3))
    print(registry.summary())

    print(f"\n🔢 Building vector store...")
    store = build_vector_store(registry, save_path=str(VECTOR_STORE))
    print(f"\n✅ Done. {store.total} vectors indexed → {VECTOR_STORE}")
    print("\nRun: python main_v2.py run")


def cmd_show():
    from src.intent_registry import IntentRegistry
    registry = IntentRegistry(str(EXCEL_V3))
    print(registry.summary())


def cmd_run():
    from src.intent_registry import IntentRegistry
    from src.vector_store_v2 import IntentVectorStore
    from src.classifier_v2 import IntentClassifierV2

    _check_setup()
    registry = IntentRegistry(str(EXCEL_V3))
    store    = IntentVectorStore.load(str(VECTOR_STORE))
    clf      = IntentClassifierV2(registry, store)

    PERSONAS = ["END_USER", "PROCESS_OWNER", "DATA_OWNER", "APP_OWNER", "AUDITOR"]

    persona  = "END_USER"
    user     = "pjunker"
    dept     = "Procurement & Supply Chain"
    country  = "United Kingdom"

    print("\n" + "="*65)
    print(" PAX Identity EKG — Intent Classifier V2 (Interactive)")
    print("="*65)
    print(f" User: {user}  |  Persona: {persona}  |  {dept}, {country}")
    print(" Type 'persona' to switch persona, 'quit' to exit.\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); break

        if not msg:
            continue
        if msg.lower() == "quit":
            break
        if msg.lower() == "persona":
            print(f"Personas: {PERSONAS}")
            p = input("Enter persona: ").strip().upper()
            if p in PERSONAS:
                persona = p
                print(f"✅ Persona → {persona}\n")
            else:
                print(f"❌ Invalid\n")
            continue

        print("\n🔄 Classifying...")
        result = clf.classify(
            message=msg,
            authenticated_user=user,
            persona=persona,
            user_level="L4",
            department=dept,
            country=country,
        )
        _print_result(result)
        print()


def cmd_test():
    from src.intent_registry import IntentRegistry
    from src.vector_store_v2 import IntentVectorStore
    from src.classifier_v2 import IntentClassifierV2

    _check_setup()
    registry = IntentRegistry(str(EXCEL_V3))
    store    = IntentVectorStore.load(str(VECTOR_STORE))
    clf      = IntentClassifierV2(registry, store)

    # One test case per intent from V3.0 sheet
    test_cases = [
        # (message, persona, expected_intent_id, expected_entity_keys)
        (
            "I have just joined the procurement team in the UK",
            "END_USER", "VALIDATE_USER_REQUEST", ["DEPARTMENT", "COUNTRY"]
        ),
        (
            "I am leaving the organisation please revoke all my SAP accesses",
            "END_USER", "VALIDATE_USER_REQUEST", []
        ),
        (
            "I want to find out what roles my peers have",
            "END_USER", "DETECT_ROLES_REQUESTED", []
        ),
        (
            "Please give me access to this role: ZC:P2P:PO_CREATOR________:1000",
            "END_USER", "DETECT_ROLES_REQUESTED", ["SAP_ROLE"]
        ),
        (
            "I am a purchaser and need permission to create Purchase orders",
            "END_USER", "DETECT_ROLES_REQUESTED", ["TCODE"]
        ),
        (
            "I am sorry, you cannot access this role because of SoD policy. Yes I want more info.",
            "END_USER", "CHECK_FOR_SOD", []
        ),
        (
            "I would like to understand why Role assignment ZC:P2P:PO_CREATOR________:1000 for user pjunker is flagged as a risk",
            "AUDITOR", "CHECK_FOR_SOD", ["SAP_ROLE", "USERNAME"]
        ),
        (
            "Can you explain what the functions of each of these roles are",
            "END_USER", "ELABORATE_ROLE_FUNCTIONS", []
        ),
        (
            "I need a new Goods Receipt Processor UK plant 1000 role that only posts goods receipts",
            "PROCESS_OWNER", "ELABORATE_ROLE_FUNCTIONS", ["TCODE"]
        ),
        (
            "Please can you provide me with a summary of all accesses my direct reports have",
            "END_USER", "KNOWLEDGE_GRAPH_BASED_SCENARIOS", []
        ),
        (
            "I want to assess all access risks in the SAP application",
            "AUDITOR", "KNOWLEDGE_GRAPH_BASED_SCENARIOS", []
        ),
        (
            "If I remove ME29 and the release authorization objects from ZC:P2P:PO_CREATOR________:1000 will the SoD conflict disappear?",
            "PROCESS_OWNER", "RUN_A_WHATIF_SCENARIO", ["SAP_ROLE", "TCODE"]
        ),
        # Persona gate: END_USER should NOT get what-if
        (
            "If I remove ME28 from this role will the SoD conflict go away?",
            "END_USER", "OUT_OF_SCOPE", []
        ),
    ]

    print("\n" + "="*70)
    print(" PAX Intent Classifier V2 — Test Suite")
    print("="*70)

    passed = failed = 0
    for msg, persona, expected_intent, expected_entity_keys in test_cases:
        r = clf.classify(
            message=msg,
            authenticated_user="test_user",
            persona=persona,
            user_level="L4",
            department="Procurement & Supply Chain",
            country="United Kingdom",
        )
        intent_ok = r.intent_id == expected_intent
        entity_ok = all(k in r.extracted_entities for k in expected_entity_keys)
        ok = intent_ok and entity_ok

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} | Persona={persona:15}")
        print(f"       Query   : {msg[:70]!r}")
        print(f"       Expected: {expected_intent:40} | Got: {r.intent_id}")
        if expected_entity_keys:
            found = {k: r.extracted_entities.get(k) for k in expected_entity_keys}
            print(f"       Entities: expected_keys={expected_entity_keys} | found={found}")
        print(f"       Sub-intent: {r.sub_intent[:60]}")
        print(f"       Confidence: {r.confidence:.2f} | Reasoning: {r.reasoning[:70]}")

    print(f"\n{'='*70}")
    print(f" Results: {passed}/{passed+failed} passed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_setup():
    if not (VECTOR_STORE / "index.faiss").exists():
        print("❌ Vector store not found. Run: python main_v2.py setup")
        sys.exit(1)


def _print_result(result):
    print(f"\n{'─'*60}")
    print(f"  Intent ID    : {result.intent_id}")
    print(f"  Intent Name  : {result.intent_name}")
    print(f"  Sub-Intent   : {result.sub_intent}")
    print(f"  Confidence   : {result.confidence:.2f}")
    print(f"  Persona OK   : {result.persona_gate_passed}")
    print(f"  Clarify?     : {result.requires_clarification}")

    if result.requires_clarification and result.clarification_question:
        print(f"\n  🤔 Bot asks  : {result.clarification_question}")

    if result.extracted_entities:
        print(f"\n  Entities:")
        for k, v in result.extracted_entities.items():
            print(f"    {k:20}: {v}")

    if result.bot_response_template:
        preview = result.bot_response_template[:120].replace("\n", " ")
        print(f"\n  Bot template : {preview}...")

    print(f"\n  Reasoning    : {result.reasoning}")
    print(f"  Time         : {result.processing_time_ms}ms")

    if result.retrieval_info:
        print(f"\n  Top examples retrieved:")
        for ex in result.retrieval_info[:3]:
            print(f"    [{ex['score']:.2f}] {ex['intent_id']:35} | {ex['text'][:50]!r}")

    print(f"{'─'*60}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"setup": cmd_setup, "run": cmd_run, "test": cmd_test, "show": cmd_show}.get(
        cmd, lambda: print(f"Unknown: {cmd}\nUsage: python main_v2.py [setup|run|test|show]")
    )()
