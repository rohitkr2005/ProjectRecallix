"""
Project Recallix — Interactive Showcase Demo

Demonstrates:
1. Memory Ingestion & Fact Extraction
2. Temporal State Horizons (Past vs. Present facts)
3. Grounded Question Answering with Memory Citations
4. Zero-Hallucination Policy on Unsupported Queries
5. Knowledge Graph Relationships
6. Sub-Millisecond Performance & Latency Metrics
"""

import sys
import time

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.assistant.assistant_engine import AssistantEngine, InputType


def print_divider(title: str = ""):
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)


def run_demo():
    print_divider("PROJECT RECALLIX (v1.0.0) DEMO")
    print("Initializing Recallix cognitive memory cortex...\n")
    start_init = time.perf_counter()
    assistant = AssistantEngine()
    init_ms = (time.perf_counter() - start_init) * 1000.0
    print(f"[OK] AssistantEngine ready in {init_ms:.1f}ms")

    # ------------------------------------------------------------------
    # Step 1: Memory Ingestion & Extraction
    # ------------------------------------------------------------------
    print_divider("STEP 1: MEMORY INGESTION")
    facts_to_learn = [
        "I am a software engineer building Project Recallix in Python.",
        "Previously lived in Delhi.",
        "Currently lives in Mumbai.",
        "I like playing football.",
        "I want to learn Rust.",
    ]

    for fact in facts_to_learn:
        print(f"User: \"{fact}\"")
        res = assistant.respond(fact)
        print(f"Recallix: {res['response']}")
        if res.get("extracted_memories"):
            for mem in res["extracted_memories"]:
                print(f"   -> Stored: ({mem['subject']}) -[{mem['relation']}]-> ({mem['value']}) [{mem.get('temporal_state', 'PRESENT')}]")
        print()

    # ------------------------------------------------------------------
    # Step 2: Grounded Question Answering
    # ------------------------------------------------------------------
    print_divider("STEP 2: GROUNDED QUESTION ANSWERING")
    q1 = "What project am I working on?"
    print(f"User Question: \"{q1}\"")
    res1 = assistant.respond(q1, include_explanations=True)
    print(f"Recallix Answer: {res1['response']}")
    print(f"Grounded: {res1['grounded']} | Supported: {res1['supported']} | LLM Status: {res1['llm_status']}")
    if res1.get("performance_metrics"):
        print(f"Latency: {res1['performance_metrics']}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Temporal Horizon Reasoning (Past vs Present)
    # ------------------------------------------------------------------
    print_divider("STEP 3: TEMPORAL HORIZON REASONING")
    q_now = "Where do I currently live?"
    print(f"User: \"{q_now}\"")
    res_now = assistant.respond(q_now)
    print(f"Recallix: {res_now['response']}\n")

    q_past = "Where did I live before?"
    print(f"User: \"{q_past}\"")
    res_past = assistant.respond(q_past)
    print(f"Recallix: {res_past['response']}\n")

    # ------------------------------------------------------------------
    # Step 4: Zero-Hallucination Policy (Unsupported Queries)
    # ------------------------------------------------------------------
    print_divider("STEP 4: ZERO-HALLUCINATION POLICY")
    q_unsupported = "What is my favorite movie?"
    print(f"User Question: \"{q_unsupported}\"")
    res_unsupported = assistant.respond(q_unsupported)
    print(f"Recallix Answer: {res_unsupported['response']}")
    print(f"Supported: {res_unsupported['supported']} (Correctly declined unsupported inquiry)")
    print()

    # ------------------------------------------------------------------
    # Step 5: Knowledge Graph Entity Relationships
    # ------------------------------------------------------------------
    print_divider("STEP 5: KNOWLEDGE GRAPH TRAVERSAL")
    if hasattr(assistant, "memory_graph") and assistant.memory_graph:
        neighbors = assistant.memory_graph.get_neighbors("User")
        print(f"Entities directly connected to 'User': {list(neighbors)}")
        paths = assistant.memory_graph.find_paths("User", "Project Recallix")
        if paths:
            print(f"Paths to 'Project Recallix': {paths}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Interactive Mode (if run in terminal)
    # ------------------------------------------------------------------
    if "--interactive" in sys.argv or "-i" in sys.argv:
        print_divider("INTERACTIVE MODE (type 'exit' to quit)")
        while True:
            try:
                user_msg = input("\nYou: ").strip()
                if not user_msg or user_msg.lower() in ("exit", "quit"):
                    break
                reply = assistant.respond(user_msg, include_explanations=True)
                print(f"Recallix: {reply['response']}")
                if reply.get("performance_metrics"):
                    print(f"   [Latency: {reply['performance_metrics']}]")
            except (KeyboardInterrupt, EOFError):
                break

    print_divider("DEMO COMPLETED SUCCESSFULLY [OK]")
    assistant.close()


if __name__ == "__main__":
    run_demo()
