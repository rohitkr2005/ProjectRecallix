from app.assistant.assistant_engine import AssistantEngine
print("=" * 60)
print("RECALLIX STEP 5.1 — FULL ASSISTANT PIPELINE TEST")
print("=" * 60)
assistant = AssistantEngine()
print("\n--- TEST 1: Ollama availability ---")
available = assistant.is_available()
print("Ollama available:", available)
if not available:
    print("Ollama is not reachable.")
    assistant.close()
    raise SystemExit(1)
print("\n--- TEST 2: Project memory retrieval + filtering ---")
question = "What projects am I working on?"
print("User:", question)
result = assistant.respond(
    user_message=question,
    top_k=5,
    min_score=0.0,
    memory_relevance=0.25,
)
print("\nDetected intent:")
print(result["intent"])
print("\n--- ALL RETRIEVED MEMORIES ---")
for index, item in enumerate(
    result["retrieved_memories"],
    start=1,
):
    memory = item["memory"]
    print(
        f"{index}. "
        f"{memory.subject} | "
        f"{memory.relation} | "
        f"{memory.value} | "
        f"{memory.category} | "
        f"score: {item['score']:.4f}"
    )
print("\n--- MEMORIES SENT TO LLM ---")
for index, item in enumerate(
    result["memories"],
    start=1,
):
    memory = item["memory"]
    print(
        f"{index}. "
        f"{memory.subject} | "
        f"{memory.relation} | "
        f"{memory.value} | "
        f"{memory.category} | "
        f"score: {item['score']:.4f}"
    )
print("\n--- MEMORY CONTEXT SENT TO LLM ---")
context = assistant.llm_engine._build_memory_context(
    result["memories"]
)
print(context)
print("\n--- GENERATED RESPONSE ---")
print(result["response"])
print("\n--- TEST 3: Preference question ---")
question = "What sports do I like?"
print("User:", question)
result = assistant.respond(
    user_message=question,
    top_k=5,
    min_score=0.0,
    memory_relevance=0.25,
)
print("\nDetected intent:")
print(result["intent"])
print("\nMemories sent to LLM:")
for index, item in enumerate(
    result["memories"],
    start=1,
):
    memory = item["memory"]
    print(
        f"{index}. "
        f"{memory.subject} | "
        f"{memory.relation} | "
        f"{memory.value} | "
        f"{memory.category} | "
        f"score: {item['score']:.4f}"
    )
print("\nGenerated response:")
print(result["response"])
assistant.close()
print("\n" + "=" * 60)
print("STEP 5.1 ASSISTANT PIPELINE TEST COMPLETE")
print("=" * 60)
