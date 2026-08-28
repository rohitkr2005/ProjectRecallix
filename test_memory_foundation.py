from app.memory.memory_store import MemoryStore


store = MemoryStore()


print("\n========================================")
print("RECALLIX MEMORY FOUNDATION TEST")
print("========================================")


# ---------------------------------------------------------
# 1. Create a new memory
# ---------------------------------------------------------

print("\n[1] Creating new memory...")

memory, status = store.save_memory(
    subject="IntegrationUser",
    relation="likes",
    value="Python",
    category="TEST",
    importance=8
)

print("Status:", status)
print("ID:", memory.id)


# ---------------------------------------------------------
# 2. Exact duplicate
# ---------------------------------------------------------

print("\n[2] Testing duplicate detection...")

memory, status = store.save_memory(
    subject="IntegrationUser",
    relation="likes",
    value="Python",
    category="TEST",
    importance=8
)

print("Status:", status)
print("ID:", memory.id)


# ---------------------------------------------------------
# 3. Multi-value relation
# ---------------------------------------------------------

print("\n[3] Testing multi-value relation...")

memory, status = store.save_memory(
    subject="IntegrationUser",
    relation="likes",
    value="SQL",
    category="TEST",
    importance=7
)

print("Status:", status)
print("ID:", memory.id)


# ---------------------------------------------------------
# 4. Single-value relation
# ---------------------------------------------------------

print("\n[4] Creating single-value memory...")

memory, status = store.save_memory(
    subject="IntegrationUser",
    relation="lives_in",
    value="Delhi",
    category="TEST",
    importance=8
)

print("Status:", status)
print("ID:", memory.id)
print("Value:", memory.value)


# ---------------------------------------------------------
# 5. Update single-value relation
# ---------------------------------------------------------

print("\n[5] Testing single-value update...")

memory, status = store.save_memory(
    subject="IntegrationUser",
    relation="lives_in",
    value="Mumbai",
    category="TEST",
    importance=9
)

print("Status:", status)
print("ID:", memory.id)
print("Value:", memory.value)


# ---------------------------------------------------------
# 6. Archive memory
# ---------------------------------------------------------

print("\n[6] Testing memory archival...")

result = store.deactivate_memory(memory.id)

print("Deactivation successful:", result)


# ---------------------------------------------------------
# 7. Verify archived memory
# ---------------------------------------------------------

print("\n[7] Checking archived memories...")

for item in store.get_archived_memories():

    if item.subject == "IntegrationUser":

        print(
            item.id,
            "|",
            item.subject,
            "|",
            item.relation,
            "|",
            item.value
        )


# ---------------------------------------------------------
# 8. Restore memory
# ---------------------------------------------------------

print("\n[8] Testing memory restoration...")

result = store.restore_memory(memory.id)

print("Restoration successful:", result)


# ---------------------------------------------------------
# 9. Final verification
# ---------------------------------------------------------

print("\n[9] Final active memories...")

integration_memories = [
    item
    for item in store.get_all_memories()
    if item.subject == "IntegrationUser"
]

for item in integration_memories:

    print(
        item.id,
        "|",
        item.subject,
        "|",
        item.relation,
        "|",
        item.value,
        "| active:",
        item.active
    )


# ---------------------------------------------------------
# 10. Validation
# ---------------------------------------------------------

print("\n[10] Validation...")

python_memories = [
    item
    for item in integration_memories
    if item.relation == "likes"
    and item.value == "Python"
]

sql_memories = [
    item
    for item in integration_memories
    if item.relation == "likes"
    and item.value == "SQL"
]

location_memories = [
    item
    for item in integration_memories
    if item.relation == "lives_in"
]

if len(python_memories) == 1 and \
   len(sql_memories) == 1 and \
   len(location_memories) == 1 and \
   location_memories[0].value == "Mumbai" and \
   location_memories[0].active:

    print("\nALL MEMORY FOUNDATION TESTS PASSED!")

else:

    print("\nMEMORY FOUNDATION TEST FAILED.")


store.close()