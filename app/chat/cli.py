import sys
import os

from app.assistant.assistant_engine import AssistantEngine, InputType


class ChatCLI:
    """Interactive command-line chat interface for Project Recallix."""

    BANNER = """================================================================
          PROJECT RECALLIX — MEMORY-AWARE ASSISTANT
================================================================
Commands:
  /memories  - List all active memories currently stored
  /explain   - Toggle retrieval explanations on/off
  /help      - Display this help message
  /clear     - Clear the console screen
  /exit      - Exit the assistant
================================================================"""

    def __init__(self, assistant=None, out_stream=None):
        self.assistant = assistant or AssistantEngine()
        self.out_stream = out_stream or sys.stdout
        self.include_explanations = False

    def print(self, message: str = ""):
        print(message, file=self.out_stream, flush=True)

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if the chat should continue, False to exit."""
        cmd = cmd.strip().lower()

        if cmd in ("/exit", "/quit"):
            self.print("Goodbye! Your memories remain securely stored.")
            return False

        if cmd == "/help":
            self.print(self.BANNER)
            return True

        if cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            return True

        if cmd == "/explain":
            self.include_explanations = not self.include_explanations
            status = "ENABLED" if self.include_explanations else "DISABLED"
            self.print(f"[Explainability: {status}]")
            return True

        if cmd == "/memories":
            self._list_memories()
            return True

        self.print(f"Unknown command: '{cmd}'. Type /help for available commands.")
        return True

    def _list_memories(self):
        memories = self.assistant.memory_store.get_all_memories()
        active_memories = [m for m in memories if getattr(m, "active", True)]

        if not active_memories:
            self.print("No active memories stored yet.")
            return

        self.print(f"\n--- Stored Memories ({len(active_memories)}) ---")
        for i, m in enumerate(active_memories, start=1):
            subj = getattr(m, "subject", "User")
            rel = getattr(m, "relation", "").replace("_", " ")
            val = getattr(m, "value", "")
            cat = getattr(m, "category", "")
            self.print(f"  {i}. [{cat}] {subj} {rel} {val}")
        self.print("---------------------------------\n")

    def process_input(self, user_input: str) -> str:
        """Process a line of user input, return the formatted assistant response."""
        if not user_input or not user_input.strip():
            return ""

        text = user_input.strip()
        if text.startswith("/"):
            # Command handled externally or returns empty response
            self.handle_command(text)
            return ""

        result = self.assistant.respond(
            user_message=text,
            include_explanations=self.include_explanations,
        )

        response_text = result.get("response", "")
        output_lines = [f"Recallix > {response_text}"]

        # If memories were saved, provide detail
        saved = result.get("extracted_memories", [])
        if saved:
            for s in saved:
                status = s.get("status", "saved")
                rel = s.get("relation", "").replace("_", " ")
                val = s.get("value", "")
                output_lines.append(f"  [Memory {status}: {rel} -> {val}]")

        # If explanations enabled, show why_used
        if self.include_explanations:
            why_used = result.get("why_used", [])
            if why_used:
                output_lines.append("  [Explanations:")
                for r in why_used:
                    output_lines.append(f"    * {r}")
                output_lines.append("  ]")

        formatted = "\n".join(output_lines)
        return formatted

    def run(self):
        """Run the interactive REPL loop."""
        self.print(self.BANNER)

        while True:
            try:
                user_input = input("\nUser > ").strip()
            except (EOFError, KeyboardInterrupt):
                self.print("\nExiting Recallix. Goodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = self.handle_command(user_input)
                if not should_continue:
                    break
                continue

            output = self.process_input(user_input)
            if output:
                self.print(output)


def run_chat_cli():
    cli = ChatCLI()
    cli.run()


if __name__ == "__main__":
    run_chat_cli()
