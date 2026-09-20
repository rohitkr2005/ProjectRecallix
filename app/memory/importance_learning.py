from datetime import datetime
from typing import Dict, List, Optional, Any


class ImportanceLearner:
    """
    10.1 Memory Importance Learning
    10.2 Retrieval Feedback
    10.3 Memory Reinforcement

    Tracks usage patterns and feedback to dynamically learn and reinforce memory importance.
    """

    def __init__(self, session=None):
        self.session = session

    @staticmethod
    def calculate_effective_importance(
        base_importance: int = 5,
        access_count: int = 0,
        helpful_count: int = 0,
        reinforcement_score: float = 0.0,
    ) -> float:
        """
        Calculate dynamic importance learned from access history and feedback.
        Clamped between 1.0 and 10.0.
        """
        base = float(base_importance if base_importance is not None else 5)
        acc = int(access_count or 0)
        help_cnt = int(helpful_count or 0)
        reinf = float(reinforcement_score or 0.0)

        # Helpful reinforcement boost (up to +3.0)
        boost = min(3.0, (help_cnt * 0.5) + (reinf * 0.3))

        # Penalty if accessed repeatedly but never helpful (up to -2.0)
        unhelpful_accesses = max(0, acc - help_cnt)
        penalty = min(2.0, unhelpful_accesses * 0.2)

        effective = base + boost - penalty
        return round(max(1.0, min(10.0, effective)), 2)

    def record_access(
        self,
        memory: Any,
        was_helpful: bool = False,
        session=None,
    ) -> Dict[str, Any]:
        """
        Record that a memory was retrieved and whether it was helpful/cited.
        Updates access_count, helpful_count, last_accessed_at, and reinforcement_score.
        """
        s = session or self.session

        current_access = getattr(memory, "access_count", 0) or 0
        current_helpful = getattr(memory, "helpful_count", 0) or 0
        current_reinf = getattr(memory, "reinforcement_score", 0.0) or 0.0
        base_imp = getattr(memory, "importance", 5) or 5

        new_access = current_access + 1
        new_helpful = current_helpful + (1 if was_helpful else 0)

        if was_helpful:
            new_reinf = round(current_reinf + 0.5, 2)
        else:
            new_reinf = round(max(0.0, current_reinf - 0.1), 2)

        now = datetime.utcnow()

        if hasattr(memory, "access_count"):
            memory.access_count = new_access
        if hasattr(memory, "helpful_count"):
            memory.helpful_count = new_helpful
        if hasattr(memory, "reinforcement_score"):
            memory.reinforcement_score = new_reinf
        if hasattr(memory, "last_accessed_at"):
            memory.last_accessed_at = now

        effective_imp = self.calculate_effective_importance(
            base_importance=base_imp,
            access_count=new_access,
            helpful_count=new_helpful,
            reinforcement_score=new_reinf,
        )

        if s is not None and hasattr(s, "commit"):
            try:
                s.commit()
            except Exception:
                pass

        return {
            "memory_id": getattr(memory, "id", None),
            "access_count": new_access,
            "helpful_count": new_helpful,
            "reinforcement_score": new_reinf,
            "effective_importance": effective_imp,
            "last_accessed_at": now.isoformat(),
        }

    def apply_retrieval_feedback(
        self,
        retrieved_memories: List[Any],
        supported_values: Optional[List[str]] = None,
        session=None,
    ) -> Dict[str, Any]:
        """
        Apply feedback across a list of retrieved memories based on whether their values
        were cited in the grounded answer.
        """
        supported_set = {str(v).strip().lower() for v in (supported_values or []) if v}
        helpful_ids = []
        unhelpful_ids = []
        updates = []

        for item in retrieved_memories or []:
            mem = item.get("memory") if isinstance(item, dict) else item
            if mem is None:
                continue

            mem_val = str(getattr(mem, "value", "")).strip().lower()
            mem_id = getattr(mem, "id", None)

            # Check if memory value was cited in the answer grounding
            was_helpful = bool(mem_val and any(sup in mem_val or mem_val in sup for sup in supported_set))

            res = self.record_access(mem, was_helpful=was_helpful, session=session)
            updates.append(res)
            if was_helpful:
                helpful_ids.append(mem_id)
            else:
                unhelpful_ids.append(mem_id)

        return {
            "total_processed": len(updates),
            "helpful_ids": helpful_ids,
            "unhelpful_ids": unhelpful_ids,
            "updates": updates,
        }
