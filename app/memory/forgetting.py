from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from app.memory.importance_learning import ImportanceLearner


class MemoryForgetter:
    """
    10.4 Memory Forgetting

    Controlled, non-destructive memory forgetting / archival based on:
    - Stale last access / age
    - Low effective importance
    - Lack of reinforcement
    """

    DEFAULT_DECAY_DAYS = 30
    DEFAULT_MIN_IMPORTANCE = 2

    def __init__(self, session=None, importance_learner=None):
        self.session = session
        self.learner = importance_learner or ImportanceLearner(session=session)

    def evaluate_forgetting(
        self,
        memory: Any,
        now: Optional[datetime] = None,
        decay_days: int = DEFAULT_DECAY_DAYS,
        min_importance: int = DEFAULT_MIN_IMPORTANCE,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate if an active memory should be forgotten / archived.
        Returns: (should_forget, reason, details)
        """
        if not getattr(memory, "active", True):
            return False, "already_inactive", {"active": False}

        reference_time = now or datetime.utcnow()
        created_at = getattr(memory, "created_at", None) or reference_time
        last_accessed = getattr(memory, "last_accessed_at", None) or created_at

        # Calculate age and dormancy
        dormant_days = (reference_time - last_accessed).total_seconds() / 86400.0

        # Calculate effective learned importance
        base_imp = getattr(memory, "importance", 5) or 5
        access_cnt = getattr(memory, "access_count", 0) or 0
        help_cnt = getattr(memory, "helpful_count", 0) or 0
        reinf = getattr(memory, "reinforcement_score", 0.0) or 0.0

        effective_imp = self.learner.calculate_effective_importance(
            base_importance=base_imp,
            access_count=access_cnt,
            helpful_count=help_cnt,
            reinforcement_score=reinf,
        )

        details = {
            "memory_id": getattr(memory, "id", None),
            "dormant_days": round(dormant_days, 2),
            "effective_importance": effective_imp,
            "helpful_count": help_cnt,
            "access_count": access_cnt,
            "decay_threshold_days": decay_days,
            "min_importance_threshold": min_importance,
        }

        # Rule 1: Highly reinforced or high importance memories are NEVER forgotten automatically
        if effective_imp >= 6.0 or help_cnt >= 3:
            return False, "protected_by_high_importance_or_reinforcement", details

        # Rule 2: Low effective importance and dormant beyond threshold
        if effective_imp <= min_importance and dormant_days >= decay_days:
            return True, "low_importance_and_dormant", details

        # Rule 3: Accessed repeatedly without being helpful, and dormant
        if access_cnt >= 5 and help_cnt == 0 and dormant_days >= (decay_days / 2):
            return True, "unhelpful_repeated_access_decay", details

        return False, "retained", details

    def forget_memory(
        self,
        memory: Any,
        session=None,
        reason: str = "FORGOTTEN_CONTROLLED_DECAY",
    ) -> Tuple[Any, str, Dict[str, Any]]:
        """
        Non-destructively forget a memory by marking active=False.
        Preserves all data for potential restoration.
        """
        s = session or self.session
        memory.active = False
        now = datetime.utcnow()
        if hasattr(memory, "updated_at"):
            memory.updated_at = now

        if s is not None and hasattr(s, "commit"):
            try:
                s.commit()
            except Exception:
                pass

        metadata = {
            "memory_id": getattr(memory, "id", None),
            "active": False,
            "forgotten_at": now.isoformat(),
            "reason": reason,
        }
        return memory, "archived", metadata

    def run_forgetting_cycle(
        self,
        memories: List[Any],
        decay_days: int = DEFAULT_DECAY_DAYS,
        min_importance: int = DEFAULT_MIN_IMPORTANCE,
        session=None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Scan a list of active memories and archive those eligible for controlled forgetting.
        """
        evaluated = 0
        forgotten_records = []

        for item in memories or []:
            mem = item.get("memory") if isinstance(item, dict) else item
            if mem is None or not getattr(mem, "active", True):
                continue

            evaluated += 1
            should_forget, reason, details = self.evaluate_forgetting(
                memory=mem,
                now=now,
                decay_days=decay_days,
                min_importance=min_importance,
            )

            if should_forget:
                _, status, meta = self.forget_memory(
                    memory=mem,
                    session=session,
                    reason=reason,
                )
                forgotten_records.append({
                    "memory": mem,
                    "status": status,
                    "metadata": meta,
                    "evaluation": details,
                })

        return {
            "evaluated_count": evaluated,
            "forgotten_count": len(forgotten_records),
            "forgotten": forgotten_records,
        }
