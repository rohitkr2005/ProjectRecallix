from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any

from sqlalchemy import select
from app.database.models import Memory, MemoryEdge


class MemoryGraph:
    """
    10.6 Memory Relationships
    10.7 Memory Graph

    Constructs and queries an entity-relation knowledge graph over stored memories
    and explicit relationship edges.
    """

    def __init__(self, session=None):
        self.session = session
        # Adjacency lists: node -> list of edge dicts
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.edges = []
        self._synced = False

    def clear(self):
        self.outgoing.clear( )
        self.incoming.clear()
        self.edges.clear()
        self._synced = False

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        category: Optional[str] = None,
        weight: float = 1.0,
        memory_id: Optional[int] = None,
        persist: bool = False,
    ):
        """Add a directed relationship edge: source -[relation]-> target."""
        if not source or not target or not relation:
            return

        s = str(source).strip()
        t = str(target).strip()
        r = str(relation).strip()

        edge = {
            "source": s,
            "relation": r,
            "target": t,
            "category": category,
            "weight": float(weight),
            "memory_id": memory_id,
        }

        self.edges.append(edge)
        self.outgoing[s.lower()].append(edge)
        self.incoming[t.lower()].append(edge)

        if persist and self.session is not None:
            db_edge = MemoryEdge(
                source_entity=s,
                relation=r,
                target_entity=t,
                category=category,
                weight=weight,
            )
            self.session.add(db_edge)
            try:
                self.session.commit()
            except Exception:
                pass

    def build_from_memories(self, memories: List[Any]):
        """Populate the graph from a list of Memory models."""
        for m in memories or []:
            mem = m.get("memory") if isinstance(m, dict) else m
            if mem is None or not getattr(mem, "active", True):
                continue

            sub = getattr(mem, "subject", "User")
            rel = getattr(mem, "relation", "")
            val = getattr(mem, "value", "")
            cat = getattr(mem, "category", "")
            mid = getattr(mem, "id", None)

            if sub and rel and val:
                self.add_edge(
                    source=sub,
                    relation=rel,
                    target=val,
                    category=cat,
                    weight=1.0,
                    memory_id=mid,
                )

    def sync_from_database(self, session=None):
        """Synchronize the graph with active memories and stored edges in DB."""
        s = session or self.session
        if s is None:
            return

        self.clear()

        # 1. Load active memories
        stmt = select(Memory).where(Memory.active.is_(True))
        try:
            active_mems = s.execute(stmt).scalars().all()
            self.build_from_memories(active_mems)
        except Exception:
            pass

        # 2. Load explicit edges
        edge_stmt = select(MemoryEdge)
        try:
            db_edges = s.execute(edge_stmt).scalars().all()
            for de in db_edges:
                self.add_edge(
                    source=de.source_entity,
                    relation=de.relation,
                    target=de.target_entity,
                    category=de.category,
                    weight=de.weight or 1.0,
                    persist=False,
                )
        except Exception:
            pass

        self._synced = True

    def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        relation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get adjacent edges for an entity.
        direction: 'out', 'in', or 'both'.
        """
        if not entity:
            return []

        ent_lower = str(entity).strip().lower()
        results = []

        if direction in ("out", "both"):
            for e in self.outgoing.get(ent_lower, []):
                if relation is None or e["relation"].lower() == relation.lower():
                    results.append(e)

        if direction in ("in", "both"):
            for e in self.incoming.get(ent_lower, []):
                if relation is None or e["relation"].lower() == relation.lower():
                    results.append(e)

        return results

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """Find all directed/undirected paths between source and target up to max_depth."""
        s_lower = str(source).strip().lower()
        t_lower = str(target).strip().lower()

        if s_lower == t_lower:
            return []

        paths = []
        queue = deque([([(s_lower, None)], set([s_lower]))])  # list of (node, edge), visited

        while queue:
            current_path, visited = queue.popleft()
            curr_node, _ = current_path[-1]

            if len(current_path) - 1 >= max_depth:
                continue

            # Check all neighbors
            neighbors = self.get_neighbors(curr_node, direction="both")
            for edge in neighbors:
                next_node = edge["target"].lower() if edge["source"].lower() == curr_node else edge["source"].lower()
                if next_node == t_lower:
                    full_edges = [p[1] for p in current_path[1:]] + [edge]
                    paths.append(full_edges)
                elif next_node not in visited and len(current_path) < max_depth:
                    new_visited = set(visited)
                    new_visited.add(next_node)
                    new_path = current_path + [(next_node, edge)]
                    queue.append((new_path, new_visited))

        return paths

    def calculate_proximity(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
    ) -> float:
        """
        Calculate graph proximity between two entities in [0.0, 1.0].
        1 hop -> 1.0
        2 hops -> 0.67
        3 hops -> 0.33
        disconnected -> 0.0
        """
        if not entity_a or not entity_b:
            return 0.0

        if str(entity_a).strip().lower() == str(entity_b).strip().lower():
            return 1.0

        paths = self.find_paths(entity_a, entity_b, max_depth=max_depth)
        if not paths:
            return 0.0

        shortest = min(len(p) for p in paths)
        if shortest == 1:
            return 1.0
        elif shortest == 2:
            return 0.67
        elif shortest == 3:
            return 0.33
        return 0.0

    def get_connected_entities(
        self,
        seed_entity: str,
        max_depth: int = 2,
    ) -> Set[str]:
        """Return all entity names reachable from seed_entity within max_depth hops."""
        if not seed_entity:
            return set()

        seed_lower = str(seed_entity).strip().lower()
        visited = set([seed_lower])
        queue = deque([(seed_lower, 0)])

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge in self.get_neighbors(curr, direction="both"):
                neighbor = edge["target"].lower() if edge["source"].lower() == curr else edge["source"].lower()
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return visited
