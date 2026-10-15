"""Public search-result contract shared by MCTS consumers."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One completed root search without encoding decisions in visit counts."""

    selected_move: Any | None
    visits: Mapping[Any, float]
    completed_q: Mapping[Any, float]
    improved_policy: Mapping[Any, float]
    selection_score: Mapping[Any, float]
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    training_policy: Mapping[Any, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.selected_move is not None and bool(self.visits)

    @classmethod
    def from_legacy_result(
        cls,
        visits: Mapping[Any, float] | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SearchResult":
        """Build the public result from the stable ``search_many`` payload."""
        visit_map = dict(visits or {})
        metadata_map = dict(metadata or {})
        selected_move = metadata_map.get("selected_move_override")
        return cls(
            selected_move=selected_move,
            visits=visit_map,
            completed_q=dict(metadata_map.get("completed_q_by_move") or {}),
            improved_policy=dict(metadata_map.get("improved_policy_by_move") or {}),
            selection_score=dict(metadata_map.get("selection_score_by_move") or {}),
            metadata=metadata_map,
            training_policy=dict(
                metadata_map.get("policy_target_by_move")
                or metadata_map.get("policy_target_probs_override")
                or {}
            ),
        )
