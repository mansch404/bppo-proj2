import heapq
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class SimulationEvent:
    timestamp: float
    # case_id ist wichtig für Person D (Logger) und Person B (Routing)
    case_id: str = field(compare=False)
    activity_name: str = field(compare=False)
    # Typ des Events: 'start', 'complete', 'schedule' etc.
    event_type: str = field(compare=False)
    resource: Optional[str] = field(default=None, compare=False)
    priority: int = field(default=10, compare=False) # Falls Zeitstempel gleich sind

class EventQueue:
    def __init__(self):
        self._queue = []
        self._event_count = 0

    def push(self, event: SimulationEvent):
        """Fügt ein Event zur Warteschlange hinzu."""
        heapq.heappush(self._queue, event)
        self._event_count += 1

    def pop(self) -> Optional[SimulationEvent]:
        """Holt das nächste Event (kleinster Zeitstempel)."""
        if self.is_empty():
            return None
        return heapq.heappop(self._queue)

    def peek(self) -> Optional[SimulationEvent]:
        """Schaut sich das nächste Event an, ohne es zu entfernen."""
        if self.is_empty():
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self):
        return len(self._queue)