
from dataclasses import dataclass, field
import datetime
@dataclass
class MetricsSchema:
    miner_uid: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    step_time: float = 0.0
    challenge_time: float = 0.0
    reference_time: float = 0.0
    reward: float = 0.0
    rouge: float = 0.0
    relevance: float = 0.0
    reference_word_count: int = 0
    response_word_count: int = 0
    challenge_word_count: int = 0
    availability: int = 0
    response_time: float = 0.0

    def update_metrics(self, **updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Metric '{key}' not found in MetricsSchema")
