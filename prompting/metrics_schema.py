from dataclasses import dataclass

@dataclass
class MetricsSchema:
    miner_uid: str
    timestamp: str
    step_time: float
    challenge_time: float
    reference_time: float
    reward: float
    rouge: float
    relevance: float
    reference_word_count: int
    response_word_count: int
    challenge_word_count: int
    availability: int
    response_time: float
