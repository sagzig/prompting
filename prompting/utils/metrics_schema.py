
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
    def update_metrics(self, reward, rouge_score, relevance_score, response_word_count, reference_word_count, challenge_word_count, status_code, timings, challenge_time, reference_time, step_time):
        self.reward = reward
        self.rouge = rouge_score
        self.relevance = relevance_score
        self.response_word_count = response_word_count
        self.reference_word_count = reference_word_count
        self.challenge_word_count = challenge_word_count
        self.availability = status_code
        self.response_time = timings
        self.challenge_time = challenge_time
        self.reference_time = reference_time
        self.step_time = step_time
