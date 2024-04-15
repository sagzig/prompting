
from prometheus_client import Gauge, Summary, Histogram, Enum
import bittensor as bt

# Defining the Type for each metric
reward_gauge = Gauge('reward', 'Reward for each response', ['miner_uid'])
rouge_gauge = Gauge('rouge', 'Rouge score for each response', ['miner_uid'])
relevance_gauge = Gauge('relevance', 'Relevance score for each response', ['miner_uid'])
reference_word_count_gauge = Gauge('reference_word_count', 'Word count of reference response', ['miner_uid'])
response_word_count_gauge = Gauge('response_word_count', 'Word count of miner response', ['miner_uid'])
challenge_word_count_gauge = Gauge('challenge_word_count', 'Word count of challenge', ['miner_uid'])
availability_gauge = Gauge('availability', 'Availability of miner', ['miner_uid'])
response_time_histogram = Histogram('response_time', 'Response time for each response', ['miner_uid'])

def update_metrics_for_miner(uid, miner_metrics):
    reward_gauge.labels(miner_uid=uid).set(miner_metrics.reward)
    rouge_gauge.labels(miner_uid=uid).set(miner_metrics.rouge)
    relevance_gauge.labels(miner_uid=uid).set(miner_metrics.relevance)
    reference_word_count_gauge.labels(miner_uid=uid).set(miner_metrics.reference_word_count)
    response_word_count_gauge.labels(miner_uid=uid).set(miner_metrics.response_word_count)
    challenge_word_count_gauge.labels(miner_uid=uid).set(miner_metrics.challenge_word_count)
    availability_gauge.labels(miner_uid=uid).set(miner_metrics.availability)
    response_time_histogram.labels(miner_uid=uid).observe(miner_metrics.response_time)

    bt.logging.debug(f"Updating Prometheus metrics for miner UID: {uid}")
    print(f"DEBUG: Updating Prometheus metrics for miner UID: {uid}") 
