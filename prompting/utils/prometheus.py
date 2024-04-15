from prometheus_client import Gauge, Histogram
import bittensor as bt

class MinerMetrics:
    # Definition of Prometheus metrics with their types
    metrics = {
        'reward': Gauge('reward', 'Reward for each response', ['miner_uid']),
        'rouge': Gauge('rouge', 'Rouge score for each response', ['miner_uid']),
        'relevance': Gauge('relevance', 'Relevance score for each response', ['miner_uid']),
        'reference_word_count': Gauge('reference_word_count', 'Word count of reference response', ['miner_uid']),
        'response_word_count': Gauge('response_word_count', 'Word count of miner response', ['miner_uid']),
        'challenge_word_count': Gauge('challenge_word_count', 'Word count of challenge', ['miner_uid']),
        'availability': Gauge('availability', 'Availability of miner', ['miner_uid']),
        'response_time': Histogram('response_time', 'Response time for each response', ['miner_uid'])
    }

    @staticmethod
    def update_metrics_for_miner(uid, **metrics):
        for key, value in metrics.items():
            metric = MinerMetrics.metrics.get(key)
            if metric:
                if isinstance(metric, Gauge):
                    metric.labels(miner_uid=uid).set(value)
                elif isinstance(metric, Histogram):
                    metric.labels(miner_uid=uid).observe(value)
            bt.logging.debug(f"Updated Prometheus metric '{key}' for miner UID: {uid}")
        print(f"DEBUG: Updated Prometheus metrics for miner UID: {uid}")
