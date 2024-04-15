import bittensor as bt
import datetime
from prompting.utils.metrics_schema import MetricsSchema
from prompting.utils.prometheus import update_metrics_for_miner


def word_count(text):
    return len(text.split())

miner_metrics_dict = {} 
def calculate_miner_metrics(response_event, agent, reward_result):
    global miner_metrics_dict

    for uid, response, status_code, timings in zip(response_event.uids, response_event.completions, response_event.status_codes, response_event.timings):
        uid_str = str(uid.item())
        
        # Retrieve metrics
        response_wc = word_count(response)
        reference_wc = word_count(agent.task.reference) if hasattr(agent.task, 'reference') else 0
        challenge_wc = word_count(agent.challenge)
        challenge_time = getattr(agent, 'challenge_time', 0)
        reference_time = getattr(agent.task, 'reference_time', 0)
        step_time = getattr(response_event, 'step_time', 0)
        
        # Initialize or reset the scores
        rouge_score = 0
        relevance_score = 0

        # Fetch the rewards and scores for each UID
        reward = reward_result.rewards[response_event.uids == uid].item()
        for event in reward_result.reward_events:
            if uid in response_event.uids:
                index = (response_event.uids == uid).nonzero(as_tuple=True)[0].item()
                if event.model_name == "rouge":
                    rouge_score = event.rewards[index].item()
                elif event.model_name == "relevance":
                    relevance_score = event.rewards[index].item()


        # Create metrics
        miner_metrics = miner_metrics_dict.get(uid_str, MetricsSchema(
            miner_uid=uid_str,
            timestamp=datetime.datetime.now().isoformat(),
            step_time=0,
            challenge_time=0,
            reference_time=0,
            reward=0,
            rouge=0,
            relevance=0,
            reference_word_count=0,
            response_word_count=0,
            challenge_word_count=0,
            availability=0,
            response_time=0
        ))

        # Assign metrics for the current run
        miner_metrics.reward = reward
        miner_metrics.rouge = rouge_score
        miner_metrics.relevance = relevance_score
        miner_metrics.reference_word_count = reference_wc
        miner_metrics.response_word_count = response_wc
        miner_metrics.challenge_word_count = challenge_wc
        miner_metrics.availability = 1 if status_code not in [408, 503, 403] else 0
        miner_metrics.response_time = timings
        miner_metrics.challenge_time = challenge_time
        miner_metrics.reference_time = reference_time
        miner_metrics.step_time = step_time

        # Store back in the dictionary
        miner_metrics_dict[uid_str] = miner_metrics

        bt.logging.debug(f"CalcMetrics for UID {uid_str}: {miner_metrics}")

        # Update Prometheus metrics
        update_metrics_for_miner(uid_str, miner_metrics)

    return list(miner_metrics_dict.values())