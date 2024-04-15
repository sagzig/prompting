import bittensor as bt
from prompting.utils.metrics_schema import MetricsSchema
from prompting.utils.prometheus import MinerMetrics


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


        # Retrieve or create a new metric instance
        miner_metrics = miner_metrics_dict.get(uid_str, MetricsSchema(miner_uid=uid_str))

        # Update metrics with the new values
        miner_metrics.update_metrics(
            reward=reward,
            rouge_score=rouge_score,
            relevance_score=relevance_score,
            response_wc=response_wc,
            reference_wc=reference_wc,
            challenge_wc=challenge_wc,
            status_code=status_code,
            timings=timings,
            challenge_time=challenge_time,
            reference_time=reference_time,
            step_time=step_time
        )

        # Store back in the dictionary
        miner_metrics_dict[uid_str] = miner_metrics

        bt.logging.debug(f"CalcMetrics for UID {uid_str}: {miner_metrics}")

        # Update Prometheus metrics
        MinerMetrics.update_metrics_for_miner(uid_str, miner_metrics)

    return list(miner_metrics_dict.values())