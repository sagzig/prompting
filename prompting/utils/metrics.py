import bittensor as bt
from prompting.utils.metrics_schema import MetricsSchema
from prompting.utils.prometheus import MinerMetrics


def word_count(text):
    return len(text.split())

miner_metrics_dict = {} 
async def calculate_miner_metrics(response_event, agent, reward_result):
    global miner_metrics_dict

    for uid, response, status_code, timings in zip(response_event.uids, response_event.completions, response_event.status_codes, response_event.timings):
        uid_str = str(uid.item())

        # Check for an existing entry or create a new one
        if uid_str not in miner_metrics_dict:
            miner_metrics_dict[uid_str] = MetricsSchema(miner_uid=uid_str)

        # Gather metrics data
        metrics_data = {
            'reward': reward_result.rewards[response_event.uids == uid].item(),
            'response_wc': word_count(response),
            'reference_wc': word_count(agent.task.reference) if hasattr(agent.task, 'reference') else 0,
            'challenge_wc': word_count(agent.challenge),
            'status_code': status_code,
            'timings': timings,
            'challenge_time': getattr(agent, 'challenge_time', 0),
            'reference_time': getattr(agent.task, 'reference_time', 0),
            'step_time': getattr(response_event, 'step_time', 0)
        }

        # Fetch the rewards and scores for each UID
        for event in reward_result.reward_events:
            if uid in response_event.uids:
                index = (response_event.uids == uid).nonzero(as_tuple=True)[0].item()
                metrics_data['rouge_score'] = event.rewards[index].item() if event.model_name == "rouge" else 0
                metrics_data['relevance_score'] = event.rewards[index].item() if event.model_name == "relevance" else 0

        # Update MetricsSchema instance
        miner_metrics_dict[uid_str].update_metrics(**metrics_data)

        # Log and update Prometheus
        bt.logging.debug(f"CalcMetrics for UID {uid_str}: {miner_metrics_dict[uid_str]}")
        MinerMetrics.update_metrics_for_miner(uid_str, miner_metrics_dict[uid_str].__dict__)
        bt.logging.info(f"Updated metrics for miner UID: {uid_str}")

    return list(miner_metrics_dict.values())
