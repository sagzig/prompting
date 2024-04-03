import json
import os
import copy
import wandb
from wandb.apis import public
import bittensor as bt
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List
from loguru import logger
import prompting


@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: List[str]
    responses: List[str]
    miners_time: List[float]
    challenge_time: float
    reference_time: float
    rewards: List[float]
    task: dict
    # extra_info: dict


def export_logs(logs: List[Log]):
    bt.logging.info("📝 Exporting logs...")

    # Create logs folder if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the current date and time for logging purposes
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_logs_dict = [asdict(log) for log in logs]

    for logs in all_logs_dict:
        task_dict = logs.pop("task")
        prefixed_task_dict = {f"task_{k}": v for k, v in task_dict.items()}
        logs.update(prefixed_task_dict)

    log_file = f"./logs/{date_string}_output.json"
    with open(log_file, "w") as file:
        json.dump(all_logs_dict, file)

    return log_file


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return (
        not self.config.wandb.off
        and self.step
        and self.step % self.config.wandb.run_step_length == 0
    )


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [
        self.wallet.hotkey.ss58_address,
        prompting.__version__,
        str(prompting.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")
    for task in self.active_tasks:
        tags.append(task)
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def log_event(self, event):

    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    if self.config.wandb.off:
        return

    if not getattr(self, "wandb", None):
        init_wandb(self)

    # Log the event to wandb.
    self.wandb.log(event)
    
    table_data = []
    for uid, metrics in event["uid_response_pairs"].items():
        row = [
            uid,
            # Include all the required metrics
            metrics["avg_reward"],
            metrics["median_reward"],
            metrics["std_dev_reward"],
            metrics["average_rouge"],
            metrics["median_rouge"],
            metrics["std_dev_rouge"],
            metrics["average_relevance"],
            metrics["median_relevance"],
            metrics["std_dev_relevance"],
            metrics["reference_word_count"],
            metrics["response_word_count"],
            metrics["availability"],
            metrics["average_response_time"],
        ]
        table_data.append(row)

    columns = [
        "UID", 
        "Avg Reward", 
        "Median Reward", 
        "Std Dev Reward", 
        "Avg Rouge", 
        "Median Rouge", 
        "Std Dev Rouge",
        "Avg Relevance", 
        "Median Relevance", 
        "Std Dev Relevance",
        "Reference Word Count",
        "Response Word Count",
        "Availability",
        "Avg Response Time"
    ]

    wandb_table = wandb.Table(columns=columns, data=table_data)
    self.wandb.log({"Miners/Metrics": wandb_table})

