# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN
#  THE SOFTWARE.

import time
import datetime
import sys
import asyncio
import numpy as np
import bittensor as bt
from prompting.metrics_schema import MetricsSchema
from prompting.prometheus_metrics import update_metrics_for_miner
from prompting.agent import HumanAgent
from prompting.dendrite import DendriteResponseEvent
from prompting.conversation import create_task
from prompting.protocol import PromptingSynapse
from prompting.rewards import RewardResult
from prompting.utils.uids import get_random_uids
from prompting.utils.logging import log_event
from prompting.utils.misc import async_log

@async_log
async def generate_reference(agent):    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, agent.task.generate_reference, agent.llm_pipeline)
    return result    

@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses

def word_count(text):
    return len(text.split())

miner_metrics_dict = {} 

def calculate_miner_metrics(response_event, agent, reward_result):
    global miner_metrics_dict

    for uid, response, status_code, timings in zip(response_event.uids, response_event.completions, response_event.status_codes, response_event.timings):
        uid_str = str(uid.item())
        
        # Retrieve metrics
        response_wc = word_count(response)
        reference_wc = word_count(agent.task.reference)
        challenge_wc = word_count(agent.challenge)
        challenge_time = agent.challenge_time
        reference_time = agent.task.reference_time
        step_time = response_event.step_time
        
        # Retrieve rewards and scores from RewardEvent
        reward = reward_result.rewards[response_event.uids == uid].item() 
        rouge_score = sum(event.rewards.tolist() for event in reward_result.reward_events if event.model_name == "rouge")
        relevance_score = sum(event.rewards.tolist() for event in reward_result.reward_events if event.model_name == "relevance")

        # Initialize or update metrics for each miner
        if uid_str not in miner_metrics_dict:
            miner_metrics_dict[uid_str] = MetricsSchema(miner_uid=uid_str)
        miner_metrics = miner_metrics_dict[uid_str]

        # Assign metrics for current run
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

        # Update Prometheus metrics
        update_metrics_for_miner(uid_str, miner_metrics)
        bt.logging.info(f"Updated metrics for miner UID: {uid_str}")

    return list(miner_metrics_dict.values())



async def run_step(
    self, agent: HumanAgent, k: int, timeout: float, exclude: list = None
):
    """Executes a single step of the agent, which consists of:
    - Getting a list of uids to query
    - Querying the network
    - Rewarding the network
    - Updating the scores
    - Logging the event

    Args:
        agent (HumanAgent): The agent to run the step for.
        k (int): The number of uids to query.
        timeout (float): The timeout for the queries.
        exclude (list, optional): The list of uids to exclude from the query. Defaults to [].
    """

    bt.logging.debug("run_step", agent.task.name)

    # Record event start time.
    start_time = time.time()
    # Get the list of uids to query for this step.
    uids = get_random_uids(self, k=k, exclude=exclude or []).to(self.device)
    axons = [self.metagraph.axons[uid] for uid in uids]

    # Prepare the tasks
    dendrite_call_task = execute_dendrite_call(self.dendrite(axons=axons, synapse=PromptingSynapse(roles=["user"], messages=[agent.challenge]), timeout=timeout))
    
    if not agent.task.static_reference:            
        reference_generation_task = generate_reference(agent)
        _, responses = await asyncio.gather(reference_generation_task, dendrite_call_task)
    else:
        responses = await dendrite_call_task    
            
    # Encapsulate the responses in a response event (dataclass)
    response_event = DendriteResponseEvent(responses, uids)
    # Calculate and store the word count of the reference and responses
    bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}")
    # Reward the responses and get the reward result (dataclass)
    # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
    reward_result = RewardResult(
        self.reward_pipeline,
        agent=agent,
        response_event=response_event,
        device=self.device,
    )
    bt.logging.info(f"Created RewardResult:\n {reward_result}")
    
    # The original idea was that the agent is 'satisfied' when it gets a good enough response (e.g. reward critera is met, such as ROUGE>threshold)
    agent.update_progress(
        top_reward=reward_result.rewards.max(),
        top_response=response_event.completions[reward_result.rewards.argmax()],
    )

    self.update_scores(reward_result.rewards, uids)

    # Calculate metrics for each miner
    uid_response_pairs = calculate_miner_metrics(response_event, agent, reward_result)
    
    
    # Log the step event.
    event = {
        "block": self.block,
        "step_time": time.time() - start_time,
        "timestamp": datetime.datetime.now().isoformat(),
        "uid_response_pairs": uid_response_pairs,
        **agent.__state_dict__(full=self.config.neuron.log_full),
        **reward_result.__state_dict__(full=self.config.neuron.log_full),
        **response_event.__state_dict__(),
    }

    return event


async def forward(self):
    bt.logging.info("🚀 Starting forward loop...")
    forward_start_time = time.time()

    while True:
        bt.logging.info(
            f"📋 Selecting task... from {self.config.neuron.tasks} with distribution {self.config.neuron.task_p}"
        )
        # Create a specific task
        task_name = np.random.choice(
            self.config.neuron.tasks, p=self.config.neuron.task_p
        )
        bt.logging.info(f"📋 Creating {task_name} task... ")
        try:
            task = create_task(llm_pipeline=self.llm_pipeline, task_name=task_name, create_reference=False)
            break
        except Exception as e:
            bt.logging.error(
                f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task."
            )
            continue

    # Create random agent with task, topic, profile...
    bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
    agent = HumanAgent(
        task=task, llm_pipeline=self.llm_pipeline, begin_conversation=True
    )

    rounds = 0
    exclude_uids = []
    while not agent.finished:
        # when run_step is called, the agent updates its progress
        event = await run_step(
            self,
            agent,
            k=self.config.neuron.sample_size,
            timeout=self.config.neuron.timeout,
            exclude=exclude_uids,
        )
        
        # Adds forward time to event and logs it to wandb
        event['forward_time'] = time.time() - forward_start_time        
        log_event(self, event)
        
        exclude_uids += event["uids"]
        task.complete = True

        rounds += 1

    del agent
    del task
