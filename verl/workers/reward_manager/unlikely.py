# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.torch_functional import masked_mean
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register, NaiveRewardManager


@register("unlikely")
class UnlikelyRewardManager(NaiveRewardManager):
    """The unlikely reward manager."""

    def __init__(
            self, 
            tokenizer, 
            num_examine, 
            compute_score=None, 
            reward_fn_key="data_source",
            beta: float = 0.25,
            turn_off_unlikely_if_all_correct: bool = False
        ) -> None:
        """
        Initialize the UnlikelyRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        self.beta = beta
        self.turn_off_unlikely_if_all_correct = turn_off_unlikely_if_all_correct

    def __call__(self, data: DataProto, return_dict=False):
        reward_extra_info = defaultdict(list)

        extrinsic_reward_result = super().__call__(data, return_dict=True)
        extrinsic_reward_tensor = extrinsic_reward_result["reward_tensor"]
        extrinsic_reward_extra_info = extrinsic_reward_result["reward_extra_info"]
        
        unlikely_reward_tensor = extrinsic_reward_tensor.clone()
        uid = data.non_tensor_batch["uid"]
        old_log_probs = data.batch["old_log_probs"]

        response_length = data.batch["responses"].shape[-1]
        eos_mask = data.batch["attention_mask"][:, -response_length:]
        old_log_probs = masked_mean(old_log_probs, eos_mask, axis=-1)

        bsz = unlikely_reward_tensor.shape[0]

        # Map uid to all rollouts with the same uid
        uid_to_indices = defaultdict(list)
        for i in range(bsz):
            uid_to_indices[uid[i]].append(i)
        
        # Process each group
        for uid, indices in uid_to_indices.items():
            # Get the rewards and log probs for this group
            group_indices = torch.tensor(indices, device=unlikely_reward_tensor.device)
            group_rewards = unlikely_reward_tensor[group_indices]
            group_old_log_probs = old_log_probs[group_indices]

            if self.turn_off_unlikely_if_all_correct and group_rewards.sum() == len(group_indices):
                continue
            
            # Apply rank penalty
            group_ranks = torch.argsort(torch.argsort(group_old_log_probs))
            group_ranks = group_ranks / len(group_indices)
            group_rewards = group_rewards * (1 - group_ranks * self.beta).unsqueeze(-1)
            
            # Update the modified_scores for these indices
            unlikely_reward_tensor[group_indices] = group_rewards

        # Intrinsic reward extra info
        reward_extra_info["extrinsic_reward"] = extrinsic_reward_tensor.numpy()
        reward_extra_info["unlikely_scaled_extrinsic_reward"] = unlikely_reward_tensor.numpy()

        # Update with extrinsic reward extra info
        reward_extra_info.update(extrinsic_reward_extra_info)

        if return_dict:
            return {
                "reward_tensor": unlikely_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return unlikely_reward_tensor