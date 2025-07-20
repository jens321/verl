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
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register, NaiveRewardManager


@register("elliptical")
class EllipticalRewardManager(NaiveRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", beta: int = 1.0) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        self.beta = beta

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" not in data.batch:
            # this means we're doing validation, so we don't need to compute the elliptical reward
            return super().__call__(data, return_dict=return_dict)

        reward_extra_info = defaultdict(list)

        intrinsic_reward_tensor = data.batch["rm_scores"]
        data.pop(batch_keys=["rm_scores"])

        extrinsic_reward_result = super().__call__(data, return_dict=True)
        extrinsic_reward_tensor = extrinsic_reward_result["reward_tensor"]
        extrinsic_reward_extra_info = extrinsic_reward_result["reward_extra_info"]

        reward_tensor = extrinsic_reward_tensor + self.beta * intrinsic_reward_tensor

        reward_extra_info["intrinsic_reward"] = intrinsic_reward_tensor.numpy()
        reward_extra_info["beta_scaled_intrinsic_reward"] = self.beta * intrinsic_reward_tensor.numpy()
        reward_extra_info["extrinsic_reward"] = extrinsic_reward_tensor.numpy()
        reward_extra_info["total_reward"] = reward_tensor.numpy()
        reward_extra_info.update(extrinsic_reward_extra_info)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
        
