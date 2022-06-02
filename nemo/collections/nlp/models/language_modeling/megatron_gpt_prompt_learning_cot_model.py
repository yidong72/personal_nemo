# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import MegatronGPTPromptLearningModel
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_cot_dataset import GPTPromptLearningCOTDataset

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTPromptLearningCOTModel']


class MegatronGPTPromptLearningCOTModel(MegatronGPTPromptLearningModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = GPTPromptLearningCOTDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            for_train=for_train,
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader
