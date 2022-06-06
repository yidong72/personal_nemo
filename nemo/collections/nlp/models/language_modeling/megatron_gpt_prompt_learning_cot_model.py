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

import re
from unittest.util import _MAX_LENGTH

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_cot_dataset import (
    GPTPromptLearningCOTDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.prompt_table import VirtualPromptPlaceholderToken, VirtualPromptSource

try:
    from apex.transformer import parallel_state, tensor_parallel

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTPromptLearningCOTModel']


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)[:, None]
    return (1 - boolean) * val1 + boolean * val2


def switch_token(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


class MegatronGPTPromptLearningCOTModel(MegatronGPTPromptLearningModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.cot_id = self.tokenizer.token_to_id(VirtualPromptPlaceholderToken.COT.value)
        self.pad_id = self.tokenizer.token_to_id(VirtualPromptPlaceholderToken.PAD.value)
        # self.eos_emb = self.word_embeddings(torch.tensor([self.tokenizer.eos_id]).cuda())[0]
        self.eos_emb = None
        self.pad_token_id = self.pad_id
        self.eos_token_id = self.tokenizer.eos_id if self.tokenizer.eos_id is not None else self.tokenizer.unk_id
        self.min_tau = cfg.get('min_tau', 0.1)
        self.max_tau = cfg.get('max_tau', 1.0)
        self.frozen_model = self.frozen_model.half()

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.max_virtual_tokens = 0

        task_id_num = 0
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "cot_tokens": task.cot_tokens,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."

    def get_pseudo_tokens(self, num_virtual_tokens):
        """
        Takes in an integer and returns a list of strings where each string
        is a numbered virtual token placeholder. If 
        num_virtual_tokens = 3, then this function returns:

        ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

        Args:
            num_virtual_tokens: (int) Number of virtual token strings you want to make

        returns a list of string. 

        """
        pseudo_tokens = [
            VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
            for i in range(num_virtual_tokens)
        ]
        pseudo_tokens.append(VirtualPromptPlaceholderToken.COT.value)
        pseudo_tokens.append(VirtualPromptPlaceholderToken.PAD.value)
        return pseudo_tokens

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = GPTPromptLearningCOTDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            eos_token_id=self.eos_token_id,
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

    def validation_step(self, batch, batch_idx):
        input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids, cot_positions, answer_starts = batch
        # return torch.tensor([0]).cuda()
        with torch.no_grad():
            output, output_ids = self.forward(
                input_ids,
                position_ids,
                attention_mask,
                taskname_ids,
                labels,
                inference=False,
                cot_positions=cot_positions,
                answer_starts=answer_starts,
                loss_mask=loss_mask,
                get_tokens=True,
            )
            if batch_idx < 4:
                print(f'batch id: {batch_idx}')
                for i in range(len(output_ids)):
                    print(self.tokenizer.ids_to_text(output_ids[i]))
            output_tensor, _ = output
            loss = self.frozen_model.loss_func(loss_mask, output_tensor)
            self.log('val_loss', loss)

            return loss

    def _schedule_tau(self):
        return self.min_tau + (self.max_tau - self.min_tau) * (
            1.0 - self.trainer.current_epoch / self.trainer.max_epochs
        )

    def training_step(self, batch, batch_idx):
        input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids, cot_positions, answer_starts = batch
        output, output_ids = self.forward(
            input_ids,
            position_ids,
            attention_mask,
            taskname_ids,
            labels,
            inference=False,
            cot_positions=cot_positions,
            answer_starts=answer_starts,
            loss_mask=loss_mask,
            get_tokens=False,
        )
        output_tensor, encoder_hidden_states = output
        loss = self.frozen_model.loss_func(loss_mask, output_tensor)
        tau_value = self._schedule_tau()
        self.log('tau', tau_value)
        self.log('train_loss', loss)

        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])

        # Cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        taskname_ids,
        labels=None,
        inference=True,
        cot_positions=None,
        answer_starts=None,
        loss_mask=None,
        get_tokens=False,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        if self.eos_emb is None:
            self.eos_emb = self.word_embeddings(torch.tensor([self.tokenizer.eos_id]).cuda())[0]

        tau_value = self._schedule_tau()
        # Get embeddings for text tokens and insert virtual token embeddings
        if inference:
            input_embeds = self.embed_input_inference(input_ids, taskname_ids)
        else:
            input_embeds = self.embed_input_train(input_ids, taskname_ids)

        position_embeddings = self.frozen_model.model.language_model.embedding.position_embeddings(position_ids)
        encoder_input = input_embeds + position_embeddings

        context_length = cot_positions.min().item()
        context_lengths = cot_positions[0].clone()
        # stop index to copy over the embeddings after cot tokens
        # cot_stop_index = cot_positions[1].clone()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = self.tokenizer.eos_id
        counter = 0

        batch_size = input_ids.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        embedding = encoder_input
        # Generate enough tokens for the longest sequence
        maxlen = cot_positions.max().item()
        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        if get_tokens:
            output_tokens = input_ids.clone()
        else:
            output_tokens = None

        # while loop exit when all the batch is done with sampling
        while True:
            # types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                embedding2use = embedding[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                embedding2use = embedding[:, context_length - 1].view(batch_size, 1, -1)

            # Call forward on GPT model with preprocessed embeddings
            if self.float_type == torch.float32:
                output = self.frozen_model.model(
                    input_ids=None,
                    position_ids=None,
                    encoder_input=embedding2use,
                    attention_mask=attention_mask,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=maxlen,
                )
            else:
                with torch.autocast(device_type="cuda", dtype=self.float_type):
                    output = self.frozen_model.model(
                        input_ids=None,
                        position_ids=None,
                        encoder_input=embedding2use,
                        attention_mask=attention_mask,
                        set_inference_key_value_memory=set_inference_key_value_memory,
                        inference_max_sequence_len=maxlen,
                    )

            output = output.float()
            output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
            #
            assert output is not None
            logits = output[:, -1].view(batch_size, -1).contiguous().clone()

            # make sure it won't sample outside the vocab_size range
            logits[:, self.pseudo_token_ids_start :] = -float('Inf')

            # one_hot token
            one_hot_token = F.gumbel_softmax(logits, tau=tau_value, hard=True)
            one_hot_parallel = tensor_parallel.scatter_to_tensor_model_parallel_region(one_hot_token)
            output_parallel = torch.mm(one_hot_parallel.half(), self.word_embeddings.weight)
            prev = tensor_parallel.reduce_from_tensor_model_parallel_region(output_parallel).clone()
            # prev = torch.mm(one_hot_token, self.word_embeddings.weight)
            # apply the positional embedding
            prev += position_embeddings[:, context_length]

            # the flag that it is going beyond the context
            started = context_lengths <= context_length

            # index = repeat(cot_stop_index, 'k -> k 1 r', r=embedding.shape[-1])

            # copied_embed = torch.gather(embedding, 1, index).squeeze(1)

            new_emb = switch(embedding[:, context_length], prev, started)

            # Replace sampled tokens w/ done token if EOD has already been sampled
            # TODO, need to put the rest of the token_embeddings here
            #            new_emb = switch(new_emb, copied_embed, is_done)

            # Insert either new predicted or next prompt token
            # only update the not done embedding
            embedding[~is_done.bool(), context_length] = new_emb[~is_done.bool()]

            max_tokens = torch.argmax(one_hot_token, axis=-1)
            max_tokens = switch_token(input_ids[:, context_length], max_tokens, started)

            if get_tokens:
                output_tokens[~is_done.bool(), context_length] = max_tokens[~is_done.bool()]

            # when sampling the eod token and started and not done yet,  or reach the cot_end position
            done_token = ((max_tokens == eod_id).byte() & started.byte()) | (context_length >= cot_positions[1]).byte()
            #             if context_length == maxlen:
            #                 # all done
            #                 break
            #
            # first time finished?
            just_finished = (done_token & ~is_done).bool()

            # if any batch just finished, copy over the things after cot
            if just_finished.any():
                end = embedding.shape[1]
                for i in range(batch_size):
                    if just_finished[i]:
                        start = cot_positions[1, i]
                        embedding[i, context_length : end - start + context_length] = embedding[i, start:end].clone()
                        # add eos in the end
                        embedding[i, end - start + context_length :] = self.eos_emb[None, :]
                        # adjust labels
                        labels[i, context_length : end - start + context_length] = labels[i, start:end].clone()
                        # adjust loss _mask
                        loss_mask[i, context_length : end - start + context_length] = loss_mask[i, start:end].clone()
                        loss_mask[i, end - start + context_length :] = 0
                        # adjust output tokens
                        if get_tokens:
                            output_tokens[i, context_length : end - start + context_length] = input_ids[
                                i, start:end
                            ].clone()
                            output_tokens[i, end - start + context_length :] = self.eos_token_id

            # if it finishes early due ot eod_id sample, set the cot_stop early
            # cot_stop_index[just_finished] = cot_positions[1][just_finished]
            # set the context stopping position
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token

            done = torch.all(is_done)
            context_length += 1

            counter += 1
            if done:
                break

        # last call to compute the loss
        if self.float_type == torch.float32:
            output = self.frozen_model.model(
                input_ids=None,
                position_ids=None,
                encoder_input=embedding,
                labels=labels,
                attention_mask=attention_mask,
                set_inference_key_value_memory=False,
                inference_max_sequence_len=None,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                output = self.frozen_model.model(
                    input_ids=None,
                    position_ids=None,
                    encoder_input=embedding,
                    labels=labels,
                    attention_mask=attention_mask,
                    set_inference_key_value_memory=False,
                    inference_max_sequence_len=None,
                )

        return output, output_tokens

    def embed_input_train(self, input_ids: Tensor, taskname_ids: Tensor):
        """
        Replaces the virtual tokens in the input_ids with embeddings 
        calculated from either the 'prompt_table' or 'prompt_encoder'. 
        The virtual token placeholders have token_ids listed in
        `self.pseudo_token_ids`.

        params:
            input_ids: the input token ids
            taskname_ids: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids >= self.pseudo_token_ids_start)] = self.pad_token_id
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()

        # Find the indicies where virtual tokens should be inserted
        virtual_token_locations = (input_ids >= self.pseudo_token_ids_start) & (input_ids < self.cot_id)

        # If there are no virtual tokens, just return discrete token embeds
        if not virtual_token_locations.any():
            return discrete_token_embeds

        # Get virtual token embeddings from the prompt table or prompt encoder
        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
            virtual_token_embeds = [self.prompt_table(task_id_num) for task_id_num in taskname_ids]
            virtual_token_embeds = torch.stack(virtual_token_embeds)

        elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            taskname_embeddings = self.word_embeddings(taskname_ids)
            virtual_token_embeds = self.prompt_encoder(taskname_embeddings=taskname_embeddings)

        # Create index template specifying where virtual token embeddings should be placed
        batch_size, _, embedding_size = discrete_token_embeds.shape
        virtual_token_index = virtual_token_locations.nonzero().reshape((batch_size, -1, 2))[:, :, 1][:, :, None]
        virtual_token_index = virtual_token_index.expand(
            batch_size, self.total_new_task_virtual_tokens, embedding_size
        )

        # Make sure discrete_token_embeds and virtual_token_embeds share the same dtype
        discrete_token_embeds = discrete_token_embeds.type(virtual_token_embeds.dtype)

        # Insert virtual token embeddings where they belong amoung the discrete token embeddings
        discrete_token_embeds.scatter_(1, virtual_token_index, virtual_token_embeds)
        input_embeds = discrete_token_embeds

        return input_embeds
