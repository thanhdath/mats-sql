#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
import torch.nn as nn
from trl import ORPOConfig, ORPOTrainer
from peft import PeftConfig, PeftModel
from trl import DPOTrainer, create_reference_model
import random
from trl import DataCollatorForCompletionOnlyLM
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ORPOTrainerForCompletionOnly(ORPOTrainer):
    # def odds_ratio_loss(
    #     self,
    #     policy_chosen_logps: torch.FloatTensor,
    #     policy_rejected_logps: torch.FloatTensor,
    #     temperature: float = 0.1  # <-- Add temperature scaling parameter
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Compute ORPO's odds ratio (OR) loss with temperature scaling."""

    #     # Apply temperature scaling to log probabilities
    #     policy_chosen_logps = policy_chosen_logps / temperature  # <-- Scale log probabilities
    #     policy_rejected_logps = policy_rejected_logps / temperature  # <-- Scale log probabilities

    #     # Compute log odds ratio
    #     log_odds = (policy_chosen_logps - policy_rejected_logps) - (
    #         torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
    #     )

    #     sig_ratio = F.sigmoid(log_odds)
    #     ratio = torch.log(sig_ratio)
    #     losses = self.beta * ratio

    #     chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
    #     rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

    #     return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "decoder_input_ids": self._shift_right(concatenated_batch["concatenated_labels"]),
            }
            if self.is_encoder_decoder
            else {}
        )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()

        # import pdb; pdb.set_trace()
        # chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        """
        I FIXED HERE
        """
        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], concatenated_batch['concatenated_labels'][:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, ORPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

        # Replace '<|start_header_id|>user<|end_header_id|>\n' with ''
        # raw_datasets[split] = raw_datasets[split].map(
        #     lambda examples: {
        #         key: examples[key].replace('<|start_header_id|>user<|end_header_id|>\n', '').replace('<|start_header_id|>assistant<|end_header_id|>', '')
        #         if key in ["prompt", "chosen", "rejected"] else examples[key]
        #         for key in examples
        #     }
        # )

        # # Replace '<|start_header_id|>assistant<|end_header_id|>\n' with ''
        # raw_datasets[split] = raw_datasets[split].map(
        #     lambda examples: {
        #         key: examples[key].replace('<|start_header_id|>assistant<|end_header_id|>\n', '').replace('<|end|>', '<|eot_id|>')
        #         if key in ["prompt", "chosen", "rejected"] else examples[key]
        #         for key in examples
        #     }
        # )

        # # Replace '<|eot_id|>\n' in prompt with ''
        # raw_datasets[split] = raw_datasets[split].map(
        #     lambda examples: {
        #         "prompt": examples["prompt"].replace('<|eot_id|>\n', '<|reserved_special_token_247|>').replace('<|end|>', '<|eot_id|>'),
        #         **{key: value for key, value in examples.items() if key != "prompt"}
        #     }
        # )
        # raw_datasets[split] = raw_datasets[split].map(
        #     lambda examples: {
        #         "chosen": examples["chosen"].strip(),
        #         **{key: value for key, value in examples.items() if key != "chosen"}
        #     }
        # )
        # raw_datasets[split] = raw_datasets[split].map(
        #     lambda examples: {
        #         "rejected": examples["rejected"].strip(),
        #         **{key: value for key, value in examples.items() if key != "rejected"}
        #     }
        # )


        # Optionally, you can add any additional processing here
        print(f"Processed {split} dataset with {len(raw_datasets[split])} entries.")



    for index in random.sample(range(len(raw_datasets["train"])), 3):
        # logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt'] + raw_datasets['train'][index]['chosen']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt'] + raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{ raw_datasets['train'][index]['prompt'] +raw_datasets['train'][index]['rejected']}")


    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # model = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        )
    
    if tokenizer.pad_token == tokenizer.eos_token:
        print('add Pad token')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.pad_token = tokenizer.pad_token
    model.resize_token_embeddings(len(tokenizer))

    # if model_args.response_template is not None:
    collator = DataCollatorForCompletionOnlyLM(
        response_template=model_args.response_template,
        tokenizer=tokenizer, 
        mlm=False)

    ########################
    # Instantiate ORPO trainer
    #########################

    dpo_trainer = ORPOTrainerForCompletionOnly(
        model,
        # data_collator=collator,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args)
    )

    ###############
    # Training loop
    ###############
    train_result = dpo_trainer.train(resume_from_checkpoint=True)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = dpo_trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    dpo_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub is True:
            dpo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
