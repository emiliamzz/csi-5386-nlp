#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import csv

import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    pipeline
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    dropout_rate: float = 0.1



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    use_jaquad : Optional[bool] = field(
        default = False,
        metadata = {
            'help' : 'Whether to use SkelterLabsInc/JaQuAD dataset (need to parse json-like data for answer columns). Default to False. '
        }
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    question_column: str = field(
        default="question",
        metadata={
            "help": "The name of the column in the datasets containing the question."
        },
    )
    context_column: str = field(
        default="context",
        metadata={
            "help": "The name of the column in the datasets containing the context."
        },
    )
    answer_column: Optional[str] = field(
        default="answer",
        metadata={
            "help": "The name of the column in the datasets containing the answer."
        },
    )
    question_prefix: Optional[str] = field(
        default="question",
        metadata={"help": "The prefix for the questions in the dataset."},
    )
    context_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "The prefix for the contexts in the dataset."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input test data file (a jsonlines or csv file)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    output_file: str = None
    score_file: str = None


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )

    model_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=model_args.use_auth_token,
        )
    else:
        data_files = {}
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_max_length=4096
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        dropout_rate=model_args.dropout_rate,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )

    generator = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, device=0
    )

    if "validation" not in datasets:
        raise ValueError("--do_predict requires a validation dataset")
    test_dataset = datasets["validation"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))

    results = defaultdict(list)
    print("evaluating...")
    for data in test_dataset:
        inputs = (
            f"{data_args.question_prefix}: {data[data_args.question_column]}" +
            f" {data_args.context_prefix}: {data[data_args.context_column]}"
        )
        pred = generator(inputs)
        results["question"].append(data[data_args.question_column])
        results["pred"].append(pred[0]["generated_text"])
        if data_args.use_jaquad:
            results["answer"].append(data[data_args.answer_column]['text'][0])
        else:
            results["answer"].append(data[data_args.answer_column])

    results = pd.DataFrame(results)
    results["em"] = results["pred"] == results["answer"]
    results["pos"] = results["answer"] != "答えなし"
    results["neg"] = results["answer"] == "答えなし"
    results["em_pos"] = (results["em"]) & (results["pos"])
    results["em_neg"] = (results["em"]) & (results["neg"])
    results.to_csv(data_args.output_file, index=False)

    em = results["em"].sum() / len(results)
    print("em : {}".format(em))
    em_pos = results["em_pos"].sum() / results["pos"].sum()
    print("em_pos : {}".format(em_pos))
    em_neg = results["em_neg"].sum() / results["neg"].sum()
    print("em_neg : {}".format(em_neg))
    false_neg = ((results["pred"] == "答えなし") & (results["pos"])).sum() / results["pos"].sum()
    print("答えのある問題のうち，答えなしと予測した割合: {}".format(false_neg))
    with open("results.txt", "a") as f:
        text = "| {model} | {dataset} "\
            "| {em:.4f} | {em_pos:.4f} | {em_neg:.4f} | {false_neg:.4f} |".format(
                model=model_args.model_name_or_path,
                dataset=data_args.dataset_name,
                em=em,
                em_pos=em_pos,
                em_neg=em_neg,
                false_neg=false_neg,
            )
        ｆ.write(text + "\n")


if __name__ == "__main__":
    main()