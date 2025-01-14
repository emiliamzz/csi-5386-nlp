# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datasets
from datasets import disable_caching
disable_caching()

from datasets.tasks import QuestionAnsweringExtractive

from typing import List

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
No Description.
"""

# val_passage_dataset_json_path = \
#     '/inputDataset/input.jsonl'
train_passage_dataset_json_path = \
    '/workspace/data/train.jsonl'

class SquadConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class Squad(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SquadConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            # citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        return [
            # datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_passage_dataset_json_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_dir}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                # load clickbait jsonl file and convert it to squad-like dataset
                data = json.loads(line)
                id_ = data['uuid']
                title = data['targetTitle']
                question = ' '.join(data['postText'])
                context = data['targetTitle'] + ' - ' + (' '.join(data['targetParagraphs']))

                # answer_starts = [start_offset + self.calculate_position_in_context(positions_list[0],paragraph_length) for positions_list in data['spoilerPositions']] # taking only start position
                # answer = ' '.join(data['spoiler'])
                # Features currently used are "context", "question", and "answers".
                # Others are extracted here for the ease of future expansions.
                yield id_, {
                    "title": title,
                    "context": context,
                    "question": question,
                    "id": id_,
                    #dammy answer
                    'answer': ''
                }
    
    @staticmethod
    def calculate_position_in_context(postions : List[int],paragraph_length: List[int])->int:
        paragraph_num ,position_in_paragraph = postions
        position = paragraph_num + sum(paragraph_length[:paragraph_num]) + position_in_paragraph

        return position