# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
Metric for evaluating the performance of seq-to-seq model on QA
"""

import re
import unicodedata

import datasets

_CITATION = """
"""

_DESCRIPTION = """\
This metric is designed for QA.
"""

_KWARGS_DESCRIPTION = """
Calculates some scores (corrently only exact match)
Args:
    predictions: list of predictions to score.
    references: list of reference for each prediction.
Returns:
    exact_match: exact match between label and prediction.
Examples:
    >>> metric_qa = datasets.load_metric("qa")
    >>> results = metric_qa.compute(references=["林檎", "バナナ"], predictions=["林檎", "イチゴ"])
    >>> print(results)
    {'exact_match': 0.5}
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NormalizedExactMatch(datasets.Metric):
    """
    Calculates some scores (corrently only exact match) for evaluating QA model given predictions and labels.
    """

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
        )

    def _compute(self, predictions, references):
        predictions = [self._normalize(p) for p in predictions]
        references = [self._normalize(r) for r in references]
        exact_match = sum(p == r for p, r in zip(predictions, references)) / len(predictions)

        return {
            "exact_match": exact_match,
        }

    def _normalize(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r"\s+", "", text)
        return text