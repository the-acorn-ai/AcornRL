"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import pandas

from . import common
from .common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_REGEXES,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE_PRO,
    format_multichoice_question_pro,
    normalize_extracted_answer,
    normalize_response,
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
import datasets

class MMLUProEval(Eval):
    def __init__(self, num_examples: int | None = None, language: str = "EN-US"):
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        df = ds.to_pandas()
        examples = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Convert options from numpy array to list
            if 'options' in row_dict and hasattr(row_dict['options'], 'tolist'):
                row_dict['options'] = row_dict['options'].tolist()
            examples.append(row_dict)
        
        print(examples[0])
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question_pro(row), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages))
            extracted_answer = None
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE_PRO.format(answer_regex)
                match = re.search(regex, response_text)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            score = 1.0 if extracted_answer == row["answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = row["category"]
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
