from enum import Enum
import json
import os
from typing import *
import string
import tiktoken
import logging
import re

from model_runner import ModelRunner
from runnable_model_data import RunnableModel
from task_output import TaskOutput

# todo: !! custom tasks must be saved to db and loaded from it

log = logging.getLogger("task.py")
logging.basicConfig(level=logging.INFO)

class TaskType(Enum):
  READING_COMPREHENSION = 1
  SPAM_CLASSIFICATION = 2
  CUSTOM = 3

class Task:
  alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
  idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}

  def __init__(self, task_type: TaskType) -> None:
    self.type = task_type

  def is_model_applicable_for_the_task(self, model: RunnableModel):
    # todo: make this work
    return True

  def load_reading_comprehension_data(self) -> Tuple[Dict, Dict]:
    # loads RACE data.
    # Stores it as a dict of texts {text_id: text} and questions {question_id: {text_id, question, options, answer}}

    with open("./model_tasks/rc/rc_dataset.txt", 'r') as file:
      dataset = json.load(file)
    rc_texts = dataset['texts']
    rc_questions = dataset['questions']
    
    return rc_texts, rc_questions


  def _get_answer_id_from_model_output(self, model_output: str) -> Union[int, None]:
    assert model_output is not None, "Model output is none"

    matches = re.findall(r'[a-zA-Z]', model_output)
    if len(matches) == 0:
      return None
    first_letter = matches[0].upper()
    return self.alphabet2idx.get(first_letter, None)


  def run_reading_comprehension(self, model: RunnableModel, config, cost_check_callback=None) -> Union[TaskOutput, None]:
    rc_texts, rc_questions = self.load_reading_comprehension_data()

    # todo: make the preprocessing code for the specific model on a specific task
    # creates a single prompt from the text and question
    def prompt_constructor(text, question_dict):
      question_text = question_dict['question']
      options_text = ''
      for i, option_text in enumerate(question_dict['options']):
        letter = self.idx2alphabet[i]
        options_text += f'{letter}) {option_text}\n'
      options_text = options_text.strip()
      final_text = ''.join(['Text:\n', text, '\nQuestion: ', question_text, '\nOptions:\n', options_text, '\nAnswer:'])
      return final_text

    runner = ModelRunner(model, config)

    # cost estimator
    # todo: make it a function
    log.info('Running token estimator')
    if cost_check_callback is not None:
      prompts = []
      for i, question_id in enumerate(rc_questions):
        if i % 1000 == 0:
          log.info(f'{i / len(rc_questions)}%')
        question_dict = rc_questions[question_id]
        text = rc_texts[question_dict['text_id']]
        prompts.append(prompt_constructor(text, question_dict))
      token_count = runner.count_tokens(prompts)
      cost = model.price * token_count
      cost_is_appropriate = cost_check_callback(cost)
      log.info(f'{token_count} at {cost} cost')
      if not cost_is_appropriate:
        log.warning(f'Skipping model {model._id} because the cost is too high ({cost})')
        return None

    model_outputs = []
    metrics = {'accuracy': 0}
    interpreted_outputs = []
    input_codes = []
    errors = []

    # evaluator. First - get model responses. Second - evaluate them
    # WIP - continue from here. Account for exceptions!
    log.info('Running evaluator')
    completed_evaluations = 0
    for i, question_id in enumerate(rc_questions):
      log.info(f'Running question {i} out of {len(rc_questions)}')
      input_codes.append(question_id)
      question_dict = rc_questions[question_id]
      answer_id = self.alphabet2idx[question_dict['answer']]
      text = rc_texts[question_dict['text_id']]
      prompt = prompt_constructor(text, question_dict)

      try:
        model_output = runner.run_model(prompt)
      except Exception as e:
        errors.append(str(e))
        continue
      model_answer_id = self._get_answer_id_from_model_output(model_output)

      if model_answer_id is not None and model_answer_id == answer_id:
        metrics['accuracy'] += 1

      model_outputs.append(model_output)
      interpreted_outputs.append(model_answer_id)
      errors.append(None)
      completed_evaluations += 1

    metrics['accuracy'] = metrics['accuracy'] / completed_evaluations
    return TaskOutput(metrics, model_outputs, interpreted_outputs, input_codes)

  # returns a list of metrics, outputs
  def run_task(self, model: RunnableModel, config, cost_check_callback=None) -> TaskOutput:
    if self.type == TaskType.READING_COMPREHENSION:
      log.info(f'Running reading comprehension on model {model._id}')
      return self.run_reading_comprehension(model, config, cost_check_callback)
    else:
      raise NotImplementedError(f'Tried running an unsupported task type, {self.type}')



