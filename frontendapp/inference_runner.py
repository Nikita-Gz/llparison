"""
This code provides the ability to run inference initiated by the user with user-provided task type and relevant data and parameters
"""

import logging
from typing import *
import datetime
from threading import Lock
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_running.task_type import TaskType, new_tokens_limit_per_task_type_int
from model_running.runnable_model_data import RunnableModel
from model_running.prompt_constructor import UniversalTokenizer, PromptConstructor
from model_running.run_config import Config
from .frontend_data_handling import DatabaseConnector
from model_running.model_runner import ModelRunner, SimpleRunner

log = logging.getLogger("inference_runner.py")
logging.basicConfig(level=logging.DEBUG)

class InferenceRunner:
  def __init__(self, db: DatabaseConnector) -> None:
    self._inference_mutex = Lock()
    self._db = db


  def _make_reading_comprehension_prompt_arguments(self, input_fields: Dict[str, str]) -> Dict:
    """Returns kwargs to be sent to prompt constructor when running reading comprehension task"""
    created_arguments = {
      'context_text': input_fields['Context'],
      'question_dict': {
        'question': input_fields['Question'],
        'options': [input_fields[f'Answer {i+1}'] for i in range(4)]
      }
    }
    log.info(f'Created the following inference arguments: {created_arguments}')
    return created_arguments
  

  def _make_bot_detection_prompt_arguments(self, input_fields: Dict[str, str]) -> Dict:
    """Returns kwargs to be sent to prompt constructor when running bot detection task"""
    created_arguments = {
      'post_history': input_fields['Post history'].split('\n')
    }
    log.info(f'Created the following inference arguments: {created_arguments}')
    return created_arguments


  def _make_multiplication_prompt_arguments(self, input_fields: Dict[str, str]) -> Dict:
    """Returns kwargs to be sent to prompt constructor when running multiplication task"""
    created_arguments = {
      'math_expression': input_fields['Math expression']
    }
    log.info(f'Created the following inference arguments: {created_arguments}')
    return created_arguments


  def infer(
      self,
      model: RunnableModel,
      task_type: TaskType,
      input_fields: Dict[str, str],
      config: Config,
      tokenizer: UniversalTokenizer) -> Tuple[str, int]:
    """
    Creates prompt from the input fields, runs the inference with the specified config.
    Returns the model's output and the time it took in microseconds.
    """

    prompt_constructor_kwargs_builder = {
      TaskType.READING_COMPREHENSION: self._make_reading_comprehension_prompt_arguments,
      TaskType.BOT_DETECTION: self._make_bot_detection_prompt_arguments,
      TaskType.MULTIPLICATION: self._make_multiplication_prompt_arguments
    }[task_type]


    with self._inference_mutex: # only one inference thread can work at a time to limit resource use
      log.info(f'Constructing prompt')

      prompt_kwargs = prompt_constructor_kwargs_builder(input_fields)
      prompt_ctor = PromptConstructor(task_type, config.to_dict(), model, existing_tokenizer=tokenizer)
      prompt, _, _ =prompt_ctor.construct_prompt(**prompt_kwargs)
      
      log.info(f'Received prompt: {prompt}')
      log.info(f'Running inference')
      base_runner = ModelRunner(model, config)
      simple_runner = SimpleRunner(base_runner)
      start = datetime.datetime.now()
      output = simple_runner.run(prompt, token_limit=new_tokens_limit_per_task_type_int[task_type])
      end = datetime.datetime.now()

      delta = (end-start).microseconds
      log.info(f'Got output {output} in {delta} seconds')
    return output, delta
    