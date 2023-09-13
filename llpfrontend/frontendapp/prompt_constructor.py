import json
import logging
import os
import string
from typing import *

log = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)


alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}


def _construct_default_reading_comprehension_prompt(text: str, question_dict: dict) -> str:
  # todo: create a template system
  question_text = question_dict['question']
  options_text = ''
  for i, option_text in enumerate(question_dict['options']):
    letter = idx2alphabet[i]
    # todo: use .join() instead
    options_text += f'{letter}) {option_text}\n'
  options_text = options_text.strip()
  final_text = ''.join([
    'Read the text and answer the question correctly\n',
    'Text:\n', text,
    '\nQuestion: ', question_text,
    '\nOptions:\n', options_text,
    '\nThe correct answer is the letter:'])
  return final_text


PROMPT_CONSTRUCTORS_MAP = { # maps task type and prompt type to constructor functions
  'Reading Comprehension': {
    'default': _construct_default_reading_comprehension_prompt
  }
}


class PromptConstructor:
  """This class selects the appropriate prompt constructor based on task type and config, and forwards prompt constructor arguments to it when construct_prompt is called
  """
  def __init__(
      self,
      task_type_str: str,
      configuration_dict: dict) -> None:
    # task type is a string, as the class is supposed to work without TaskType object
    self.constructor_function = self._get_prompt_constructor_function(task_type_str, configuration_dict) # type: Callable


  def construct_prompt(self, **kwargs) -> str:
    """Kwargs guide:

    - Reading comprehension: text: str, question_dict: dict
    """
    return self.constructor_function(**kwargs)
  
  
  def _get_prompt_constructor_function(
      self,
      task_type_str: str,
      configuration_dict: dict) -> Callable[..., str]:
    prompt_type = configuration_dict.get('prompt_type', 'default')
    log.info(f'Finding the prompt constructor function for prompt type {prompt_type} and task {task_type_str}')

    prompt_constructors_for_task = PROMPT_CONSTRUCTORS_MAP.get(task_type_str, None) # type: Union[dict, None]
    if prompt_constructors_for_task is None:
      raise AttributeError(f'Could not find prompt constructors for task {task_type_str}')

    prompt_constructor_for_task_and_prompt_type = prompt_constructors_for_task.get(prompt_type, None) # type: Union[Callable, None]
    if prompt_constructor_for_task_and_prompt_type is None:
      raise AttributeError(f'Could not find prompt constructor for task {task_type_str} and prompt type {prompt_type}')
    
    log.info(f'Found prompt constructor named {prompt_constructor_for_task_and_prompt_type.__name__}')

    return prompt_constructor_for_task_and_prompt_type
    