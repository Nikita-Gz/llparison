import json
import logging
import os
import string
from typing import Any
import tiktoken
from typing import *
from transformers import AutoTokenizer

from task_type import TaskType, new_tokens_limit_per_task_type
from runnable_model_data import RunnableModel

log = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)


alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}


class UniversalTokenizer:
  def __init__(self, model: RunnableModel) -> None:
    log.info(f'Creating tokenizer for model {model._id}')

    # creates the name used for accessing HF repositories
    if model.owner != '':
      hf_model_name = model.owner + '/' + model.name
    else:
      hf_model_name = model.name

    # Selects an appropriate tokenizer for HF models, selects a default tokenizer for other models
    if model.source == 'hf':
      log.info(f'Using HF tokenizer for {hf_model_name}')
      tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
      tokenizer.encode_batch = lambda texts: [tokenizer.encode(text) for text in texts] # type: ignore
      tokenizer.decode_batch = lambda texts: [tokenizer.decode(text) for text in texts] # type: ignore
    else:
      DEFAULT_PROPRIETARY_TOKENIZER = 'cl100k_base'
      log.info(f'Using default tokenizer: {DEFAULT_PROPRIETARY_TOKENIZER}')
      tokenizer = tiktoken.get_encoding('cl100k_base')
    
    self._tokenizer = tokenizer
  
  def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
    if isinstance(text, str):
      return self._tokenizer.encode(text)
    elif isinstance(text, list):
      return self._tokenizer.encode_batch(text) # type: ignore
    else:
      raise TypeError(f'Not supported text type ({type(text)})')
  

  def decode(
      self,
      encoded_text: Union[List[int], List[List[int]]],
      array_of_texts_was_passed: bool) -> Union[str, List[str]]:
    if array_of_texts_was_passed:
      return self._tokenizer.decode_batch(encoded_text) # type: ignore
    else:
      return self._tokenizer.decode(encoded_text) # type: ignore


def _construct_default_reading_comprehension_prompt(
    context_text: str,
    question_dict: dict,
    model: RunnableModel,
    task_type: TaskType,
    tokenizer: UniversalTokenizer) -> Tuple[str, int, int]:
  """Creates a default RC prompt, matching for the model
  
  Returns the prompt, and the number of tokens it was cut by"""

  question_text = question_dict['question']
  options_text = ''
  for i, option_text in enumerate(question_dict['options']):
    letter = idx2alphabet[i]
    options_text += f'{letter}) {option_text}\n'
  options_text = options_text.strip()

  # compute the token size for all the necessary text (header, question, options, suffic),
  # then cull the context text if necessary

  header_text = 'Read the text and answer the question correctly\n'
  questions_text = '\nQuestion: ' + question_text
  options_text = '\nOptions:\n' + options_text
  suffix_text = '\nThe correct answer is the letter:'
  encoded_necessary_text = tokenizer.encode([header_text, questions_text, options_text, suffix_text]) # type: List[List[int]]
  necessary_text_token_count = sum([len(tokens) for tokens in encoded_necessary_text])
  assert necessary_text_token_count < model.context_size, f'Necessary text ({necessary_text_token_count}) is larger than model context ({model.context_size})'

  tokenized_context = tokenizer.encode(context_text)
  max_tokens_after_generation = necessary_text_token_count + len(tokenized_context) + new_tokens_limit_per_task_type[task_type]
  exceeded_context_size_by = max(max_tokens_after_generation - model.context_size, 0)
  if exceeded_context_size_by > 0:
    log.warning(f'Cutting down RC text by {exceeded_context_size_by + 1} tokens (1 extra to add a "..." to the end)')
    context_text = tokenizer.decode(
      tokenized_context[:-(exceeded_context_size_by + 1)],
      array_of_texts_was_passed=False) + '...'

  final_text = ''.join([header_text, context_text, questions_text, options_text, suffix_text])
  return final_text, max_tokens_after_generation, exceeded_context_size_by


def _construct_default_bot_detection_prompt(text: str, question_dict: dict) -> Tuple[str, int, int]:
  pass


PROMPT_CONSTRUCTORS_MAP = { # maps task type and prompt type to constructor functions
  'Reading Comprehension': {
    'default': _construct_default_reading_comprehension_prompt
  },
  'Bot Detection': {
    'default': _construct_default_bot_detection_prompt
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


  def construct_prompt(self, /, model: RunnableModel, task_type: TaskType, tokenizer: UniversalTokenizer, **kwargs) -> Tuple[str, int, int]:
    """Required arguments:
    - tokenizer: UniversalTokenizer
    - task_type: TaskType
    - model: RunnableModel
    
    Kwargs guide per task type:
    - Reading comprehension:
    - - context_text: str
    - - question_dict: dict

    Returns the prompt text, token count, as well as the number of tokens it was cut by
    """
    return self.constructor_function(model=model, task_type=task_type, tokenizer=tokenizer, **kwargs)
  
  
  def _get_prompt_constructor_function(
      self,
      task_type_str: str,
      configuration_dict: dict) -> Callable[..., Tuple[str, int, int]]:
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
    