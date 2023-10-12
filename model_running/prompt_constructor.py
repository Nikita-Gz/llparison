import json
import logging
import os
import string
from typing import Any
import tiktoken
from typing import *
from transformers import AutoTokenizer

from run_config import Config
from task_type import TaskType, new_tokens_limit_per_task_type_int, task_type_int_to_str
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
    self._cached_encodings = {}


  def _get_cached_encoding(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]], None]:
    """Returns already saved cached encoding, returns None if it wasnt cached"""

    # converts the list to tuple to allow dict lookup
    if isinstance(text, list):
      text = tuple(text)
    
    return self._cached_encodings.get(text, None)


  def _save_cached_encoding(self, text: Union[str, List[str]], encoding: Union[List[int], List[List[int]]]) -> None:
    """Saves cached encoding"""

    # converts the encoding to tuple to allow dict lookup
    if isinstance(text, list):
      text = tuple(text)
    
    self._cached_encodings[text] = encoding


  def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
    """If the input wasn't cached, encodes the text and caches it for reuse"""

    cached_encoding = self._get_cached_encoding(text)
    if cached_encoding is not None:
      return cached_encoding

    if isinstance(text, str):
      encoded_result = self._tokenizer.encode(text)
    elif isinstance(text, list):
      encoded_result = self._tokenizer.encode_batch(text) # type: ignore
    else:
      raise TypeError(f'Not supported text type ({type(text)})')
    
    self._save_cached_encoding(text, encoded_result)
    return encoded_result
  

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
    tokenizer: UniversalTokenizer,
    **kwargs) -> Tuple[str, int, int]:
  """Creates a default RC prompt, matching for the model
  
  Returns the prompt, token count, and the number of tokens it was cut by"""

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
  max_tokens_after_generation = necessary_text_token_count + len(tokenized_context) + new_tokens_limit_per_task_type_int[task_type]
  exceeded_context_size_by = max(max_tokens_after_generation - model.context_size, 0)
  if exceeded_context_size_by > 0:
    log.warning(f'Cutting down RC text by {exceeded_context_size_by + 1} tokens (1 extra to add a "..." to the end)')
    context_text = tokenizer.decode(
      tokenized_context[:-(exceeded_context_size_by + 1)],
      array_of_texts_was_passed=False) + '...'

  final_text = ''.join([header_text, context_text, questions_text, options_text, suffix_text])
  return final_text, max_tokens_after_generation, exceeded_context_size_by


def _construct_default_bot_detection_prompt(
    post_history: List[str],
    model: RunnableModel,
    task_type: TaskType,
    tokenizer: UniversalTokenizer,
    **kwargs) -> Tuple[str, int, int]:
  """Creates a default bot detection prompt, matching for the model
  
  Returns the prompt, token count, and the number of posts it was cut by"""

  header_text = """You are a helpful bot detection program. You will read the posts made by a user, and determine if they were made by a bot (Y/N). Here is the example:
Post: Airline employee steals plane from Seattle airport, crashes and dies - CNN: CNN Airline employee steals plane from Seattle airport, crashes and dies CNN (CNN) An airline employee stole an otherwise unoccupied passenger plane Friday from the\u2026 https://t.co/2FbjpYYuUH https://t.co/gtvQKqz4YG
Was this written by a bot?: "Y"
Post: Late night video editing. Finishing it up pre' soon. Hopefully I'll be able to put it up on YouTube in… http://t.co/9OtxogubwQ
Was this written by a bot?: "N"
Post: It's highly unlikely, but I'd love if @andy_murray beat Djokovic here. As soon as he's back Murray should take a medical time out!
Was this written by a bot?: "N"
Post: Children at Play, ca. 1895-1897 https://t.co/P9Mpn4TNYS https://t.co/MJR7laIwcM
Was this written by a bot?: "Y"
Here is the user's post history:
"""
  suffix_text = 'Were these posts made by a bot?: "'
  encoded_necessary_text = tokenizer.encode([header_text, suffix_text])
  necessary_text_token_count = sum([len(tokens) for tokens in encoded_necessary_text]) + new_tokens_limit_per_task_type_int[task_type]
  max_allowed_tokens_for_posts = model.context_size - necessary_text_token_count

  '''
  Posts text should look like this:
  Post 1) some text some text some text
  Post 2) some text some text some text
  '''
  posts_current_token_count = 0
  posts_to_add = []
  for i, post in enumerate(post_history):
    post_text_to_add = f'Post {i+1}) {post}\n###\n'
    post_token_size = len(tokenizer.encode(post_text_to_add))
    posts_new_token_count = posts_current_token_count + post_token_size
    if posts_new_token_count < max_allowed_tokens_for_posts:
      posts_to_add.append(post_text_to_add)
      posts_current_token_count = posts_new_token_count
  total_posts_text = ''.join(posts_to_add)

  final_text = ''.join([header_text, total_posts_text, suffix_text])
  total_token_count = necessary_text_token_count + posts_current_token_count
  cut_posts = len(post_history) - len(posts_to_add)

  return final_text, total_token_count, cut_posts


PROMPT_CONSTRUCTORS_MAP = { # maps task type and prompt type to constructor functions
  TaskType.READING_COMPREHENSION: {
    'default': _construct_default_reading_comprehension_prompt
  },
  TaskType.BOT_DETECTION: {
    'default': _construct_default_bot_detection_prompt,
    'without examples': _construct_default_bot_detection_prompt
  }
}


class PromptConstructor:
  """This class selects the appropriate prompt constructor based on task type and config, and forwards prompt constructor arguments to it when construct_prompt is called
  """
  def __init__(
      self,
      task_type: TaskType,
      configuration_dict: dict) -> None:
    self.task_type = task_type
    self.configuration_dict = configuration_dict
    self.constructor_function = self._get_prompt_constructor_function() # type: Callable


  def construct_prompt(self, /, model: RunnableModel, tokenizer: UniversalTokenizer, **kwargs) -> Tuple[str, int, int]:
    """Required arguments:
    - tokenizer: UniversalTokenizer
    - model: RunnableModel
    
    Kwargs guide per task type:
    - Reading comprehension:
    - - context_text: str
    - - question_dict: dict
    - Bot Detection:
    - - post_history: List[str]

    Returns the prompt text, token count, as well as the number of tokens it was cut by
    """
    return self.constructor_function(
      model=model,
      task_type=self.task_type,
      tokenizer=tokenizer,
      config=self.configuration_dict,
      **kwargs)
  
  
  def _get_prompt_constructor_function(self) -> Callable[..., Tuple[str, int, int]]:
    prompt_type = self.configuration_dict.get('prompt_type', 'default')
    log.info(f'Finding the prompt constructor function for prompt type {prompt_type} and task {self.task_type}')

    prompt_constructors_for_task = PROMPT_CONSTRUCTORS_MAP.get(self.task_type, None) # type: Union[dict, None]
    if prompt_constructors_for_task is None:
      raise AttributeError(f'Could not find prompt constructors for task {self.task_type}')

    prompt_constructor_for_task_and_prompt_type = prompt_constructors_for_task.get(prompt_type, None) # type: Union[Callable, None]
    if prompt_constructor_for_task_and_prompt_type is None:
      raise AttributeError(f'Could not find prompt constructor for task {self.task_type} and prompt type {prompt_type}')
    
    log.info(f'Found prompt constructor named {prompt_constructor_for_task_and_prompt_type.__name__}')

    return prompt_constructor_for_task_and_prompt_type
    