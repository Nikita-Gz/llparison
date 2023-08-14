"""
This class will provide the means to run a variety of model types on the input data for different tasks.
The class will chose the appropriate way to run the model (OAI API, OpenRouter, HF inference, local, etc)
"""

import tiktoken
from typing import *

from run_config import Config
from runnable_model_data import RunnableModel

class ModelRunner:
  def __init__(self, model_definition_dict: RunnableModel, config: Config) -> None:
    self.model_name = model_definition_dict.name
    self.model_owner = model_definition_dict.owner
    self.model_source = model_definition_dict.source
    self.hf_inferable = model_definition_dict.hf_inferable
    self.available = model_definition_dict.available

    self.config = config

    self.model_run_function = None
    if self.model_source == 'OpenRouter':
      self.model_run_function = self.run_openrouter_payload
    else:
      raise NotImplementedError(f"Running from source {self.model_source} is NYI")

  #def run_task_with_preprocessing(task):
  #  pass
  
  def run_openrouter_payload(self, payload) -> str:
    return 'OpenRouter response. WIP'
  
  def count_tokens(self, payloads: Union[str, List[str]]) -> int:
    # todo: make it supprot different encodings depending on the model
    tokenizer = tiktoken.get_encoding('cl100k_base')

    token_count = 0
    if isinstance(payloads, str):
      token_count += len(tokenizer.encode(payloads))
    elif isinstance(payloads, list):
      for payload in payloads:
        assert isinstance(payload, str), f'Payload {payload} is not int but a {type(payload)}'
        token_count += len(tokenizer.encode(payload))
    else:
      raise TypeError(f'Payloads have unsupported type of {type(payload)}')
    
    return token_count