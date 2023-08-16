"""
This class will provide the means to run a variety of model types on the input data for different tasks.
The class will chose the appropriate way to run the model (OAI API, OpenRouter, HF inference, local, etc)
"""

import tiktoken
import logging
import json
import requests
import time
import asyncio
import aiohttp
from typing import *

from run_config import Config
from runnable_model_data import RunnableModel


log = logging.getLogger("task.py")
logging.basicConfig(level=logging.INFO)


with open("./s/hf_read", 'r') as file:
  HF_API_TOKEN = file.read()
hf_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


class ModelRunner:
  def __init__(self, model: RunnableModel, config: Config) -> None:
    self.model = model
    self.config = config

    # todo: eeh think about hf inference failing if it's unavailable?
    self._model_run_function = None
    if self.model.source == 'OpenRouter':
      self._model_run_function = self.run_openrouter_payload
    elif self.model.source == 'hf' and self.model.hf_inferable:
      self._model_run_function = self.run_hf_inference_payload
    elif self.model.source == 'hf' and not self.model.hf_inferable:
      pass
    else:
      raise NotImplementedError(f"Running from source {self.model_source} is NYI")




  #def run_task_with_preprocessing(task):
  #  pass
  async def _query_hf_single(self, payload: Union[str, List], model_name: str, session) -> dict:
    MAX_RETRIES = 10
    retry_count = 0
    API_URL = 'https://api-inference.huggingface.co/models/' + model_name
    data = json.dumps(payload)

    #payload_part_to_show = payload[:100]
    #log.info(f'Sending HF the payload:\n{payload_part_to_show}')
    log.info(f'Sending HF the payload')

    while retry_count <= MAX_RETRIES:
      retry_count += 1
      async with session.request("POST", API_URL, headers=hf_headers, data=data) as response:
        response_txt = await response.text()

        # wait quarter the estimated time if the model is loading
        if response.status == 503 and not response_txt == '{"error":"overloaded"}':
          wait_for = json.loads(response_txt)['estimated_time'] / 4
          log.warning(f'Waiting for {wait_for}')
          time.sleep(wait_for)
        elif response.status != 200:
          log.warning(f'Retrying HF request because of bad response ({response.status}): {response_txt}')
          time.sleep(10)
        else: # success
          response_json = json.loads(response_txt)
          log.info(f'Got response: {response_json}')
          return response_json

    raise TimeoutError(f"Model {model_name} took too many attempts to reply")


  def run_model(self, payload: Union[str, List[str]]) -> Union[str, List[str]]:
    # give a warning if the prompt is > context size
    log.info(f'Running model {self.model._id}')

    payload_size = self.count_tokens(payload)
    log.info(f'{payload_size} tokens')
    #if payload_size > self.model.context_size:
    #  log.warning(f'Payload size ({payload_size}) > max context size ({self.model.context_size})')
    
    return asyncio.run(self._model_run_function(payload))


  def run_openrouter_payload(self, payload) -> str:
    raise NotImplementedError('Openrouter is WIP')
    return 'OpenRouter response. WIP'


  '''async def run_multiple_payloads(self, payload, final_parameters):
    hf_responses = []
    hf_tasks = []
    for payload_item in payload:
      assert isinstance(payload_item, str), f'Unsupported payload element type: {type(payload)}'
      hf_tasks.append(asyncio.Task(self._query_hf_single({'inputs': payload, 'parameters': final_parameters}, self.model.name)))
    hf_responses = asyncio.gather(hf_tasks)
    return hf_responses
  '''


  async def run_hf_inference_payload(self, payload: Union[str, List[str]]) -> Union[str, List[str]]:
    parameters_to_get_and_defaults = {
      'temperature': 0.0000001,
      'top_p': 0.92,
      'top_k': 500,
      'max_new_tokens': 5,
    }
    final_parameters = {
      'return_full_text': False,
      'use_cache': False,
      'do_sample': True
    }

    for key, default_value in parameters_to_get_and_defaults.items():
      final_parameters[key] = self.config.get_parameter(key, default=default_value)
    
    connector = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(connector=connector) as session:
      # run a single request if payload is a string, run a lot of requests if payload is a list
      if isinstance(payload, str):
        hf_response = await self._query_hf_single({'inputs': payload, 'parameters': final_parameters}, self.model.name, session)
        return hf_response
      elif isinstance(payload, list):
        hf_tasks = []
        for payload_item in payload:
          assert isinstance(payload_item, str), f'Unsupported payload element type: {type(payload)}'
          hf_tasks.append(self._query_hf_single({'inputs': payload_item, 'parameters': final_parameters}, self.model.name, session))
        print('!')
        hf_responses = await asyncio.gather(*hf_tasks)
        return hf_responses
      else:
        raise TypeError(f'Unsupported payload type: {type(payload)}')


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