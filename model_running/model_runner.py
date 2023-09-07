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
import os
import random

models_cache = os.getcwd() + '/.hf/hfmodels/'
idksomeothercache = os.getcwd() + '/.hf/hfother/'
os.environ['TRANSFORMERS_CACHE'] = models_cache
os.environ['HF_DATASETS_CACHE'] = idksomeothercache
from transformers import pipeline

from run_config import Config
from runnable_model_data import RunnableModel
from eval_results_callback import EvaluationResultsCallback

log = logging.getLogger("task.py")
logging.basicConfig(level=logging.INFO)


HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if HF_API_TOKEN is None:
  log.warning("No HF token env variable")
hf_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


# todo: account for different models needing inputs with "assistant"/"user" stuff


class ModelRunner:
  def __init__(self, model: RunnableModel, config: Config) -> None:
    self.model = model
    self.config = config

    # todo: eeh think about hf inference failing if it's unavailable?
    #self._model_run_function = None


  def get_run_function(self):
    self._model_run_function = None
    if self.model.source == 'OpenRouter':
      return self.run_openrouter_payload
    elif self.model.source == 'hf' and self.model.hf_inferable:
      return lambda payload: asyncio.run(self.run_hf_inference_payload(payload))
    elif self.model.source == 'hf' and not self.model.hf_inferable:
      return self.run_hf_local
    else:
      raise NotImplementedError(f"Running from source {self.model_source} is NYI")


  def _get_local_model_if_exists(self) -> None:
    pass


  def run_hf_local(self, payloads: Dict[str, str], callback: EvaluationResultsCallback):
    if self.model.owner != '':
      model_name = self.model.owner + '/' + self.model.name
    else:
      model_name = self.model.name
    model_pipeline = pipeline(model=model_name)

    parameters_to_get_and_defaults = {
      'temperature': 2.0,
      'top_p': 0.99,
      'top_k': 5000,
      'max_new_tokens': 5,
      'return_full_text': False,
      'do_sample': True,
    }
    final_parameters = dict()

    for key, default_value in parameters_to_get_and_defaults.items():
      final_parameters[key] = self.config.get_parameter(key, default=default_value)

    for input_code, payload in payloads.items():
      generated_text = model_pipeline(text_inputs=payload, **final_parameters)[0]['generated_text']
      callback.record_output(generated_text, input_code=input_code)


  async def _query_hf_single(self, payload: Union[str, List], input_code: str, callback: EvaluationResultsCallback, session) -> dict:
    MAX_RETRIES = 10
    retry_count = 0
    API_URL = '/'.join(['https://api-inference.huggingface.co/models', self.model.owner, self.model.name])
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
          callback.record_output(response_json[0]['generated_text'], input_code=input_code)
          return response_json
    
    log.warning(f'Could not run input code {input_code} on URL {API_URL}')


  def _run_rc_test_function(self, payloads: Dict[str, str], callback: EvaluationResultsCallback):
    #rng = random.Random()
    #rng.seed(11037)
    #payloads_size = len(payloads)
    output_choices = ['A', 'B', 'C', 'D']
    for i, input_code in enumerate(payloads.keys()):
      if i % 1000 == 0:
        print(f'{i} out of {len(payloads)}')
      output = output_choices[random.randint(0, 3)]
      callback.record_output(output, input_code=input_code)


  # payloads - dict of {input_key: prompt}
  def run_model(self, payloads: Dict[str, str], callback: EvaluationResultsCallback):
    # give a warning if the prompt is > context size
    log.info(f'Running model {self.model._id}')

    payload_size = self.count_tokens(list(payloads.values()))
    log.info(f'{payload_size} tokens')
    #if payload_size > self.model.context_size:
    #  log.warning(f'Payload size ({payload_size}) > max context size ({self.model.context_size})')
    
    if self.model.name == 'rc_test_model':
      self._run_rc_test_function(payloads, callback)
    elif self.model.source == 'OpenRouter':
      self.run_openrouter_payload(payloads, callback)
    elif self.model.source == 'hf' and self.model.hf_inferable:
      asyncio.run(self.run_hf_inference_payload(payloads, callback))
    elif self.model.source == 'hf' and not self.model.hf_inferable:
      self.run_hf_local(payloads, callback)
    else:
      raise NotImplementedError(f"Running from source {self.model_source} is NYI")


  def run_openrouter_payload(self, payloads, callback: EvaluationResultsCallback) -> str:
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


  async def run_hf_inference_payload(self, payloads: Dict[str, str], callback: EvaluationResultsCallback):
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
      # run a lot of requests
      hf_tasks = []
      for input_code, payload in payloads.items():
        assert isinstance(payload, str), f'Unsupported payload element type: {type(payload)}'
        hf_tasks.append(self._query_hf_single({'inputs': payload, 'parameters': final_parameters}, input_code, callback, session))
      asyncio.gather(hf_tasks)


  def count_tokens(self, payloads: Union[str, List[str]]) -> int:
    # todo: make it supprot different encodings depending on the model
    tokenizer = tiktoken.get_encoding('cl100k_base')

    token_count = 0
    if isinstance(payloads, str):
      token_count += len(tokenizer.encode(payloads))
    elif isinstance(payloads, list):
      token_count += sum([len(count) for count in tokenizer.encode_batch(payloads)])
    else:
      raise TypeError(f'Payload has unsupported type of {type(payloads)}')
    
    return token_count