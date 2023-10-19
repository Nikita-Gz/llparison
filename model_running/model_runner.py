"""
Dirty copy of a model_runner.py from model_running module to ffacilitate quicker development.
Todo: combine them all in a single module!

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

# todo: decide on this
#models_cache = os.getcwd() + '/.hf/hfmodels/'
#idksomeothercache = os.getcwd() + '/.hf/hfother/'
#os.environ['TRANSFORMERS_CACHE'] = models_cache
#os.environ['HF_DATASETS_CACHE'] = idksomeothercache

import torch
from transformers import AutoTokenizer
from transformers import pipeline

from run_config import Config
from runnable_model_data import RunnableModel
from eval_results_callback import EvaluationResultsCallback

log = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)


HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if HF_API_TOKEN is None:
  log.warning("No HF token env variable")
hf_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


# looks for openrouter token in the env or in the file if it's not in env
OPENROUTER_API_TOKEN = os.environ.get("OPENROUTER_API_TOKEN")
if HF_API_TOKEN is None:
  log.warning("Looking for OpenRouter token in the file")
  with open('./s/openrouter', 'r') as file:
    OPENROUTER_API_TOKEN = file.readline()
    log.warning('Got one')
openrouter_headers={
  "HTTP-Referer": 'http://localhost',
  "Authorization": f'Bearer {OPENROUTER_API_TOKEN}'
}


# todo: account for different models needing inputs with "assistant"/"user" stuff
class ModelRunner:
  def __init__(self, model: RunnableModel, config: Config) -> None:
    self.model = model
    self.config = config

    SIMULTANEOUS_REQUESTS_LIMIT = 2
    self.request_semaphore = asyncio.Semaphore(SIMULTANEOUS_REQUESTS_LIMIT)

    # creates the name used for accessing HF repositories
    if self.model.owner != '':
      self._hf_model_name = self.model.owner + '/' + self.model.name
    else:
      self._hf_model_name = self.model.name

    # todo: eeh think about hf inference failing if it's unavailable?
    #self._model_run_function = None


  def _get_device_code(self) -> int:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      return torch.cuda.current_device()
    else:
      return -1


  def run_hf_local(
      self,
      payloads: Dict[str, str],
      callback: EvaluationResultsCallback,
      max_new_tokens: int = 3):
    log.info(f'Running model {self.model._id} locally as HF model')

    #device_code = self._get_device_code()
    #log.info(f'Running on device #{device_code}')

    model_pipeline = pipeline(
      "text-generation",
      model=self._hf_model_name,
      device_map=0,
      tokenizer=AutoTokenizer.from_pretrained(self._hf_model_name),
      torch_dtype=torch.float16)

    model_parameters = self.assemble_model_parameters(max_new_tokens)

    iteration = 0
    for input_code, payload in payloads.items():
      generated_text = model_pipeline(text_inputs=payload, **model_parameters)[0]['generated_text']
      callback.record_output(generated_text, input_code=input_code)
      if iteration % 1 == 0:
        log.info(f'Processed {iteration} inputs out of {len(payloads)}')
        log.info(f'Example output: {generated_text}')
      iteration += 1


  async def _query_hf_single(self, payload: Union[str, List], input_code: str, callback: EvaluationResultsCallback, session) -> dict:
    MAX_RETRIES = 10
    retry_count = 0
    API_URL = 'https://api-inference.huggingface.co/models/' + self._hf_model_name
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
    log.info(f'Running model {self.model._id} as a RC test model')
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
  def run_model(
      self, payloads: Dict[str, str], callback: EvaluationResultsCallback, max_new_tokens: int):
    # give a warning if the prompt is > context size
    log.info(f'Running model {self.model._id}')
    
    if self.model.name == 'rc_test_model':
      self._run_rc_test_function(payloads, callback)
    elif self.model.source == 'OpenRouter':
      asyncio.run(self.run_openrouter_as_multiple_requests(payloads, callback, max_new_tokens=max_new_tokens))
    elif self.model.source == 'hf' and self.model.hf_inferable:
      asyncio.run(self.run_hf_inference_payload(payloads, callback))
    elif self.model.source == 'hf' and not self.model.hf_inferable:
      self.run_hf_local(payloads, callback, max_new_tokens=max_new_tokens)
    else:
      raise NotImplementedError(f"Running from source {self.model.source} is NYI")


  async def run_single_openrouter_request(
      self,
      input_code: str,
      payload: str, # aka prompt in this case
      callback: EvaluationResultsCallback,
      model_parameters: Dict,
      session: aiohttp.ClientSession) -> None:
    """Queries OpenRouter for N amount of tries, sleeps for M seconds after each, gives up if exceepds N"""
    MAX_RETRIES = 10
    retry_count = 0
    WAIT_DURATION = 10
    API_URL = 'https://openrouter.ai/api/v1/chat/completions'

    data_dict = {
      "model": f"{self.model.owner}/{self.model.name}",
      "prompt": payload
    }
    data_dict.update(model_parameters)
    request_data_json = json.dumps(data_dict)

    async def make_request() -> Union[str, None]:
      nonlocal retry_count
      while retry_count <= MAX_RETRIES:
        retry_count += 1
        async with session.request("POST", API_URL, headers=openrouter_headers, data=request_data_json) as response:
          log.info(f'Requesting from OpenRouter')
          response_txt = await response.text()

          if response.status == 200:
            response_dict = json.loads(response_txt)
            log.info(f'Got response: {response_dict}')
            response_model_text = response_dict['choices'][0]['text']
            return response_model_text
          else: # just wait
            log.warn(f'Got response code {response.status} on input code {input_code} with response {response_txt}')
            time.sleep(WAIT_DURATION)
      
      return None # all retries failed
    
    try:
      response_model_text = await make_request()
    except Exception as e:
      log.error(e)
      response_model_text = None
      callback.increment_counter_in_notes('Exceptions on inference')
    
    if response_model_text is None:
      log.error(f'Could not run input code {input_code} on URL {API_URL}')
    else:
      callback.record_output(response_model_text, input_code=input_code)


  async def run_openrouter_as_multiple_requests(
      self,
      payloads: Dict[str, str],
      callback: EvaluationResultsCallback,
      max_new_tokens: int):
    log.info(f'Running model {self.model._id} via OpenRouter inference API')

    model_parameters = self.assemble_model_parameters(max_new_tokens=max_new_tokens)
    model_parameters['max_tokens'] = model_parameters['max_new_tokens']
    
    connector = aiohttp.TCPConnector(limit=1)
    session = aiohttp.ClientSession(connector=connector)
    
    try: # close the session in case of an exception
      openrouter_tasks = []
      for input_code, payload in payloads.items():
        assert isinstance(payload, str), f'Unsupported payload element type: {type(payload)}'
        openrouter_tasks.append(self.run_single_openrouter_request(
          input_code=input_code,
          payload=payload,
          callback=callback,
          model_parameters=model_parameters,
          session=session))
      
      await asyncio.gather(*openrouter_tasks)
    except Exception as e:
      session.close()
      raise e
    session.close()


  def assemble_model_parameters(self, max_new_tokens: int) -> dict:
    """Using hte model config, assembles a dict of parameters that go into the model during inference.
    Only takes the parameters with pre-defined name, uses pre-defined default values if the parameter is not present in the config"""
    parameters_to_get_and_defaults_with_types = {
      'temperature': (0.01, float),
      'top_p': (0.9, float),
      'top_k': (1, int),
      'max_new_tokens': (max_new_tokens, int),
      'return_full_text': (False, bool),
      'do_sample': (True, bool),
    }
    final_parameters = dict()

    # fills the model's parameters list with those from the saved config, converting type and using default if nevessary
    for key, (default_value, expected_type) in parameters_to_get_and_defaults_with_types.items():
      try:
        parameter_value = self.config.get_parameter(key, default=default_value)
        final_parameters[key] = expected_type(parameter_value)
      except TypeError as e:
        error_msg = f'Could not interpret type {type(parameter_value)} as {expected_type} for parameter {key} (got value {parameter_value})'
        log.error(error_msg)
        raise TypeError(error_msg)
    return final_parameters
  


  async def run_hf_inference_payload(self, payloads: Dict[str, str], callback: EvaluationResultsCallback):
    log.info(f'Running model {self.model._id} via HF inference API')

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



class SimpleRunner:
  """Allows for a quick one-time generation using ModelRunner, without the need to manage callbacks and everything else"""
  def __init__(self, runner: ModelRunner) -> None:
    class RecorderCallback:
      def __init__(self) -> None:
        self.output = None
      
      def record_output(self, output, **kwargs):
        self.output = output
    
    self.callback = RecorderCallback()
    self.runner = runner
  
  def run(self, prompt, token_limit: int) -> str:
    self.runner.run_model({'_': prompt}, callback=self.callback, max_new_tokens=token_limit)
    return self.callback.output

