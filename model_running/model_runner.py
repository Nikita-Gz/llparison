"""
This class will provide the means to run a variety of model types on the input data for different tasks.
The class will chose the appropriate way to run the model (OAI API, OpenRouter, HF inference, local, etc)
"""

from run_config import Config

class ModelRunner:
  def __init__(self, model_definition_dict: dict, config: Config) -> None:
    self.model_name = model_definition_dict['name']
    self.model_owner = model_definition_dict['owner']
    self.hf_inferable = model_definition_dict['hf_inference_api_supported']
    self.available = model_definition_dict['available']

    self.config = config

    self.model_run_function = None
    if self.model_owner == 'OpenRouter':
      self.model_run_function = self.run_openrouter_payload

  #def run_task_with_preprocessing(task):
  #  pass
  
  def run_openrouter_payload(payload) -> str:
    return 'OpenRouter response. WIP'
