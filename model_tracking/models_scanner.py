# this file will scan all HuggingFace text generation models, select top ones, and return them as pandas frame

# use this lib: https://huggingface.co/docs/huggingface_hub/v0.5.1/en/index

import json
import requests
import pandas as pd
import logging

from typing import *

log = logging.getLogger("model_scanner")
logging.basicConfig(level=logging.INFO)

from db_connector import DatabaseConnector

def get_openrouter_models(tracking_date: str) -> Dict[str, Dict]:
  opentouter_models_link = 'https://openrouter.ai/api/v1/models'
  response = requests.request("GET", opentouter_models_link)
  assert response.status_code == 200, f"OpenRouter models request returned with code {response.status_code}"
  response_json = json.loads(response.content.decode("utf-8"))
  models_list = response_json['data']

  formatted_models = []
  for model in models_list:
    source = 'OpenRouter'
    model_data = {
      'first_tracked_on': tracking_date,
      'last_tracked_on': tracking_date,
      'available': None,
      '_id': source + ':' + model['id'].split('/')[0] + ':' + model['id'].split('/')[1],
      'owner': model['id'].split('/')[0],
      'name': model['id'].split('/')[1],
      'price_prompt': model['pricing']['prompt'],
      'hf_inference_api_supported': False,
      'source': 'OpenRouter',
      'price_completion': model['pricing']['completion'],
      'context': model['context_length'],
      'prompt_limit': model['input_limits']['prompt_tokens'],
      'max_tokens_limit': model['input_limits']['max_tokens'],
    }
    #assert set(model_data.keys()) == set(DatabaseConnector.columns_list)
    formatted_models.append(model_data)
  
  return formatted_models


def scan_for_trackable_models(tracking_date: str) -> pd.DataFrame:
  empty_df = pd.DataFrame(columns=DatabaseConnector.columns_list)
  openrouter_df = get_openrouter_models(tracking_date)

  total_df = pd.concat([empty_df, openrouter_df], ignore_index=True)

  return total_df


def check_availability_if_applicable(models_list: pd.DataFrame) -> pd.Series:
  # True - availability is applicable and possible
  # None - availability is n/a
  # False - not available
  def availability_check(model_record):
    return None
  
  availability_series = models_list.apply(availability_check)
  return availability_series


def perform_tracking(tracking_date: str):
  # 1) get a list of trackable models
  # 2) get a list of previously tracked HF models
  # 3) combine the lists into a list of currently trackable models
  # 4) check availability of the models, drop those for which the check is applicable but failed
  # 5) save currently trackable models to db, saving first tracking date if the model is present in list #1

  print('\n\nStarting tracking on date {tracking_date}')

  db = DatabaseConnector()

  # only works with openrouter rn

  models_trackable_now = get_openrouter_models(tracking_date)
  log.info(f'Got {len(models_trackable_now)} models_trackable_now')
  for model in models_trackable_now:
    log.info(f'Saving model {model}')
    db.save_model_tracking_properly(model, tracking_date)

if __name__ == '__main__':
  perform_tracking('aaaaa')
