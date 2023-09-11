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
      '_id': source + ':' + model['id'].split('/')[0] + ':' + model['id'].split('/')[1],
      'owner': model['id'].split('/')[0],
      'name': model['id'].split('/')[1],
      'source': 'OpenRouter',
      'tracking_history': [
        {
          'date': tracking_date,
          'hf_inference_api_supported': False,
          'available': None,
          'context_size': int(model['context_length']),
          'price_prompt': float(model['pricing']['prompt']),
          'price_completion': float(model['pricing']['completion']),
          'prompt_limit': int(model['input_limits']['prompt_tokens']),
          'max_tokens_limit': int(model['input_limits']['max_tokens'])
        }
      ]
    }
    #assert set(model_data.keys()) == set(DatabaseConnector.columns_list)
    formatted_models.append(model_data)
  log.info(f'Got {len(formatted_models)} models from OpenRouter')
  return formatted_models


def scan_for_new_models(tracking_date: str, db_connector: DatabaseConnector) -> List[Dict]:
  openrouter_df = get_openrouter_models(tracking_date)
  # scan for other models here too

  currently_found_models = [] + openrouter_df
  previously_unknown_models = []
  previously_known_models_ids = [model['_id'] for model in list(db_connector.get_models())]
  for currently_found_model in currently_found_models:
    if currently_found_model['_id'] not in previously_known_models_ids:
      previously_unknown_models.append(currently_found_model)
  
  return previously_unknown_models


def check_availability_if_applicable(models_list: pd.DataFrame) -> pd.Series:
  # True - availability is applicable and possible
  # None - availability is n/a
  # False - not available
  def availability_check(model_record):
    return None
  
  availability_series = models_list.apply(availability_check)
  return availability_series


def update_hf_model(model: Dict, db_connector: DatabaseConnector, tracking_date: str):
  tracking_history = model['tracking_history']
  id = model['_id']
  if len(tracking_history) == 0:
    # todo: there should be an API call to check for inferability, as well as the ability to download it
    db_connector.save_model_tracking_properly(
      {
        '_id': id,
        'hf_inference_api_supported': False,
        'available': True,
        'context_size': 2000,
        'price_prompt': 0,
        'price_completion': 0,
        'prompt_limit': 0,
        'max_tokens_limit': 0
      },
      tracking_date)
  else:
    log.info(f'Skipping tracking HF model {id} because it was already tracked')


def update_openrouter_models(models: List[Dict], db_connector: DatabaseConnector, tracking_date: str):
  currently_available_models = get_openrouter_models(tracking_date)
  currently_available_models_dict = {model['_id']:model for model in currently_available_models}
  #ids_of_models_to_update = [model['_id'] for model in models]

  for model_to_update in models:
    if model_to_update['_id'] in currently_available_models_dict.keys():
      db_connector.save_model_tracking_properly(
        currently_available_models_dict[model_to_update['_id']]['tracking_history'][0],
        tracking_date)



def update_existing_models_tracking(tracking_date: str, db_connector: DatabaseConnector):
  all_models = list(db_connector.get_models())
  log.info(f'Updating {len(all_models)} models')

  # huggingface updates
  for model in all_models:
    if model['source'] == 'hf':
      update_hf_model(model)
  
  # openrouter updates
  update_openrouter_models([model for model in all_models if model['source'] == 'OpenRouter'], db_connector, tracking_date)


def reworked_perform_tracking(tracking_date: str):
  db = DatabaseConnector()

  previously_unknown_models = scan_for_new_models(tracking_date, db)
  db.save_models(previously_unknown_models)

  update_existing_models_tracking(tracking_date, db)




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
