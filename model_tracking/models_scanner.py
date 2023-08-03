# this file will scan all HuggingFace text generation models, select top ones, and return them as pandas frame

# use this lib: https://huggingface.co/docs/huggingface_hub/v0.5.1/en/index

import json
import requests
import pandas as pd
import logging

log = logging.getLogger("model_scanner")
logging.basicConfig(level=logging.INFO)

from db_connector import DatabaseConnector

def get_openrouter_models(tracking_date: str) -> pd.DataFrame:
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
      'original_name': source + ':' + model['id'],
      'owner': model['id'].split('/')[0],
      'name': model['id'].split('/')[1],
      'price_prompt': model['pricing']['prompt'],
      'ff_inference_api_supported': False,
      'source': 'OpenRouter',
      'price_completion': model['pricing']['completion'],
      'context': model['context_length'],
      'prompt_limit': model['input_limits']['prompt_tokens'],
      'max_tokens_limit': model['input_limits']['max_tokens'],
    }
    assert set(model_data.keys()) == set(DatabaseConnector.columns_list)
    formatted_models.append(model_data)
  
  df = pd.DataFrame(formatted_models)
  return df


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

  print('Starting tracking on date {tracking_date}')

  db = DatabaseConnector()

  currently_trackable_models = scan_for_trackable_models(tracking_date)
  log.info(f'Got {len(currently_trackable_models)} currently_trackable_models')
  previously_tracked_hf_models = db.get_unique_tracked_hf_models()
  log.info(f'Got {len(previously_tracked_hf_models)} previously_tracked_hf_models')

  all_trackable_models = pd.concat(
    [currently_trackable_models, previously_tracked_hf_models],
    ignore_index=True)
  all_trackable_models = all_trackable_models.drop_duplicates('original_name')
  all_trackable_models = all_trackable_models.copy()
  log.info(f'Got {len(all_trackable_models)} all_trackable_models')
  
  assert set(all_trackable_models.columns) == set(DatabaseConnector.columns_list), f'{all_trackable_models.columns}'

  all_trackable_models['available'] = check_availability_if_applicable(all_trackable_models)
  all_trackable_models['last_tracked_on'] = tracking_date

  unique_already_tracked_models = db.get_unique_tracked_models()
  log.info(f'Got unique_already_tracked_models: \n{unique_already_tracked_models.head()}')
  already_tracked_names = unique_already_tracked_models['original_name']

  def first_tracking_date_assigner(row):
    name = row['original_name']
    if name in already_tracked_names:
      log.info(f'Model {name} is already tracked')
      return unique_already_tracked_models.loc[already_tracked_names == name, 'first_tracked_on'].iloc[0]
    else:
      return name
  all_trackable_models['first_tracked_on'] = all_trackable_models.apply(first_tracking_date_assigner, axis=1)

  db.save_models_tracking(all_trackable_models)

if __name__ == '__main__':
  perform_tracking('aaaaa')
