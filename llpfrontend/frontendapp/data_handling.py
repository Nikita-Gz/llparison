# todo: turn all db connector files into one?

import pandas as pd
import pymongo
import json
from os import environ
from typing import *
import mongomock
import logging

from .fake_run_data import get_fake_testing_evaluations

log = logging.getLogger("data_handling.py")
logging.basicConfig(level=logging.INFO)

RC_TEXTS = None # type: Dict[str, str]
RC_QUESTIONS = None # type: Dict[str, Dict]

def _load_raw_reading_comprehension_data() -> Tuple[Dict, Dict]:
  global RC_TEXTS
  global RC_QUESTIONS
  log.info('Loading RC dataset')
  with open("./rc_dataset.txt", 'r') as file:
    dataset = json.load(file)
  RC_TEXTS = dataset['texts']
  RC_QUESTIONS = dataset['questions']
_load_raw_reading_comprehension_data()


class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(self) -> None:
    # use a mock DB if the app is not on k8s

    running_on_k8s = environ.get('K8S_DEPLOYMENT') is not None
    log.info(f'running_on_k8s: {running_on_k8s}')
    if running_on_k8s:
      self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    else:
      self.mongo_client = mongomock.MongoClient()

    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']
    self.experiments = self.db['experiments']

    if not running_on_k8s:
      self._fill_with_testing_stuff()


  def _get_fake_testing_evaluations(self):
    return get_fake_testing_evaluations()

  
  def _fill_with_testing_stuff(self):
    testing_evals = self._get_fake_testing_evaluations()
    inserted_ids = self.experiments.insert_many(testing_evals).inserted_ids
    assert len(inserted_ids) == len(testing_evals)


  def get_unique_task_types(self) -> list:
    unique_tasks = self.experiments.distinct("task_type")
    return list(unique_tasks)


  def get_evaluations_for_llm_config_task_combination(self, task_type: str, combination: Dict):
    experiments_matching_model = list(self.experiments.find({
      'task_type': task_type,
      'finished': True,
      'too_expensive': False,
      'model_id': combination['model_id']}))
    config_to_look_for = combination['config']
    experiments_matching_models_and_configs = [
      experiment for experiment in experiments_matching_model
      if experiment['config'] == config_to_look_for
    ]
    return experiments_matching_models_and_configs


  def get_evaluations_for_llm_config_task_combinations(self, task_type: str, combinations: List[Dict]):
    experiments_matching_models = list(self.experiments.find({
      'task_type': task_type,
      'finished': True,
      'too_expensive': False,
      'model_id': {'$in': [combination['model_id'] for combination in combinations]}
    }))

    configs_to_look_for = [combination['config'] for combination in combinations]
    experiments_matching_models_and_configs = [
      experiment for experiment in experiments_matching_models
      if experiment['config'] in configs_to_look_for
    ]
    return experiments_matching_models_and_configs


  def get_unique_model_ids_with_finished_evaluations(self) -> list:
    # unique model - unique by id, present in experiments
    #return pd.DataFrame(columns=self.columns_list)
    unique_models = list(set(
      [
        experiment['model_id']
        for experiment in list(self.experiments.find({
        'finished': True,
        'too_expensive': False}))
      ]))
    return list(unique_models)
  

  def get_finished_evaluations_for_model(self, model_id) -> list:
    evaluations = self.experiments.find({
      'model_id': model_id,
      'finished': True,
      'too_expensive': False
      })
    return list(evaluations)


def get_unique_input_codes_from_evaluations(evaluations: List[Dict]) -> List[str]:
  all_codes = set()
  for evaluation in evaluations:
    codes = set([output['input_code'] for output in evaluation['outputs']])
    all_codes = all_codes.union(codes)
  return list(all_codes)


"""
Should report back unique filters in the following way:
[
  {
    'name': 'temperature',
    'values':
    [
      1.0, 0.5, 0.01
    ],
    'default': 1.0
  },
  {
    'name': 'top-p',
    'values':
    [
      0.2, 0.5, 0.9
    ],
    'default': 0.5
  },
],
"""


def get_possible_config_combinations_in_evaluations(evaluations: list) -> List[Dict[str, Any]]:
  combinations = set() # type Set[str]

  # encode config dicts as a sorted json string, for the purposes of using them in a set
  for evaluation in evaluations:
    config = evaluation['config'] # type: dict
    config_string = json.dumps(config, sort_keys=True)
    combinations.add(config_string)
  
  # decode them back to dicts
  final_combinations_list = []
  for combination in combinations:
    final_combinations_list.append(json.loads(combination))

  return final_combinations_list

# wait wtf arent these two (func above and below) almost the same?
# todo: make the lower one have a distinction

def get_unique_config_params_in_evaluations(evaluations: list) -> List[Dict]:
  all_unique_config_params = dict() # type: Dict[Set]
  # The key - unique param name, value - list of unique values for the param

  for evaluation in evaluations:
    config = evaluation['config'] # type: dict
    for config_param_name, config_param_value in config.items():
      existing_set_of_unique_vals = all_unique_config_params.get(config_param_name, set()) # type: Set
      all_unique_config_params[config_param_name] = existing_set_of_unique_vals.union({config_param_value,})

  # todo: rework fking all of this, why do i need a defailt value if i am jsut goign to set it to "all" later?
  final_list = []
  for parameter_name, unique_values in all_unique_config_params.items():
    unique_values = list(unique_values)
    dict_to_put = {
      'name': parameter_name,
      'values': unique_values,
      'default': unique_values[0]
    }
    final_list.append(dict_to_put)

  return final_list


"""
Example cols in df: model_id, task_type, parameter_1, parameter_2, metric_name, value
"""
def create_evaluations_df(evaluations: list, all_possible_parameters: List[Dict], filter_values: Dict[str, Any], date_filter: str) -> pd.DataFrame:
  """Returns a DF with following columns: model_id, task_type, metric_name, value
  """
  unique_params = [parameter['name'] for parameter in all_possible_parameters]

  # todo: ugh rework it to use pandas pivot table or smth
  dicts_to_put_in_df = []
  for evaluation in evaluations:
    if date_filter is not None and date_filter != 'all' and evaluation['date'] != date_filter:
      log.info(f'Filtered out {evaluation} because of the date filter ({date_filter})')
      continue

    model_id = evaluation['model_id']
    task_type = evaluation['task_type']
    
    # setting parameter columns
    parameter_values = dict()
    for unique_parameter_name in unique_params:
      parameter_values[unique_parameter_name] = evaluation['config'].get(unique_parameter_name, None)
    
    # putting it all in records
    for metric_name, metric_value in evaluation['metrics'].items():
      final_record = dict()
      final_record['model_id'] = model_id
      final_record['task_type'] = task_type
      final_record.update(parameter_values)
      final_record['metric_name'] = metric_name
      final_record['value'] = metric_value
      dicts_to_put_in_df.append(final_record)
  evals_pivot_table = pd.DataFrame(dicts_to_put_in_df)
  log.info(f'Created pivot table at length {len(evals_pivot_table)}')

  # filters the table
  for filter_name, filter_value in filter_values.items():
    log.info(f'Filtering by {filter_name}')
    if filter_value == 'all':
      log.info(f'Skipping filter {filter_name}')
    elif filter_value == 'none':
      log.info(f'Filtering {filter_name} by none')
      evals_pivot_table = evals_pivot_table[evals_pivot_table[filter_name].isna()]
    else:
      log.info(f'Filtering {filter_name} by {filter_value}')
      print(evals_pivot_table)
      evals_pivot_table = evals_pivot_table[evals_pivot_table[filter_name] == filter_value]
      print(evals_pivot_table[filter_name] == filter_value)
    log.info(f'DF length after the filter: {len(evals_pivot_table)}')
  
  return evals_pivot_table.groupby(
    ['model_id', 'task_type'] + ['metric_name'],
    as_index=False,
    dropna=False)['value'].mean()


def count_interpreted_answers_for_input_code(evaluations: List[Dict], input_code: str) -> Dict[str, int]:
  interpreted_output_counts = dict()
  for evaluation in evaluations:
    #print('ASDFSDF'*50)
    #print(evaluation)
    interpreted_outputs = [output['interpreted_output'] for output in evaluation['outputs'] if output['input_code'] == input_code]
    for interpreted_output in interpreted_outputs:
      interpreted_output_counts[interpreted_output] = interpreted_output_counts.get(interpreted_output, 0) + 1
  return interpreted_output_counts


def prettify_config_dict(config: Dict) -> str:
  parameter_strings = []
  for parameter_name, parameter_value in config.items():
    parameter_string = f'{parameter_name}={parameter_value}'
    parameter_strings.append(parameter_string)
  return ', '.join(parameter_strings)


# test
if __name__ == '__main__':
  c = DatabaseConnector()
  models = c.get_unique_model_ids_with_finished_evaluations()
  model_id = 'hf::gpt2'
  evals = c.get_finished_evaluations_for_model(model_id)
  unique_params = get_unique_config_params_in_evaluations(evals)
  a = create_evaluations_df(evals, unique_params, {'temperature': 1, 'testparam': 'none'}, 'all')
  print(a)
