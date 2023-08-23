# todo: turn all db connector files into one?

import pandas as pd
import pymongo
from os import environ
from typing import *
import mongomock
import logging

log = logging.getLogger("data_handling.py")
logging.basicConfig(level=logging.INFO)

class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(self) -> None:
    # use a mock DB if the app is not on k8s

    running_on_k8s = environ.get('K8S_DEPLOYMENT') is not None
    if running_on_k8s:
      self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    else:
      self.mongo_client = mongomock.MongoClient()

    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']
    self.experiments = self.db['experiments']

    if not running_on_k8s:
      self._fill_with_testing_stuff()


  def _get_testing_evaluations(self):
    return [
      {
        #'_id': uuid4().int,
        'date': '2023-01-02',
        'model_id': 'hf::gpt2',
        'iterations': 2,
        'config': {'temperature': 1.0, 'top-p': 0.5},
        'notes': '',
        'task_type': 'Reading Comprehension',
        'metrics': {'accuracy': 0.65, 'f1': 0.6},
      },
      {
        #'_id': uuid4().int,
        'date': '2023-01-03',
        'model_id': 'hf::gpt2',
        'iterations': 2,
        'config': {'temperature': 1.0, 'top-p': 0.5},
        'notes': '',
        'task_type': 'Reading Comprehension',
        'metrics': {'accuracy': 0.75, 'f1': 0.7},
      },
      {
        #'_id': uuid4().int,
        'date': '2023-01-02',
        'model_id': 'hf::gpt2',
        'iterations': 2,
        'config': {'temperature': 0.01, 'top-p': 0.5},
        'notes': '',
        'task_type': 'Reading Comprehension',
        'metrics': {'accuracy': 0.75, 'f1': 0.7},
      },
      {
        #'_id': uuid4().int,
        'date': '2023-01-02',
        'model_id': 'hf::gpt2',
        'iterations': 2,
        'config': {'temperature': 0.01, 'top-p': 0.5, 'testparam': 1},
        'notes': '',
        'task_type': 'Reading Comprehension',
        'metrics': {'accuracy': 0.75, 'f1': 0.7},
      },
      {
        #'_id': uuid4().int,
        'date': '2023-01-02',
        'model_id': 'hf::idkanymoreplshelp',
        'iterations': 2,
        'config': {'temperature': 1},
        'notes': '',
        'task_type': 'Reading Comprehension',
        'metrics': {'accuracy': 0.75, 'f1': 0.7},
      },
    ]

  
  def _fill_with_testing_stuff(self):
    testing_evals = self._get_testing_evaluations()
    inserted_ids = self.experiments.insert_many(testing_evals).inserted_ids
    assert len(inserted_ids) == len(testing_evals)


  def get_unique_models_with_evaluations(self) -> list:
    # unique model - unique by id, present in experiments
    #return pd.DataFrame(columns=self.columns_list)
    unique_models = self.experiments.distinct("model_id")
    return list(unique_models)
  

  def get_evaluations_for_model(self, model_id) -> list:
    evaluations = self.experiments.find({'model_id': model_id})
    return list(evaluations)

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


# test
if __name__ == '__main__':
  c = DatabaseConnector()
  models = c.get_unique_models_with_evaluations()
  model_id = 'hf::gpt2'
  evals = c.get_evaluations_for_model(model_id)
  unique_params = get_unique_config_params_in_evaluations(evals)
  a = create_evaluations_df(evals, unique_params, {'temperature': 1, 'testparam': 'none'}, 'all')
  print(a)
