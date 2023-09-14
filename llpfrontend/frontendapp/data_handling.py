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

  # todo: rework fking all of this, why do i need a default value if i am just goign to set it to "all" later?
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


def filter_experiments_by_filters(
    experiments: List[Dict],
    config_filter_values: Dict[str, Any],
    date_filter: str) -> List[Dict]:
  """Leaves out only those experiments that have passed all of the filters
  """
  log.info(f'Applying filters to {len(experiments)} experiments')
  experiments_that_have_passed_filtering = []
  for experiment in experiments:
    log.info(f'Applying filters to experiment "{experiment["_id"]}"')

    # filtering by date
    if date_filter is not None and date_filter != 'all' and experiment['date'] != date_filter:
      log.info(f'Filtered out experiment {experiment["_id"]} because of the date filter ({date_filter} does not match {experiment["date"]})')
      continue
    
    experiment_config = experiment['config'] # type: Dict[str, Any]
    
    # filtering by config parameters
    did_experiment_pass_config_filters = True
    for config_parameter_name, config_filter_value in config_filter_values.items():
      log.info(f'Filtering by config parameter "{config_parameter_name}" by "{config_filter_value}"')
      if config_filter_value == 'all':
        # allows all values
        log.info(f'Skipping filter {config_parameter_name} as it is set to "all"')
      elif config_filter_value == 'none':
        # disallows all values
        if config_parameter_name in experiment_config.keys():
          log.info(f'Experiment did not pass the filter (it contained the forbidden parameter)')
          did_experiment_pass_config_filters = False
          break
      else:
        # checks if the config parameter value matches the filter
        experiment_config_value = experiment_config.get(config_parameter_name, None)
        if experiment_config_value != config_filter_value:
          log.info(f'Experiment did not pass the filter ({experiment_config_value} did not match {config_filter_value})')
          did_experiment_pass_config_filters = False
          break
      log.info(f'Experiment has passed filtering by "{config_parameter_name}"')
    
    if did_experiment_pass_config_filters:
      log.info(f'Experiment "{experiment["_id"]}" has passed all the filters')
      experiments_that_have_passed_filtering.append(experiment)
    else:
      log.info(f'Experiment "{experiment["_id"]}" did not pass the fitlering')
    
  log.info(f'Got {len(experiments_that_have_passed_filtering)} experiments after the filtering')

  return experiments_that_have_passed_filtering


def aggregate_metrics_from_experiments(experiments: List[Dict]) -> Dict[str, float]:
  """Computes averages for metrics present in the experiments list
  """
  # creates a dict of lists for all metrics present in the experiment set
  all_metric_values = dict() # type Dict[str, List[float]]
  for experiment in experiments:
    for metric_name, metric_value in experiment['metrics'].items():
      existing_metric_values_list = all_metric_values.get(metric_value, list())
      all_metric_values[metric_name] = existing_metric_values_list + [metric_value]
  
  # computes the aggregate metrics
  aggregate_values = dict()
  for metric_name, metric_values_list in all_metric_values.items():
    # todo: add more aggregation options besides averages
    aggregate_values[metric_name] = sum(metric_values_list) / len(metric_values_list)

  return aggregate_values


def create_metrics_df(
    experiments: list,
    all_possible_parameters: List[Dict]) -> pd.DataFrame:
  """Returns a DF with following columns: model_id, task_type, metric_name, value
  """
  log.info(f'Computing metrics for {len(experiments)} experiments')
  unique_params = [parameter['name'] for parameter in all_possible_parameters]

  # todo: ugh rework it to use pandas pivot table or smth
  # This loop creates a DF with following columns: model_id, task_type, [parameter 1], ..., [parameter n], metric_name, value
  dicts_to_put_in_df = []
  for evaluation in experiments:
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
