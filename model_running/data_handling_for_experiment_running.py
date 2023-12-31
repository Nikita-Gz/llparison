"""
This file contains the class and functions for interacting with databases and datasets
"""

import pandas as pd
import random
import pymongo
import pickle
from os import environ
from typing import *
import mongomock
import json
import datetime
import logging
from pymongo import UpdateMany, UpdateOne
from dateutil.parser import parse as parse_date
import pymongo

from .task_type import TaskType
from .runnable_model_data import RunnableModel
from .run_config import Config

log = logging.getLogger("data_handling.py")
logging.basicConfig(level=logging.INFO)

class DatabaseConnector:
  """
  Manages calls to the database, switches between real MongoDB database and MongoMock depending on the environment and settings
  """

  def __init__(
      self,
      testing_mode: bool,
      insert_testing_models: bool,
      data_to_insert_by_default: Union[Dict[str, List[Dict]], None]=None) -> None:
    # use a mock DB if the app is not on k8s

    running_on_k8s = environ.get('K8S_DEPLOYMENT') is not None
    real_data_mode = running_on_k8s and not testing_mode
    log.info(f'real_data_mode: {real_data_mode}')
    if real_data_mode:
      self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    else:
      self.mongo_client = mongomock.MongoClient()

    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']
    self.experiments = self.db['experiments']

    if insert_testing_models:
      self._fill_with_testing_stuff()
    
    if data_to_insert_by_default is not None:
      self._insert_default_data_in_collections(data_to_insert_by_default)


  _DEFAULT_DUMP_PATH = './db_dump'
  def load_data_from_file(self, path: str = _DEFAULT_DUMP_PATH):
    log.info('Loading data from file')
    with open(path, 'rb') as file:
      data = pickle.load(file)
    self._insert_default_data_in_collections(data)
    log.info('Loaded data from file')


  def save_data_to_file(self, path: str = _DEFAULT_DUMP_PATH):
    log.info('Saving data to file')
    data = {
      'models': list(self.models.find()),
      'experiments': list(self.experiments.find())
    }
    with open(path, 'wb') as file:
      pickle.dump(data, file)
    log.info('Saved data to file')

  
  def _insert_default_data_in_collections(self, data: Dict[str, List[Dict]]):
    log.info(f'Inserting default data in {len(data)} collections')
    for collection, items in data.items():
      log.info(f'Inserting {len(items)} default items in {collection}')
      try:
        self.db[collection].insert_many(items)
      except Exception as e:
        log.warn(f'Got a default data write error: \n' + str(e))


  def _get_testing_models(self):
    return [
      {
        '_id': 'hf::gpt2',
        'owner': '',
        'name': 'gpt2',
        'source': 'hf',
        'first_tracked_on': 0,
        'last_tracked_on': 1,
        'tracking_history': [
          {
            'date': "Thisisadate",
            'hf_inference_api_supported': True,
            'available': True,
            'context_size': 1000,
            'price_prompt': 0,
            'price_completion': 0,
            'prompt_limit': 1000,
            'max_tokens_limit': 1000,
          }
        ]
      }
    ]


  def _check_if_experiment_matches_config(
      self,
      experiment: dict,
      config: Config) -> bool:
    return experiment['config'] == config.to_dict()

  
  def get_latest_evaluation_for_combination(
      self,
      model: RunnableModel,
      task_name: str,
      config: Config) -> Union[Dict, None]:
    latest_experiment = None
    model_task_experiments = list(self.experiments.find({
      'model_id': model._id,
      'task_type': task_name,
      'finished': True,
      'too_expensive': False,
      }))
    experiments_with_matching_configs = [
      experiment for experiment in model_task_experiments
      if self._check_if_experiment_matches_config(experiment, config)
    ] # type: List[Dict]

    # finds the latest experiment
    for experiment in experiments_with_matching_configs:
      current_experiment_date = parse_date(experiment['date'])
      if latest_experiment is None or current_experiment_date > parse_date(latest_experiment['date']):
        latest_experiment = experiment
    
    return latest_experiment


  def _fill_with_testing_stuff(self):
    try:
      self.models.insert_many(self._get_testing_models())
    except pymongo.errors.BulkWriteError as e:
      log.warn(f'Got a testing data write error: \n' + e.details['writeErrors'])

  # todo: verify columns?

  def get_models_available_for_evaluating(self) -> List[RunnableModel]:
    """
    Returns all models for now
    """
    db_models = self.models.find()

    models_to_return = []
    for model_obj in list(db_models): # had to make this one a list because otherwise Mongomock is non-deterministic, sometimes iterating and sometimes don't...
      models_to_return.append(RunnableModel(
        _id=model_obj['_id'],
        owner=model_obj['owner'],
        name=model_obj['name'],
        source=model_obj['source'],
        context_size=model_obj['tracking_history'][-1]['context_size'],
        hf_inferable=model_obj['tracking_history'][-1]['hf_inference_api_supported'],
        available=model_obj['tracking_history'][-1]['available'],
        price=max(model_obj['tracking_history'][-1]['price_completion'], model_obj['tracking_history'][-1]['price_prompt']),
        discount=model_obj['tracking_history'][-1].get('discount', 0.0)))
    return models_to_return
  

  def _make_run_id(self, model: RunnableModel, task_type: str, config: Config, experiment_date: str):
    parts = [
      str(task_type),
      str(model._id),
      json.dumps(config.to_dict()),
      experiment_date
    ]
    experiment_id = ''.join(parts)
    return experiment_id
  

  def create_experiment_stump(self, model: RunnableModel, task_type: str, config: Config, experiment_date: str) -> str:
    experiment_id = self._make_run_id(model, task_type, config, experiment_date)
    experiment_dict = {
      '_id': experiment_id,
      'date': experiment_date,
      'finished': False,
      'too_expensive': False,
      'model_id': model._id,
      'iterations': 1,
      'config': config.to_dict(),
      'notes': {},
      'task_type': task_type,
      'metrics': {},
      'outputs': []
    }
    result = self.experiments.insert_one(experiment_dict)
    assert result.inserted_id is not None, f'Failed inserting ID {experiment_id}'
    return experiment_id


  def get_unfinished_experiments(self, task_type: str) -> List[Dict]:
    return list(self.experiments.find({
      'task_type': task_type,
      'finished': False}))


  def get_experiment_from_id(self, experiment_id: str) -> Dict:
    return list(self.experiments.find({'_id': experiment_id}))[0]


  def mark_experiment_as_finished(self, _id, too_expensive):
    log.info(f'Marking experiment {_id} as finished (too_expensive={too_expensive})')
    self.experiments.update_one(
      {'_id': _id},
      {
        '$set':
        {
          'finished': True,
          'too_expensive': too_expensive
        }
      })


  def get_model_from_id(self, _id) -> Union[RunnableModel, None]:
    # todo: make it use propper cursors
    model_cursor = list(self.models.find({'_id': _id}))
    if len(model_cursor) == 0:
      log.error(f'Could not find model ID {_id}')
      return None
    else:
      model_obj = model_cursor[0]

    return RunnableModel(
        _id=model_obj['_id'],
        owner=model_obj['owner'],
        name=model_obj['name'],
        source=model_obj['source'],
        context_size=model_obj['tracking_history'][-1]['context_size'],
        hf_inferable=model_obj['tracking_history'][-1]['hf_inference_api_supported'],
        available=model_obj['tracking_history'][-1]['available'],
        price=max(model_obj['tracking_history'][-1]['price_completion'], model_obj['tracking_history'][-1]['price_prompt']),
        discount=model_obj['tracking_history'][-1].get('discount', 0.0))


  def append_output_to_experiment(self, experiment_id, output):
    self.experiments.update_one(
      {'_id': experiment_id},
      {'$push': {'outputs': output}})
    

  def append_many_outputs_to_experiments(self, experiment_id, outputs: List[dict]):
    self.experiments.update_one(
      {'_id': experiment_id},
      {'$push': {'outputs': { '$each': outputs }}}
    )


  def set_metrics_to_experiment(self, experiment_id, metrics):
    self.experiments.update_one(
      {'_id': experiment_id},
      {'$set': {'metrics': metrics}})
  

  def increment_counter_in_notes(
      self,
      experiment_id: str,
      notes_key: str):
    """Increments a value in the notes by 1. Sets the value to 1 if it does not exist yet
    This can be used for counting specific error occurences"""
    # todo: remake it to use only one DB request ;_;

    existing_notes = self.experiments.find_one(
      {'_id': experiment_id},
      {'notes': 1})['notes']
    
    # fixes previous version of notes to use dict instead of string
    if isinstance(existing_notes, str):
      log.warning(f'Clearing existing string notes for experiment {experiment_id} to use dict instead')
      existing_notes = {}
    
    existing_notes[notes_key] = existing_notes.get(notes_key, 0) + 1

    self.experiments.update_one(
      {'_id': experiment_id},
      {'$set': {'notes': existing_notes}})


  # saves new model if it is not tracked yet, adds one tracking entry if it is
  def save_model_tracking_properly(self, model_dict, dt):
    _id = model_dict['_id']
    tracking_record_data = {
      'date': dt,
      'hf_inference_api_supported': model_dict['hf_inference_api_supported'],
      'available': model_dict['available'],
      'context_size': model_dict['context'],
      'price_prompt': model_dict['price_prompt'],
      'price_completion': model_dict['price_completion'],
      'prompt_limit': model_dict['prompt_limit'],
      'max_tokens_limit': model_dict['max_tokens_limit'],
    }

    existing_model = self.models.find_one(_id)
    if existing_model is not None:
      self.models.update_one({'_id': _id},
        {'$push': {'tracking_history': tracking_record_data},
         '$set': {'last_tracked_on': dt}
         })
    else: 
      self.models.insert_one({
        '_id': _id,
        'owner': model_dict['owner'],
        'name': model_dict['name'],
        'source': model_dict['source'],
        'first_tracked_on': dt,
        'last_tracked_on': dt,
        'tracking_history': [tracking_record_data]
        })

  def get_unique_tracked_models(self) -> pd.DataFrame:
    # unique model - unique by owner, name and source
    #return pd.DataFrame(columns=self.columns_list)
    unique_models = self.tracking_history.find()
    df = pd.DataFrame(list(unique_models))
    df = df.drop_duplicates('original_name')
    return df
  
  def get_unique_tracked_hf_models(self) -> pd.DataFrame:
    #return pd.DataFrame(columns=self.columns_list)
    unique_hf_models = self.tracking_history.find({'source': 'hf'})
    df = pd.DataFrame(list(unique_hf_models))
    df = df.drop_duplicates('original_name')
    return df
  
  def save_models_tracking(self, models: pd.DataFrame):
    #return
    self.tracking_history.insert_many(models.to_dict('records'))


def _load_raw_reading_comprehension_data() -> Tuple[Dict, Dict]:
  log.info('Loading RC dataset')
  with open("./rc_dataset.txt", 'r') as file:
    dataset = json.load(file)
  rc_texts = dataset['texts']
  rc_questions = dataset['questions']
  return rc_texts, rc_questions


def _load_raw_science_questions_data() -> Dict[str, Tuple[bool, List[str]]]:
  """Fills Science Questions dataset with data of the following format:

  {
    input_code:
    {
      question, # true if the post is made by a bot
      option0,
      option1,
      option2,
      option3,
      answer_index # starts from 0, an integer
    }
  }
  """

  log.info('Loading science questions dataset')
  with open("./sciq.json", 'r') as file:
    dataset = json.load(file)
  return dataset


def _load_raw_bot_detection_data() -> Dict[str, Tuple[bool, List[str]]]:
  """Fills BOT_DETECTION_DATASET with data of the following format:

  {
    input_code:
    (
      bool, # true if the post is made by a bot

      [str, ...] # post history
    )
  }

  Input code is represented by user ID
  """

  log.info('Loading bot detection dataset')
  with open("./bot_or_not.json", 'r') as file:
    dataset = json.load(file)
  
  # keeps only necessary data, transform into the required format
  dataset = {
    dataset_entry['user_id']: (
      True if dataset_entry['human_or_bot'] == 'bot' else False,
      dataset_entry['post_history']
    )
    for dataset_entry in dataset
  }
  return dataset


def _load_raw_multiplication_data() -> Dict:
  """Returns a dict of {input_code: ("90*47=", 4230), ...}"""
  log.info('Loading basic math dataset')
  
  # Equations will be generated by first sampling a random range of values in [min_value, max_value]
  # Then, to generate the equations, each of these values will be taken as the left operand of the math operation, combined with each
  # of these values as the right operand
  # Because the amout of combinations equals to the square of the range of values, the code will sample sqrt(math_equations_to_generate) values
  # the input ID will have the format of {left_value}:{right_value}
  math_equations_to_generate = 5000
  min_value = 0
  max_value = 999
  values_to_sample = int(math_equations_to_generate ** 0.5)
  log.info(f'Sampling {values_to_sample} values in range [{min_value}, {max_value}]')

  final_equations = {}
  random_sampler = random.Random(11037)
  sampled_range = random_sampler.sample(range(min_value, max_value+1), values_to_sample)
  for left_value in sampled_range:
    for right_value in sampled_range:
      input_id = f'{left_value}:{right_value}'
      equation = f'{left_value}*{right_value}='
      answer = int(left_value * right_value)
      final_equations[input_id] = (equation, answer)
  
  return final_equations


def load_appropriate_dataset_for_task(task_type: TaskType) -> Any:
  """
  - For reading comprehension: returns a dict of {input_code: question text} and a dict of {input_code: question data}
  - For bot detection: returns a dict of {input_code: (is a bot, list of post history strings)}
  - For basic math: returns a dict of {input_code: (math expression like "90+47=", answer value), ...} 
  - For science questions, returns a dict of {input_code: a dictionary of question details that includes "answer_index" key}
  """
  appropriate_data_loader_for_task = {
    TaskType.READING_COMPREHENSION: _load_raw_reading_comprehension_data,
    TaskType.BOT_DETECTION: _load_raw_bot_detection_data,
    TaskType.MULTIPLICATION: _load_raw_multiplication_data,
    TaskType.SCIENCE_QUESTIONS: _load_raw_science_questions_data
  }.get(task_type, None)

  if appropriate_data_loader_for_task is not None:
    return appropriate_data_loader_for_task()
  else:
    raise TypeError(f"Unknown task: {task_type}")

