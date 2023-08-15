# todo: turn all db connector files into one?

import pandas as pd
import pymongo
from os import environ
from typing import *
import mongomock

from runnable_model_data import RunnableModel
from task_output import TaskOutput
from run_config import Config

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


  def _get_testing_models(self):
    return [
      {
        '_id': 'hf:gpt2',
        'owner': 'Open AI',
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

  
  def _fill_with_testing_stuff(self):
    self.models.insert_many(self._get_testing_models())

  # todo: verify columns?

  def get_models_available_for_evaluating(self) -> List[RunnableModel]:

    # todo: CHECK IF MODELS EXIST
    assert self.models.count_documents({}) > 0, 'No models are present!'

    #models_to_evaluate
    # gets the latest date of tracking saved
    latest_tracking_date = self.models.aggregate([{
      '$project': {
        'last_tracked': {
          '$max': '$tracking_history.date'
        }
      }
    },
    {
      "$sort": {
        "last_tracked": -1
      }
    },
    {
      '$limit': 1
    }]).next()['last_tracked']

    #[0]['last_tracked']

    # gets the model data at the last_tracked date
    '''db_models = self.models.find({
      'tracking_history.date': {'$eq': latest_tracking_date}
    },
    {
      'tracking_history.$': 1
    })'''
    # the one above uses positional projection, but it's not supported in mongomock
    db_models = self.models.find({
      'tracking_history.date': {'$eq': latest_tracking_date}
    })

    models_to_return = []
    for model_obj in db_models:
      models_to_return.append(RunnableModel(
        _id=model_obj['_id'],
        owner=model_obj['owner'],
        name=model_obj['name'],
        source=model_obj['source'],
        context_size=model_obj['tracking_history'][0]['context_size'],
        hf_inferable=model_obj['tracking_history'][0]['hf_inference_api_supported'],
        available=model_obj['tracking_history'][0]['available'],
        price=model_obj['tracking_history'][0]['price_completion']))
    return models_to_return
  

  def save_run(self, model: RunnableModel, task_type: int, iterations: int, config: Config, experiment_outputs: List[TaskOutput], experiment_date: str):
    # todo: redo to use saving of several runs in one experiment
    experiment_dict = {
      'date': experiment_date,
      'model_id': model._id,
      'iterations': iterations,
      'config': config.to_dict(),
      'notes': '',
      'task_type': task_type,
      'metrics': [output.metrics for output in experiment_outputs],
      'model_outputs': [output.model_outputs for output in experiment_outputs],
      'interpreted_outputs': [output.interpreted_outputs for output in experiment_outputs],
      'input_codes': [output.input_codes for output in experiment_outputs],
    }
    self.experiments.insert_one(experiment_dict)

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
