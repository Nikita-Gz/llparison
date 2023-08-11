# todo: turn all db connector files into one?

import pandas as pd
import pymongo
from typing import *

from runnable_model_data import RunnableModel
from task_output import TaskOutput
from run_config import Config

class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(self) -> None:
    #return
    self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']
    self.experiments = self.db['experiments']

  # todo: verify columns?

  def get_models_available_for_evaluating(self) -> List[RunnableModel]:

    # todo: CHECK IF MODELS EXIST
    assert self.models.count_documents() > 0, 'No models are present!'

    #models_to_evaluate
    # gets the latest date of tracking saved
    latest_tracking_date = self.models.aggregate([{
      '$project': {
        'last_tracked': {
          '$max': 'tracking_history.date'
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
    }])[0]['last_tracked']

    # gets the model data at the last_tracked date
    db_models = self.models.find({
      'tracking_history.date': {'$eq': 1}
    },
    {
      'tracking_history.$': 1
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
