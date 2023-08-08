import pandas as pd
import pymongo
from typing import *

class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(self) -> None:
    #return
    self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']

  # todo: verify columns?

  def get_models(self) -> pymongo.CursorType:
    return self.models.find()

  def get_model_if_exists(self, _id: str):
    return self.models.find_one(_id)
  
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
