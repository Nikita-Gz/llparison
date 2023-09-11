import pandas as pd
import pymongo
import mongomock
import logging
import os
from os import environ
from typing import *

log = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(level=logging.INFO)

class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(
      self,
      testing_mode: bool,
      insert_testing_models: bool, # NYI
      data_to_insert_by_default: Union[Dict[str, List[Dict]], None]) -> None:
    
    running_on_k8s = environ.get('K8S_DEPLOYMENT') is not None
    real_data_mode = running_on_k8s and not testing_mode
    log.info(f'real_data_mode: {real_data_mode}')
    if real_data_mode:
      self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    else:
      self.mongo_client = mongomock.MongoClient()

    self.db = self.mongo_client["llparison_db"]
    self.models = self.db['models']

    if data_to_insert_by_default is not None:
      self._insert_default_data_in_collections(data_to_insert_by_default)


  def _insert_default_data_in_collections(self, data: Dict[str, List[Dict]]):
    for collection, items in data.items():
      try:
        self.db[collection].insert_many(items)
      except Exception as e:
        log.warn(f'Got a default data write error: \n' + str(e))


  # todo: verify columns?

  def get_models(self) -> pymongo.CursorType:
    return self.models.find()

  def get_model_if_exists(self, _id: str):
    return self.models.find_one(_id)
  

  def save_models(self, models_dicts: List[Dict]):
    self.models.insert_many(models_dicts)


  # saves new model if it is not tracked yet, adds one tracking entry if it is
  def save_model_tracking_properly(self, model_dict, dt):
    _id = model_dict['_id']
    log.info(f'Saving model tracking for {_id}')
    tracking_record_data = model_dict['tracking_history'][0]

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
