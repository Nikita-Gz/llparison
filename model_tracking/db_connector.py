import pandas as pd
import pymongo

class DatabaseConnector:
  columns_list = ['first_tracked_on', 'last_tracked_on', 'available', 'original_name', 'owner', 'name', 'price_prompt', 'ff_inference_api_supported', 'source', 'price_completion', 'context', 'prompt_limit', 'max_tokens_limit']

  def __init__(self) -> None:
    #return
    self.mongo_client = pymongo.MongoClient("mongodb://mongodb/", username='root', password='root')
    self.db = self.mongo_client["llparison_db"]
    self.tracking_history = self.db['tracking_history']

  # todo: verify columns?

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
