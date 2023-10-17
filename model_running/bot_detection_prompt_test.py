from typing import *
import json
import logging
import datetime

from data_handling import DatabaseConnector
from task import Task
from task_type import TaskType
from prompt_constructor import UniversalTokenizer, PromptConstructor

log = logging.getLogger("bot detection test.py")
logging.basicConfig(level=logging.INFO)

db = DatabaseConnector(
  testing_mode=True,
  insert_testing_models=False,
  data_to_insert_by_default={
    'models': [
      {
        '_id': 'hf:bigscience:bloomz-1b1',
        'owner': 'bigscience',
        'name': 'bloomz-1b1',
        'source': 'hf',
        'first_tracked_on': str(datetime.datetime.now()),
        'last_tracked_on': str(datetime.datetime.now()),
        'tracking_history': [
          {
            'date': str(datetime.datetime.now()),
            'hf_inference_api_supported': False,
            'available': True,
            'context_size': 2048,
            'price_prompt': 0,
            'price_completion': 0,
            'prompt_limit': 1000,
            'max_tokens_limit': 1000,
            'discount': 0.0
          }
        ]
      }
    ]
  }
)

BOT_DETECTION_DATASET = {} # type: Dict[str, Tuple[bool, List[str]]]
def _load_raw_bot_detection_data():
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
  
  # keeps only necessary keys
  global BOT_DETECTION_DATASET
  BOT_DETECTION_DATASET = {
    dataset_entry['user_id']: (
      True if dataset_entry['human_or_bot'] == 'bot' else False,
      dataset_entry['post_history']
    )
    for dataset_entry in dataset
  }
  a = [value for key, value in BOT_DETECTION_DATASET.items() if value[0] == False]
  return 11037
#_load_raw_bot_detection_data()


def _create_bot_detection_prompt(
    bot_detection_dataset: Dict[str, Tuple[bool, List[str]]],
    prompt_size: int) -> Dict[str, Tuple[bool, str]]:
  pass

task = Task(TaskType.BOT_DETECTION)

task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=500,
  path_to_save_db_on_update=None
)
