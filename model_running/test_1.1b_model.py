import logging
import datetime
import time
import os

from model_data_loader import DatabaseConnector
from task import Task
from task_type import TaskType
from model_runner import ModelRunner

try:
  __file__
except NameError:
  __file__ = 'nofile'
log = logging.getLogger(os.path.basename(__file__))
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

DB_DUMP_FILE = './db_dump'
def load_one_experiment_db_dump(db: DatabaseConnector):
  db.load_data_from_file(DB_DUMP_FILE)
  outputs = list(db.experiments.find())[0]['outputs']
  log.info(f'Loaded {len(outputs)} experiment outputs')

load_one_experiment_db_dump(db)

task = Task(TaskType.READING_COMPREHENSION)


task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=500,
  save_db_on_cache_flush=DB_DUMP_FILE
)
