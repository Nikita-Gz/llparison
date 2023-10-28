import logging
import datetime
import time
import os
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_running.data_handling_for_experiment_running import DatabaseConnector
from model_running.task_type import TaskType
from experiment_task import Task

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
      },
      {
        '_id': 'hf::gpt2',
        'owner': '',
        'name': 'gpt2',
        'source': 'hf',
        'first_tracked_on': str(datetime.datetime.now()),
        'last_tracked_on': str(datetime.datetime.now()),
        'tracking_history': [
          {
            'date': str(datetime.datetime.now()),
            'hf_inference_api_supported': False,
            'available': True,
            'context_size': 1024,
            'price_prompt': 0,
            'price_completion': 0,
            'prompt_limit': 1000,
            'max_tokens_limit': 1000,
            'discount': 0.0
          }
        ]
      },
      {
        '_id': 'OpenRouter:mistralai:mistral-7b-instruct',
        'owner': 'mistralai',
        'name': 'mistral-7b-instruct',
        'source': 'OpenRouter',
        'first_tracked_on': str(datetime.datetime.now()),
        'last_tracked_on': str(datetime.datetime.now()),
        'tracking_history': [
          {
            'date': str(datetime.datetime.now()),
            'hf_inference_api_supported': False,
            'available': True,
            'context_size': 4096,
            'price_prompt': 0,
            'price_completion': 0,
            'prompt_limit': 4096,
            'max_tokens_limit': 4096,
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
  #outputs = list(db.experiments.find())[0]['outputs']
  #log.info(f'Loaded {len(outputs)} experiment outputs')

load_one_experiment_db_dump(db)

'''
task = Task(TaskType.MULTIPLICATION)
while True:
  task.run_task(
    db_connection=db,
    date=str(datetime.datetime.now()),
    cost_limit=None,
    db_cache_limit=500,
    path_to_save_db_on_update=DB_DUMP_FILE
  )
task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=500,
  path_to_save_db_on_update=DB_DUMP_FILE
)
'''

'''
task = Task(TaskType.BOT_DETECTION)

task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=60,
  path_to_save_db_on_update=DB_DUMP_FILE
)
task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=60,
  path_to_save_db_on_update=DB_DUMP_FILE
)
'''


task = Task(TaskType.READING_COMPREHENSION)
while True:
  task.run_task(
    db_connection=db,
    date=str(datetime.datetime.now()),
    cost_limit=None,
    db_cache_limit=100,
    path_to_save_db_on_update=DB_DUMP_FILE
  )
task.run_task(
  db_connection=db,
  date=str(datetime.datetime.now()),
  cost_limit=None,
  db_cache_limit=500,
  path_to_save_db_on_update=DB_DUMP_FILE
)
