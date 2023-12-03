"""
This file defines the process for running one iteration over all possible experiment task types
"""

import logging
import datetime
import time
import os
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_running.data_handling_for_experiment_running import DatabaseConnector
from model_running.task_type import TaskType, task_type_int_to_str
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
      },
      {
        '_id': 'hf:PY007:TinyLlama-1.1B-intermediate-step-480k-1T',
        'owner': 'PY007',
        'name': 'TinyLlama-1.1B-intermediate-step-480k-1T',
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
            'prompt_limit': 2048,
            'max_tokens_limit': 2048,
            'discount': 0.0
          }
        ]
      },
      {
        '_id': 'OpenRouter:huggingfaceh4:zephyr-7b-beta',
        'owner': 'huggingfaceh4',
        'name': 'zephyr-7b-beta',
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


DB_DUMP_FILE = './db_dump_'
def load_db_dump(db: DatabaseConnector):
  if os.path.isfile(DB_DUMP_FILE):
    log.info(f'Loading DB dump file at {DB_DUMP_FILE}')
    db.load_data_from_file(DB_DUMP_FILE)
  else:
    log.warning(f'No DB dump file found at {DB_DUMP_FILE}')
load_db_dump(db)


task_types_to_test = list(task_type_int_to_str.keys())
experiments_date = str(datetime.datetime.now())
log.info(f'Running experimentation iteration at {experiments_date} on the following tasks: {task_types_to_test}')

for task_type in task_types_to_test:
  log.info(f'Running the followint experiment: {task_type_int_to_str[task_type]}')
  task = Task(task_type=task_type)

  task.run_task(
    db_connection=db,
    date=experiments_date,
    cost_limit=None,
    db_cache_limit=500,
    path_to_save_db_on_update=DB_DUMP_FILE
  )

  log.info(f'Done running experiment')
