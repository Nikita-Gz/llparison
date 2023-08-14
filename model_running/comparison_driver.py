# this file drives the comparison process on it's own

import time
import datetime

from task import Task, TaskType
from runnable_model_data import RunnableModel
from model_data_loader import DatabaseConnector
from run_config import Config
from task_output import TaskOutput

def only_free_cost_check_callback(cost: float):
  return cost == 0

def drive():
  db_connector = DatabaseConnector()

  timestamp = datetime.datetime.now()

  print(f'Running comparison at {timestamp}')
  
  # 1) load applicable models
  # 2) go over tasks
  # 3) get a list of models runnable on the task
  # 4) create a combination of configs
  # 5) run on the task
  # 6) save the results

  evaluatable_models = db_connector.get_models_available_for_evaluating()
  tasks_to_run = [Task(TaskType.READING_COMPREHENSION)]

  for task in tasks_to_run:
    for model in evaluatable_models:
      if task.is_model_applicable_for_the_task(model):
        continue
      
      configs = [Config()]
      for config in configs:
        task_output = task.run_task(model, config, only_free_cost_check_callback)
        if task_output is None:
          print('None output')
          continue
        db_connector.save_run(model, task.type, 1, config, [task_output], timestamp)
    



  time.sleep(60*60*24)
  pass

if __name__ == '__main__':
  drive()
