import logging

from model_data_loader import DatabaseConnector
from task import Task
from task_type import TaskType

log = logging.getLogger("test_driver.py")
logging.basicConfig(level=logging.INFO)


def get_testing_db():
  db = DatabaseConnector(
    testing_mode=True,
    insert_testing_models=False,
    data_to_insert_by_default=
    {
      'models':
      [
        {
          '_id': 'me:me:rc_test_model',
          'owner': 'me',
          'name': 'rc_test_model',
          'source': 'me',
          'first_tracked_on': '2020',
          'last_tracked_on': '2021',
          'tracking_history': [
            {
              'date': "2021",
              'hf_inference_api_supported': False,
              'available': True,
              'context_size': 10000,
              'price_prompt': 0,
              'price_completion': 0,
              'prompt_limit': 10000,
              'max_tokens_limit': 10000,
            }
          ]
        }
      ]
    }
  )
  return db


def print_database(db_connector: DatabaseConnector):
  collection_names = ['models', 'experiments']
  for collection_name in collection_names:
    collection = db_connector.db[collection_name]
    results = list(collection.find())
    to_display = 10
    log.info(f'Found {len(results)} results for {collection_name}, displaying up to {to_display}')
    for result in results[:to_display]:
      print(result)


def test_driver():
  # 0) create DB in testing mode
  # 1) fill DB with RC testing model
  # 2) create the task for RC
  # 3) run the task
  # 4) view the results

  db = get_testing_db()
  task = Task(TaskType.READING_COMPREHENSION)
  task.run_reworked_reading_comprehension(db, '2020:idkk', None)
  print_database(db)



if __name__ == "__main__":
  test_driver()
else:
  log.warn("This file is not supposed to be imported")