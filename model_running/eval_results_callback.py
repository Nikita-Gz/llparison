import string
import re
import logging
from typing import *

from model_data_loader import DatabaseConnector
from task_type import TaskType

log = logging.getLogger("eval_results_callback.py")
logging.basicConfig(level=logging.INFO)

# computes scores and saves perliminary results into a db
class EvaluationResultsCallback:
  alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
  idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}

  def __init__(self,
               db_connection: Union[DatabaseConnector, None],
               experiment_id: str,
               task: TaskType,
               existing_processed_outputs=None,
               validation_data=None,
               db_enabled=True,
               db_cache_limit=500,
               **kwargs) -> None:
    self.db_connection = db_connection
    self.experiment_id = experiment_id
    self.task = task
    self.db_enabled = db_enabled
    self.test_data = validation_data # type: Union[None, Dict]
    self.arguments = kwargs # kwargs are task-specific data

    if existing_processed_outputs is None or len(existing_processed_outputs) == 0:
      self.processed_outputs = dict()
    else:
      self.processed_outputs = existing_processed_outputs
    
    self._cached_output_writes = []
    self._db_cache_limit = db_cache_limit
  

  def _compute_reading_comprehension_metrics(self):
    processed_output_values = self.processed_outputs.values()
    accuracy = sum([output['correct'] for output in processed_output_values]) / len(processed_output_values)
    return {
      'accuracy': accuracy
    }
  

  def _flush_the_cache(self):
    log.info(f'Writing {len(self._cached_output_writes)} outputs to DB')
    self.db_connection.append_many_outputs_to_experiments(self.experiment_id, self._cached_output_writes)
    self._cached_output_writes = []


  def _cache_or_write_output_to_db(self, processed_output: Union[dict, None]):
    if self.db_connection is None:
      log.warning('Ignoring DB cache/write call as there is no DB')
      return
    
    # just write the cache if there is no processed output
    if processed_output is None:
      self._flush_the_cache()
    else:
      self._cached_output_writes.append(processed_output)
      if len(self._cached_output_writes) >= self._db_cache_limit:
        self._flush_the_cache()


  def finalize_evaluation(self):
    '''
    Call this when the evaluation is finished. This flushes the chached DB writes
    '''
    if self.db_connection is not None:
      self._cache_or_write_output_to_db(processed_output=None)


  def compute_and_save_metrics(self):
    if len(self._cached_output_writes) > 0:
      log.warning(f'There are {len(self._cached_output_writes)} unsaved writes when compute_and_save_metrics was called')

    # go through all processed outputs. Flushes DB cache beforehand
    if self.task == TaskType.READING_COMPREHENSION:
      metrics = self._compute_reading_comprehension_metrics()
    else:
      raise NotImplementedError(f'Metrics for task {self.task} are NYI')
    
    self.db_connection.set_metrics_to_experiment(self.experiment_id, metrics)


  def _get_reading_comprehension_answer_id_from_model_output(self, model_output: str) -> Union[int, None]:
    assert model_output is not None, "Model output is none"

    matches = re.findall(r'[a-zA-Z]', model_output)
    if len(matches) == 0:
      return None
    first_letter = matches[0].upper()
    return self.alphabet2idx.get(first_letter, None)


  # ranks the raw output and creates a dict that will go into the DB
  def _process_reading_comprehension_raw_output(self, raw_output: str, input_code: str) -> dict:
    model_answer_id = self._get_reading_comprehension_answer_id_from_model_output(raw_output)
    correct_answer_letter = self.test_data[input_code]['answer']
    correct_answer_id = self.alphabet2idx[correct_answer_letter]
    correct = model_answer_id is not None and model_answer_id == correct_answer_id
    processed_output = {
      'interpreted_output': model_answer_id,
      'model_output': raw_output,
      'input_code': input_code,
      'correct': correct
    }
    return processed_output


  def record_output(self, model_raw_output: str, input_code: str, **kwargs) -> None:
    assert input_code not in self.processed_outputs.keys(), f'input code {input_code} is already recorded ({self.processed_outputs.get(input_code, None)})'
    if self.task == TaskType.READING_COMPREHENSION:
      processed_output = self._process_reading_comprehension_raw_output(model_raw_output, input_code)
      self.processed_outputs[input_code] = processed_output
    else:
      raise NotImplementedError(f'Task {self.task} is NYI')
    
    if self.db_connection is not None:
      self._cache_or_write_output_to_db(processed_output)

