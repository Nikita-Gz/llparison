import string
import re
import logging
import pickle
from collections import Counter
from typing import *

from data_handling import DatabaseConnector
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
               validation_data: Union[None, Dict]=None,
               db_enabled=True,
               db_cache_limit=500,
               path_to_save_db_on_update: Union[str, None]=None,
               **kwargs) -> None:
    self.db_connection = db_connection
    self.experiment_id = experiment_id
    self.task = task
    self.db_enabled = db_enabled
    self.test_data = validation_data # type: Union[None, Dict]
    self.arguments = kwargs # kwargs are task-specific data
    self.path_to_save_db_on_update = path_to_save_db_on_update

    if existing_processed_outputs is None or len(existing_processed_outputs) == 0:
      log.info(f'Creating an empty processed outputs dict')
      self.processed_outputs = dict()
    else:
      log.info(f'Filling the existing processed outputs dict')
      self.processed_outputs = {output['input_code']:output for output in existing_processed_outputs}
      log.info(f'Filled with {len(self.processed_outputs)} outputs')
    
    self._cached_output_writes = []
    self._db_cache_limit = db_cache_limit
  

  def _compute_reading_comprehension_metrics(self):
    log.info(f'Computing metrics for reading comprehension')
    processed_output_values = self.processed_outputs.values()
    accuracy = sum([output['correct'] for output in processed_output_values]) / len(processed_output_values)

    answer_counts = list(Counter([output['interpreted_output'] for output in processed_output_values]).values())
    max_count = max(answer_counts)
    min_counts = min(answer_counts)
    answer_count_difference = max_count - min_counts

    answer_disproportion = 1 - (answer_count_difference / len(processed_output_values))

    unfit_answers = sum([output['interpreted_output'] not in ['A', 'B', 'C', 'D'] for output in processed_output_values]) / len(processed_output_values)
    metrics = {
      'accuracy': accuracy,
      'answer_disproportion': answer_disproportion,
      'unfit_answers': unfit_answers
    }
    log.info(f'Metrics: {metrics}')
    return metrics
  

  def _compute_bot_detection_metrics(self):
    log.info(f'Computing metrics for bot detection')
    processed_output_values = self.processed_outputs.values()
    accuracy = sum([output['correct'] for output in processed_output_values]) / len(processed_output_values)

    # counts true positives, false negatives, false positives
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    unfit_answers = 0
    for input_code, output in self.processed_outputs.items():
      correct_answer = self.test_data[input_code]
      model_answer = output['interpreted_output']

      if model_answer is None:
        unfit_answers += 1
        continue

      if correct_answer: # captures true positives and false negatives
        if model_answer: # captures true positive
          true_positives += 1
        else: # captures false negative
          false_negatives += 1
      else: # captures false positives and true negatives
        if model_answer: # captures false positive
          false_positives += 1
        else: # captures true negative
          true_negatives += 1
        
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1 = 2 * (precision * recall) / (precision + recall)
    unfit_answers_portion = unfit_answers / len(self.processed_outputs)
    metrics = {
      'recall': recall,
      'precision': precision,
      'f1': f1,
      'accuracy': accuracy,
      'unfit_answers': unfit_answers_portion
    }
    log.info(f'Metrics: {metrics}')
    return metrics
  

  def _dump_db_if_applicable(self):
    if self.path_to_save_db_on_update is not None:
      log.info(f'Dumping DB')
      self.db_connection.save_data_to_file(self.path_to_save_db_on_update)


  def _flush_the_cache(self):
    log.info(f'Writing {len(self._cached_output_writes)} outputs to DB')
    self.db_connection.append_many_outputs_to_experiments(self.experiment_id, self._cached_output_writes)
    self._dump_db_if_applicable()
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
    elif self.task == TaskType.BOT_DETECTION:
      metrics = self._compute_bot_detection_metrics()
    else:
      raise NotImplementedError(f'Metrics for task {self.task} are NYI')
    
    self.db_connection.set_metrics_to_experiment(self.experiment_id, metrics)
    self._dump_db_if_applicable()


  def _process_reading_comprehension_raw_output(self, raw_output: str, input_code: str) -> dict:
    """Processes the raw model output into an interpreted output format applicable for the RC task"""

    def get_reading_comprehension_answer_id_from_model_output(model_output: str) -> Union[int, None]:
      """The first letter of the output will be counted as the answer the model chose"""
      assert model_output is not None, "Model output is none"
      matches = re.findall(r'[a-zA-Z]', model_output) # type: List[str]
      if len(matches) == 0:
        return None
      first_letter = matches[0].upper()
      return EvaluationResultsCallback.alphabet2idx.get(first_letter, None)

    model_answer_id = get_reading_comprehension_answer_id_from_model_output(raw_output)
    correct_answer_letter = self.test_data[input_code]
    correct_answer_id = self.alphabet2idx[correct_answer_letter]
    model_answer_letter = self.idx2alphabet.get(model_answer_id, None)
    correct = model_answer_id is not None and model_answer_id == correct_answer_id
    processed_output = {
      'interpreted_output': model_answer_letter,
      'model_output': raw_output,
      'input_code': input_code,
      'correct': correct
    }
    return processed_output
  

  def _process_bot_detection_raw_output(self, raw_output: str, input_code: str) -> dict:
    """Processes the raw model output into an interpreted output format applicable for the bot detection task"""

    def get_bot_detection_answer_from_model_output(model_output: str) -> Union[bool, None]:
      """Returns True if the first letter appearing in the model output is "Y", False if "N", None in other case"""
      assert model_output is not None, "Model output is none"
      matches = re.findall(r'[yYnN]', model_output) # type: List[str]
      if len(matches) == 0:
        return None
      first_letter = matches[0].lower()
      return first_letter == 'y'

    interpreted_model_answer = get_bot_detection_answer_from_model_output(raw_output)
    correct_answer = self.test_data[input_code]
    if interpreted_model_answer is not None:
      correct = interpreted_model_answer == correct_answer
    else:
      correct = False
    processed_output = {
      'interpreted_output': interpreted_model_answer,
      #'expected_answer': correct_answer,
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
    elif self.task == TaskType.BOT_DETECTION:
      processed_output = self._process_bot_detection_raw_output(model_raw_output, input_code)
      self.processed_outputs[input_code] = processed_output
    else:
      raise NotImplementedError(f'Task {self.task} is NYI')
    
    if self.db_connection is not None:
      self._cache_or_write_output_to_db(processed_output)

  
  def increment_counter_in_notes(
      self,
      notes_key: str):
    """Increments a value in the notes by 1. Sets the value to 1 if it does not exist yet
    This can be used for counting specific error occurences"""
    if self.db_connection is not None:
      log.info(f'Incrementing notes key "{notes_key}"')
      self.db_connection.increment_counter_in_notes(self.experiment_id, notes_key)
    else:
      log.error(f'Tried incrementing notes key "{notes_key}", but no DB is attached')
