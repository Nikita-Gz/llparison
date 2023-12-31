import string
import re
import logging
import pickle
import numpy as np
from collections import Counter
from typing import *
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from .data_handling_for_experiment_running import DatabaseConnector
from .task_type import TaskType

log = logging.getLogger("eval_results_callback.py")
logging.basicConfig(level=logging.INFO)

# computes scores and saves perliminary results into a db
class EvaluationResultsCallback:
  """
  This class processes and records LLM's generation results to the database for a specified experiment and task, and it can compute and save the metrics
  """
  
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

    answer_counter = Counter([output['interpreted_output'] for output in processed_output_values if output['interpreted_output'] is not None])
    answer_counts = list(answer_counter.values())
    max_count = max(answer_counts)
    min_counts = min(answer_counts)
    answer_count_difference = max_count - min_counts

    biggest_answer_count_difference = (answer_count_difference / len(processed_output_values))

    unfit_answers = sum([output['interpreted_output'] not in ['A', 'B', 'C', 'D'] for output in processed_output_values]) / len(processed_output_values)
    metrics = {
      'accuracy': accuracy,
      'biggest_answer_count_difference': biggest_answer_count_difference,
      'unfit_answers': unfit_answers
    }
    log.info(f'Metrics: {metrics}')
    return metrics
  

  def _compute_science_questions_metrics(self):
    log.info(f'Computing metrics for science questions')
    processed_output_values = self.processed_outputs.values()
    accuracy = sum([output['correct'] for output in processed_output_values]) / len(processed_output_values)

    answer_counts = list(Counter([output['interpreted_output'] for output in processed_output_values]).values())
    max_count = max(answer_counts)
    min_counts = min(answer_counts)
    biggest_answer_count_difference = max_count - min_counts
    biggest_answer_count_difference = (biggest_answer_count_difference / len(processed_output_values))

    unfit_answers = sum([output['interpreted_output'] not in [1, 2, 3, 4] for output in processed_output_values]) / len(processed_output_values)
    metrics = {
      'accuracy': accuracy,
      'biggest_answer_count_difference': biggest_answer_count_difference,
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
  

  def _compute_multiplication_metrics(self):
    log.info(f'Computing metrics for multiplication')
    processed_output_values = self.processed_outputs.values()
    accuracy = sum([output['correct'] for output in processed_output_values]) / len(processed_output_values)

    # assembles float arrays of predictions and of true values for input codes that have an answer
    unfit_answers = 0
    preds = []
    true_values = []
    r2_score, mean_squared_error, mean_absolute_percentage_error
    for input_code, output in self.processed_outputs.items():
      prediction = output['interpreted_output']
      if prediction is None:
        unfit_answers += 1
        continue
      
      preds.append(float(prediction))
      true_values.append(float(self.test_data[input_code]))
    preds = np.array(preds)
    true_values = np.array(true_values)
    
    unfit_answers_portion = unfit_answers / len(self.processed_outputs)
    metrics = {
      #'R2': r2_score(true_values, preds),
      #'MAPE': mean_absolute_percentage_error(true_values, preds),
      #'RMSE': mean_squared_error(true_values, preds)**0.5,
      'MAE': mean_absolute_error(true_values, preds),
      'Median Absolute Error': np.median(np.abs(true_values - preds)),
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

    # goes through all processed outputs. Flushes DB cache beforehand

    metrics_computator = {
      TaskType.READING_COMPREHENSION: self._compute_reading_comprehension_metrics,
      TaskType.BOT_DETECTION: self._compute_bot_detection_metrics,
      TaskType.MULTIPLICATION: self._compute_multiplication_metrics,
      TaskType.SCIENCE_QUESTIONS: self._compute_science_questions_metrics
    }.get(self.task, None)
    if metrics_computator is None:
      raise NotImplementedError(f'Metrics for task {self.task} are NYI')
    metrics = metrics_computator()
    
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
  

  def _process_science_questions_raw_output(self, raw_output: str, input_code: str) -> dict:
    """Processes the raw model output into an interpreted output format applicable for the SQ task"""

    def get_science_questions_answer_id_from_model_output(model_output: str) -> Union[int, None]:
      assert model_output is not None, "Model output is none"
      matches = re.findall(r'\d+', model_output) # type: List[str]
      if len(matches) == 0:
        return None
      answer_id = int(matches[0])
      return answer_id

    model_answer_id = get_science_questions_answer_id_from_model_output(raw_output)
    correct_answer_id = self.test_data[input_code]
    correct = model_answer_id is not None and model_answer_id == correct_answer_id
    processed_output = {
      'interpreted_output': model_answer_id,
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


  def _process_multiplication_raw_output(self, raw_output: str, input_code: str) -> dict:
    """Processes the raw model output into an interpreted output format applicable for the multiplication task"""

    def get_multiplication_answer_from_model_output(model_output: str) -> Union[int, None]:
      """Returns the parsed int from the answer by looking for the first full integer in the output, ignoring commas"""
      assert model_output is not None, "Model output is none"
      matches = re.findall(r'\d+', model_output.replace(',', '')) # type: List[str]
      if len(matches) == 0:
        return None
      first_answer = matches[0]
      return int(first_answer)

    interpreted_model_answer = get_multiplication_answer_from_model_output(raw_output)
    correct_answer = self.test_data[input_code]

    if interpreted_model_answer is not None:
      correct = interpreted_model_answer == correct_answer
    else:
      correct = False
    processed_output = {
      'interpreted_output': interpreted_model_answer,
      'model_output': raw_output,
      'input_code': input_code,
      'correct': correct
    }
    return processed_output


  def record_output(self, model_raw_output: str, input_code: str, **kwargs) -> None:
    assert input_code not in self.processed_outputs.keys(), f'input code {input_code} is already recorded ({self.processed_outputs.get(input_code, None)})'

    # these functions convert model's raw output into the output appropriate for the task
    output_interpreter = {
      TaskType.READING_COMPREHENSION: self._process_reading_comprehension_raw_output,
      TaskType.BOT_DETECTION: self._process_bot_detection_raw_output,
      TaskType.MULTIPLICATION: self._process_multiplication_raw_output,
      TaskType.SCIENCE_QUESTIONS: self._process_science_questions_raw_output
    }.get(self.task, None)

    if output_interpreter is None:
      raise NotImplementedError(f'Task {self.task} is NYI')

    processed_output = output_interpreter(model_raw_output, input_code)
    self.processed_outputs[input_code] = processed_output
    
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
