from enum import Enum
import json
import os
from typing import *
import string
import tiktoken
import logging
import re
import random
import mongomock
import datetime


from model_runner import ModelRunner
from model_data_loader import DatabaseConnector
from runnable_model_data import RunnableModel
from run_config import Config
from task_output import TaskOutput
from task_type import TaskType, task_type_int_to_str
from cost_callback import CostCallback
from eval_results_callback import EvaluationResultsCallback

# todo: !! custom tasks must be saved to db and loaded from it

log = logging.getLogger("task.py")
logging.basicConfig(level=logging.INFO)


class Task:
  alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
  idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}


  def __init__(self, task_type: TaskType) -> None:
    self.type = task_type

  def is_model_applicable_for_the_task(self, model: RunnableModel):
    # todo: make this work
    return True


  def _construct_rc_prompt(self, text, question_dict):
    question_text = question_dict['question']
    options_text = ''
    for i, option_text in enumerate(question_dict['options']):
      letter = self.idx2alphabet[i]
      options_text += f'{letter}) {option_text}\n'
    options_text = options_text.strip()
    final_text = ''.join([
      'Text:\n', text,
      '\nQuestion: ', question_text,
      '\nOptions:\n', options_text,
      '\nThe correct answer is the letter:'])
    return final_text
  

  def _load_raw_reading_comprehension_data(self) -> Tuple[Dict, Dict]:
    log.info('Loading RC dataset')
    with open("./rc_dataset.txt", 'r') as file:
      dataset = json.load(file)
    rc_texts = dataset['texts']
    rc_questions = dataset['questions']
    return rc_texts, rc_questions


  # loads RACE data as prompts.
  # Returns it as a dict of {question_id: prompt}
  def _prepare_reading_comprehension_prompts(self, rc_texts: Dict[str, str], rc_questions: Dict[str, Dict], excluded_question_ids: set) -> Dict:
    log.info('Preparing RC prompts')
    prepared_prompts = dict()
    excluded_count = 0
    for question_id, question_dict in rc_questions.items():
      if question_id in excluded_question_ids:
        excluded_count += 1
        continue
      question_context_text = rc_texts[question_dict['text_id']]
      prompt = self._construct_rc_prompt(question_context_text, question_dict)
      prepared_prompts[question_id] = prompt
    log.info(f'Loaded {len(prepared_prompts)} questions ({excluded_count} were excluded)')
    
    return prepared_prompts


  def _get_reading_comprehension_answer_id_from_model_output(self, model_output: str) -> Union[int, None]:
    assert model_output is not None, "Model output is none"

    matches = re.findall(r'[a-zA-Z]', model_output)
    if len(matches) == 0:
      return None
    first_letter = matches[0].upper()
    return self.alphabet2idx.get(first_letter, None)


  def _get_unfinished_experiment_if_any(self, db_connector: DatabaseConnector):
    task_name = task_type_int_to_str[self.type]
    unfinished_experiments = db_connector.get_unfinished_experiments(task_name)
    experiment_count = len(unfinished_experiments)
    log.info(f'Got {experiment_count} unfinished experiments for task {task_name} from DB')
    if experiment_count == 0:
      return None
    randomly_chosen_experiment = random.choice(unfinished_experiments)
    log.info(f'Returning experiment {randomly_chosen_experiment}')
    return randomly_chosen_experiment


  def _get_llm_config_combinations(self, models_for_evaluating: List[RunnableModel]) -> List[Tuple[RunnableModel, Config]]:
    def get_combinations_for_rc():
      def get_configs_for_llm(model: RunnableModel) -> List[Config]:
        config1 = Config()
        config1.set_parameter('temperature', 0.5)
        config1.set_parameter('top-p', 0.5)
        return [config1]
      
      combinations = [] # type: List[Tuple[RunnableModel, Config]]
      for model in models_for_evaluating:
        combinations_for_model = []
        for config in get_configs_for_llm(model):
          combinations_for_model.append((model, config))
        log.info(f'Got {len(combinations_for_model)} combinations for model {model._id}')
        combinations.extend(combinations_for_model)
      return combinations
    
    '''
    def get_combinations_for_rc_test():
      combinations = [] # type: List[Tuple[RunnableModel, Config]]
      for model in models_for_evaluating:
        combinations.append((model, Config()))
      log.info(f'Got {len(combinations)} combinations for RC testing model')
      return combinations
    '''

    if self.type == TaskType.READING_COMPREHENSION:
      return get_combinations_for_rc()
    else:
      raise NotImplemented(f'Unknown task type {self.type}')


  def _convert_date_to_datetime(self, date: Union[str, datetime.datetime]) -> datetime.datetime:
    if isinstance(date, str):
      return datetime.datetime.strptime(date)
    elif isinstance(date, datetime.datetime):
      return date
    else:
      raise TypeError(f'Invalid date type ({type(date)})')


  def _pick_out_combinations_needing_evaluations(
      self,
      possible_combinations: List[Tuple[RunnableModel, Config]],
      db_connector: DatabaseConnector,
      current_date: datetime.datetime) -> List[Tuple[RunnableModel, Config]]:
    # picks out combinations that either were never tested,
    # or are hosted by OpenRouter and haven't been tested in a while
    PROPRIETARY_MODEL_TIMEOUT = datetime.timedelta(days=31)
    PROPRIETARY_SOURCES_LIST = ['OpenRouter']
    combinations_to_evaluate = []
    for considered_combination in possible_combinations:
      model, configuration = considered_combination
      model_source = model.source
      latest_experiment = db_connector.get_latest_evaluation_for_combination(model, task_type_int_to_str[self.type], configuration)
      if latest_experiment is None:
        combinations_to_evaluate.append(considered_combination)
      elif model_source not in PROPRIETARY_SOURCES_LIST:
        combinations_to_evaluate.append(considered_combination)
      elif (current_date - self._convert_date_to_datetime(latest_experiment['date'])) > PROPRIETARY_MODEL_TIMEOUT:
        combinations_to_evaluate.append(considered_combination)
    return combinations_to_evaluate


  # returns a tuple of experiment ID and evaluation llm-config combination
  def _create_new_experiment(self, db_connection: DatabaseConnector, date: datetime.datetime):
    log.info(f'Creating a new experiment record for task {self.type} at {date}')
    models_for_evaluating = db_connection.get_models_available_for_evaluating()
    log.info(f'Got {len(models_for_evaluating)} models up for evaluations')

    llm_config_combinations = self._get_llm_config_combinations(models_for_evaluating)
    log.info(f'Got {len(llm_config_combinations)} total combinations')
    
    assert len(llm_config_combinations) != 0, f'No possible LLM-config combinations'
    combinations_up_for_evaluation = self._pick_out_combinations_needing_evaluations(
      llm_config_combinations,
      db_connection,
      date)
    
    if len(combinations_up_for_evaluation) == 0:
      log.info(f'No combinations are up for evaluation in task {self.type} at date {date}')

    combination_for_evaluation = random.choice(combinations_up_for_evaluation)
    experiment_id = db_connection.create_experiment_stump(
      model=combination_for_evaluation[0],
      task_type=task_type_int_to_str[self.type],
      config=combination_for_evaluation[1],
      experiment_date=date)
    return experiment_id, combination_for_evaluation


  # returns the llm-config combination
  def _recover_unfinished_experiment_llm_config_combination(self, db_connection: DatabaseConnector, experiment: dict):
    log.info(f'Recovering experiment {experiment}')
    model_id = experiment['model_id']
    model = db_connection.get_model_from_id(model_id)
    config = Config(experiment['config'])
    return (model, config)
  

  def _compute_prompts_cost(self, prompts: List[str], model_runner: ModelRunner) -> float:
    if model_runner.model.price == 0:
      total_cost = 0
    else:
      token_count = model_runner.count_tokens(prompts)
      log.info(f'Total token count is {token_count}')
      total_cost = model_runner.model.price * token_count

    log.info(f'Total token cost is {total_cost}')
    return total_cost



  def run_reworked_reading_comprehension(
      self,
      db_connection: DatabaseConnector,
      date: datetime.datetime,
      cost_limit=None,
      db_cache_limit=500,
      cost_callback: Union[CostCallback, None] = None):
    '''
    1) Check for unfinished experiments in DB (pick a random one if there are)
    2) Get the list of known models
    3) Create a list of possible LLM-config combiations
    4) Create a list of combinations that should be evaluated
    5) Pick one
    6) Create an experiment record
    7) Prepare the testing data
    7) Estimate and check the cost, mark experiment as finished if its too expensive
    8) For eachtest, run it on the model
    9) record results
    '''

    # use llm-config combination from an unfinished experiment, or create a new one
    unfinished_experiment = self._get_unfinished_experiment_if_any(db_connection)
    already_completed_outputs = []
    if unfinished_experiment is None:
      experiment_id, combination_for_evaluation = self._create_new_experiment(db_connection, date)
    else:
      experiment_id = unfinished_experiment['_id']
      combination_for_evaluation = self._recover_unfinished_experiment_llm_config_combination(
        db_connection,
        unfinished_experiment)
      already_completed_outputs = unfinished_experiment['outputs']
    model, config = combination_for_evaluation
    
    rc_texts, rc_questions = self._load_raw_reading_comprehension_data()
    prompts_dict = self._prepare_reading_comprehension_prompts(
      rc_texts, rc_questions,
      excluded_question_ids=set([output['input_code'] for output in already_completed_outputs]))
    runner = ModelRunner(model, config)

    # process the cost
    if cost_limit is not None:
      total_cost = self._compute_prompts_cost(list(prompts_dict.values()), runner)
      if total_cost > cost_limit:
        log.error(f'Cost is too high, aborting experiment {experiment_id} and marking it as finished')
        db_connection.mark_experiment_as_finished(experiment_id, too_expensive=True)
        return
      elif cost_callback is not None:
        cost_callback.register_estimated_initial_cost(total_cost)
        log.info('Registering the cost with the callback')
      log.info('The cost is acceptable')
    else:
      log.info('Cost limit is none')

    #_get_reading_comprehension_answer_id_from_model_output
    evaluation_callback = EvaluationResultsCallback(
      db_connection=db_connection,
      experiment_id=experiment_id,
      task=self.type,
      existing_processed_outputs=already_completed_outputs,
      validation_data=rc_questions,
      db_enabled=True,
      db_cache_limit=db_cache_limit)
    try:
      runner.run_model(prompts_dict, callback=evaluation_callback)
      evaluation_callback.finalize_evaluation()
    except Exception as e:
      log.error(e)
      raise e
    
    # checks if all evaluations were completed, save metrics if so, complain and die if not
    all_input_codes = set(rc_questions.keys())
    processed_input_codes = set([
      output['input_code'] for output in db_connection.get_experiment_from_id(experiment_id)['outputs']])
    if all_input_codes == processed_input_codes:
      evaluation_callback.compute_and_save_metrics()
      db_connection.mark_experiment_as_finished(experiment_id, too_expensive=False)
    else:
      log.error(f'Not all input codes were processed, the experiment was not marked as finished: {len(all_input_codes.difference(processed_input_codes))} unprocessed inputs')


  def run_reading_comprehension(self, model: RunnableModel, config, cost_check_callback=None) -> Union[TaskOutput, None]:
    rc_texts, rc_questions = self._load_reading_comprehension_data()

    # todo: make the preprocessing code for the specific model on a specific task
    # creates a single prompt from the text and question
    def prompt_constructor(text, question_dict):
      question_text = question_dict['question']
      options_text = ''
      for i, option_text in enumerate(question_dict['options']):
        letter = self.idx2alphabet[i]
        options_text += f'{letter}) {option_text}\n'
      options_text = options_text.strip()
      final_text = ''.join(['Text:\n', text, '\nQuestion: ', question_text, '\nOptions:\n', options_text, '\nAnswer:'])
      return final_text

    runner = ModelRunner(model, config)

    # cost estimator
    # todo: make it a function
    log.info('Running token estimator')
    if cost_check_callback is not None:
      prompts = []
      for i, question_id in enumerate(rc_questions):
        if i % 1000 == 0:
          log.info(f'{i / len(rc_questions)}%')
        question_dict = rc_questions[question_id]
        text = rc_texts[question_dict['text_id']]
        prompts.append(prompt_constructor(text, question_dict))
      token_count = runner.count_tokens(prompts)
      cost = model.price * token_count
      cost_is_appropriate = cost_check_callback(cost)
      log.info(f'{token_count} at {cost} cost')
      if not cost_is_appropriate:
        log.warning(f'Skipping model {model._id} because the cost is too high ({cost})')
        return None

    model_outputs = []
    metrics = {'accuracy': 0}
    interpreted_outputs = []
    input_codes = []
    errors = []

    # evaluator. First - get model responses. Second - evaluate them
    # WIP - continue from here. Account for exceptions!
    log.info('Running evaluator')
    completed_evaluations = 0
    for i, question_id in enumerate(rc_questions):
      log.info(f'Running question {i} out of {len(rc_questions)}')
      input_codes.append(question_id)
      question_dict = rc_questions[question_id]
      answer_id = self.alphabet2idx[question_dict['answer']]
      text = rc_texts[question_dict['text_id']]
      prompt = prompt_constructor(text, question_dict)

      try:
        model_output = runner.run_model(prompt)
      except Exception as e:
        errors.append(str(e))
        continue
      model_answer_id = self._get_answer_id_from_model_output(model_output)

      if model_answer_id is not None and model_answer_id == answer_id:
        metrics['accuracy'] += 1

      model_outputs.append(model_output)
      interpreted_outputs.append(model_answer_id)
      errors.append(None)
      completed_evaluations += 1

    metrics['accuracy'] = metrics['accuracy'] / completed_evaluations
    return TaskOutput(metrics, model_outputs, interpreted_outputs, input_codes)

  # returns a list of metrics, outputs
  def run_task(
      self,
      db_connection: DatabaseConnector,
      date: datetime.datetime,
      cost_limit=None,
      db_cache_limit=500) -> TaskOutput:
    if self.type == TaskType.READING_COMPREHENSION:
      log.info(f'Running reading comprehension on date {date} with cost limit {cost_limit}')
      return self.run_reworked_reading_comprehension(db_connection, date, cost_limit, db_cache_limit=db_cache_limit)
    else:
      raise NotImplementedError(f'Tried running an unsupported task type, {self.type}')



