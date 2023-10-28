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
from dateutil.parser import parse as parse_date
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_running.model_runner import ModelRunner
from model_running.data_handling_for_experiment_running import DatabaseConnector, load_appropriate_dataset_for_task
from model_running.runnable_model_data import RunnableModel
from model_running.run_config import Config
from model_running.task_type import TaskType, task_type_int_to_str, new_tokens_limit_per_task_type_int
from model_running.cost_callback import CostCallback
from model_running.eval_results_callback import EvaluationResultsCallback
from model_running.prompt_constructor import PromptConstructor, UniversalTokenizer, NAME_OF_MULTIPLICATION_PROMPT_WITH_EXAMPLES

# todo: !! custom tasks must be saved to db and loaded from it

log = logging.getLogger("task.py")
logging.basicConfig(level=logging.INFO)


class Task:
  alphabet2idx = {letter:i for i, letter in enumerate(string.ascii_uppercase)}
  idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}


  def __init__(self, task_type: TaskType) -> None:
    self.type = task_type

  def is_model_applicable_for_the_task(self, model: RunnableModel):
    # todo: implement this
    return True


  # TODO: combine code that prepares datasets for different tasks into one flexible funciton
  def _load_and_prepare_reading_comprehension_prompts(
      self,
      excluded_input_ids: set,
      configuration: Config,
      model: RunnableModel) -> Tuple[Dict, int, int, Dict]:
    """Loads RACE data as prompts, excluding a set of IDs.
      Returns the dataset as a dict of {question_id: prompt}, total token count, number of cut prompts, and validation data for each input code"""
    log.info('Preparing RC prompts')

    rc_texts, rc_questions = load_appropriate_dataset_for_task(self.type)

    # converts the returned rc_texts (dict of texts by input code) and rc_questions (dict of question data by input code)
    # into the following format:
    # {
    #   input_code: prompt_text
    # }
    prepared_prompts = dict()
    excluded_count = 0
    prompt_constructor = PromptConstructor(
      model=model,
      task_type=self.type,
      configuration_dict=configuration.to_dict())
    total_token_count = 0
    total_prompts_cut = 0
    for question_id, question_dict in rc_questions.items():
      if question_id in excluded_input_ids:
        excluded_count += 1
        continue
      question_context_text = rc_texts[question_dict['text_id']]
      prompt, token_count, cut_by_n_tokens = prompt_constructor.construct_prompt(
        context_text=question_context_text,
        question_dict=question_dict)
      prepared_prompts[question_id] = prompt
      total_token_count += token_count
      if cut_by_n_tokens > 0:
        total_prompts_cut += 1
    log.info(f'Loaded {len(prepared_prompts)} questions, ({excluded_count} were excluded, {total_prompts_cut} were cut)')
    
    validation_data = {input_code: question['answer'] for input_code, question in rc_questions.items()}
    return prepared_prompts, total_token_count, total_prompts_cut, validation_data


  def _load_and_prepare_bot_detection_prompts(
      self,
      excluded_input_ids: set,
      configuration: Config,
      model: RunnableModel) -> Tuple[Dict, int, int, Dict]:
    """Loads bot detection data as prompts, excluding a set of IDs.
      Returns the dataset as a dict of {input_id: prompt}, total token count, number of cut prompts, and validation data for each input code"""
    log.info('Preparing bot detection prompts')

    bot_detection_dataset = load_appropriate_dataset_for_task(self.type)
    prepared_prompts = dict()
    excluded_count = 0
    cut_posts_count = 0
    total_token_count = 0
    prompt_constructor = PromptConstructor(
      model=model,
      task_type=self.type,
      configuration_dict=configuration.to_dict())
    for i, (input_id, (_, posts)) in enumerate(bot_detection_dataset.items()):
      if input_id in excluded_input_ids:
        excluded_count += 1
        continue

      prompt_text, token_count, posts_cut = prompt_constructor.construct_prompt(post_history=posts)
      prepared_prompts[input_id] = prompt_text
      total_token_count += token_count
      if posts_cut > 0:
        cut_posts_count += 1
      
      if i % 100 == 0:
        log.info(f'Processed bot detection prompt {i} out of {len(bot_detection_dataset)}')

    log.info(f'Loaded {len(prepared_prompts)} prompts, ({excluded_count} were excluded, {cut_posts_count} were cut, {total_token_count} total tokens)')
    
    validation_data = {input_code: is_bot for input_code, (is_bot, _) in bot_detection_dataset.items()}
    return prepared_prompts, total_token_count, cut_posts_count, validation_data


  def _load_and_prepare_multiplication_prompts(
      self,
      excluded_input_ids: set,
      configuration: Config,
      model: RunnableModel) -> Tuple[Dict, int, int, Dict]:
    """Loads multiplication data as prompts, excluding a set of IDs.
      Returns the dataset as a dict of {input_id: prompt}, total token count, number of cut prompts, and validation data for each input code"""
    log.info('Preparing multiplication prompts')

    multiplication_dataset = load_appropriate_dataset_for_task(self.type)
    prepared_prompts = dict()
    validation_data = dict()
    excluded_count = 0
    cut_posts_count = 0
    total_token_count = 0
    prompt_constructor = PromptConstructor(
      model=model,
      task_type=self.type,
      configuration_dict=configuration.to_dict())
    for i, (input_id, (expression, answer)) in enumerate(multiplication_dataset.items()):
      validation_data[input_id] = answer

      if input_id in excluded_input_ids:
        excluded_count += 1
        continue

      prompt_text, token_count, _ = prompt_constructor.construct_prompt(math_expression=expression)
      prepared_prompts[input_id] = prompt_text
      total_token_count += token_count
      
      if i % 100 == 0:
        log.info(f'Processed multiplication prompt {i} out of {len(multiplication_dataset)}')

    log.info(f'Loaded {len(prepared_prompts)} prompts, ({excluded_count} were excluded, {cut_posts_count} were cut, {total_token_count} total tokens)')
    return prepared_prompts, total_token_count, cut_posts_count, validation_data


  def _load_and_prepare_prompts_for_task(
      self,
      excluded_input_ids: set,
      configuration: Config,
      model: RunnableModel) -> Tuple[Dict, int, int, Dict]:
    """Loads data as prompts, excluding a set of IDs.
      Returns the dataset as a dict of {input_id: prompt}, total token count, the number of cut prompts, and validation data for each input code"""
    prompt_preparators = {
      TaskType.READING_COMPREHENSION: self._load_and_prepare_reading_comprehension_prompts,
      TaskType.BOT_DETECTION: self._load_and_prepare_bot_detection_prompts,
      TaskType.MULTIPLICATION: self._load_and_prepare_multiplication_prompts,
    }
    return prompt_preparators[self.type](
      excluded_input_ids=excluded_input_ids,
      configuration=configuration,
      model=model)


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
    log.info(f'Returning experiment {randomly_chosen_experiment.get("_id")}')
    return randomly_chosen_experiment


  def _get_llm_config_combinations(self, models_for_evaluating: List[RunnableModel]) -> List[Tuple[RunnableModel, Config]]:
    def get_configs_for_rc(model: RunnableModel) -> List[Config]:
      config1 = Config()
      config1.set_parameter('temperature', 0.01)
      config1.set_parameter('top-p', 0.5)
      return [config1]
    
    def get_configs_for_bot_detection(model: RunnableModel) -> List[Config]:
      config1 = Config()
      config1.set_parameter('prompt_type', 'without explaination')
      config2 = Config()
      config2.set_parameter('prompt_type', 'with explaination')
      return [config1, config2]
    
    def get_configs_for_multiplication(model: RunnableModel) -> List[Config]:
      config1 = Config()
      config1.set_parameter('prompt_type', 'without examples')
      config1.set_parameter('top-k', 1)
      config2 = Config()
      config2.set_parameter('prompt_type', NAME_OF_MULTIPLICATION_PROMPT_WITH_EXAMPLES)
      config2.set_parameter('top-k', 1)
      return [config1, config2]
      
    config_creators_by_task_types = {
      TaskType.READING_COMPREHENSION: get_configs_for_rc,
      TaskType.BOT_DETECTION: get_configs_for_bot_detection,
      TaskType.MULTIPLICATION: get_configs_for_multiplication
    }
    config_creator = config_creators_by_task_types[self.type] # type: Callable
    
    combinations = [] # type: List[Tuple[RunnableModel, Config]]
    for model in models_for_evaluating:
      combinations_for_model = []
      for config in config_creator(model):
        combinations_for_model.append((model, config))
      log.info(f'Got {len(combinations_for_model)} combinations for model {model._id}')
      combinations.extend(combinations_for_model)
    return combinations


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
    """Picks out combinations that were either:
    1) never tested
    2) are using proprietary models, and some time has passed since the last evaluation
    """
    # picks out combinations that either were never tested,
    # or are hosted by OpenRouter and haven't been tested in a while
    PROPRIETARY_MODEL_TIMEOUT = datetime.timedelta(days=31)
    PROPRIETARY_SOURCES_LIST = ['OpenRouter']
    combinations_to_evaluate = []
    for considered_combination in possible_combinations:
      log.info(f'Considering model-config combination:\nModel: {considered_combination[0]._id}\nConfig: {considered_combination[1].to_dict()}')
      model, configuration = considered_combination
      model_source = model.source
      latest_experiment = db_connector.get_latest_evaluation_for_combination(model, task_type_int_to_str[self.type], configuration)
      if latest_experiment is None:
        log.info(f'Combination was never tested completely, adding it to combinations up for evaluations')
        combinations_to_evaluate.append(considered_combination)
      elif model_source in PROPRIETARY_SOURCES_LIST:
        time_since_last_experiment = parse_date(current_date) - parse_date(latest_experiment['date'])
        if time_since_last_experiment > PROPRIETARY_MODEL_TIMEOUT:
          log.info(f'Combination is proprietary and was not tested in a while ({str(time_since_last_experiment)}), adding it to combinations up for evaluations')
          combinations_to_evaluate.append(considered_combination)
        else:
          log.info(f'Combination is proprietary, but it was recently tested ({str(time_since_last_experiment)}), skipping it')
      else:
        log.info(f'Skipping combination')
    return combinations_to_evaluate


  def _create_new_experiment(
      self,
      db_connection: DatabaseConnector,
      date: datetime.datetime):
    """Returns a tuple of experiment ID and llm-config combination.

    Returns None if there was no applicable new experiment to create
    """
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
      return None, None

    combination_for_evaluation = random.choice(combinations_up_for_evaluation)
    log.info(f'Chosen combination {combination_for_evaluation}')
    experiment_id = db_connection.create_experiment_stump(
      model=combination_for_evaluation[0],
      task_type=task_type_int_to_str[self.type],
      config=combination_for_evaluation[1],
      experiment_date=date)
    return experiment_id, combination_for_evaluation


  # returns the llm-config combination
  def _recover_unfinished_experiment_llm_config_combination(self, db_connection: DatabaseConnector, experiment: dict):
    log.info(f'Recovering experiment {experiment.get("_id")}')
    model_id = experiment['model_id']
    model = db_connection.get_model_from_id(model_id)
    config = Config(experiment['config'])
    log.info(f'Experiment: model_id={model_id}, date={experiment.get("date")}, config={config.to_dict()}, output amount={len(experiment.get("outputs"))}')
    return (model, config)
  

  def _compute_prompts_cost(self, token_count: int, model: RunnableModel) -> float:
    true_price = model.get_price_with_discount()
    log.info(f'Model\'s true price is {true_price}')
    if true_price == 0:
      total_cost = 0
    else:
      total_cost = true_price * token_count

    log.info(f'Total token cost is {total_cost}')
    return total_cost
  

  def recover_or_create_experiment(
      self, /,
      db_connection: DatabaseConnector,
      date: datetime.datetime) -> Tuple[str, RunnableModel, Config, List]:
    """Recovers llm-config combination from an unfinished experiment, or creates a new one

    Returns experiment_id, model, config, and the list of already completed outputs
    """

    # use llm-config combination from an unfinished experiment, or create a new one
    unfinished_experiment = self._get_unfinished_experiment_if_any(db_connection)
    already_completed_outputs = []
    if unfinished_experiment is None:
      experiment_id, combination_for_evaluation = self._create_new_experiment(db_connection, date)
      if experiment_id is None:
        log.info(f'Stopping the task as there is no applicable experiment to create')
        return None, None, None, None
    else:
      experiment_id = unfinished_experiment['_id']
      combination_for_evaluation = self._recover_unfinished_experiment_llm_config_combination(
        db_connection,
        unfinished_experiment)
      already_completed_outputs = unfinished_experiment['outputs']
    model, config = combination_for_evaluation
    log.info(f'Returning experiment {experiment_id}')
    return experiment_id, model, config, already_completed_outputs


  def process_experiment_cost(
      self,
      cost_limit: float,
      total_token_count: int,
      model: RunnableModel,
      experiment_id: str,
      db_connection: DatabaseConnector,
      cost_callback: Union[CostCallback, None]) -> bool:
    """Returns true if the cost is acceptable, false if not"""
    if cost_limit is not None:
      total_cost = self._compute_prompts_cost(total_token_count, model)
      if total_cost > cost_limit:
        log.error(f'Cost is too high, aborting experiment {experiment_id} and marking it as finished')
        db_connection.mark_experiment_as_finished(experiment_id, too_expensive=True)
        return False
      if cost_callback is not None:
        cost_callback.register_estimated_initial_cost(total_cost)
        log.info('Registering the cost with the callback')
      log.info('The cost is acceptable')
    else:
      log.info('Cost limit is none')
    return True


  # returns a list of metrics, outputs
  def run_task(
      self,
      db_connection: DatabaseConnector,
      date: datetime.datetime,
      cost_limit=None,
      db_cache_limit=500,
      path_to_save_db_on_update: Union[str, None]=None,
      cost_callback: Callable = None):
    """
    1) Check for unfinished experiments in DB (pick a random one if there are)
    2) Get the list of known models
    3) Create a list of possible LLM-config combiations
    4) Create a list of combinations that should be evaluated
    5) Pick one
    6) Create an experiment record
    7) Prepare the testing data
    7) Estimate and check the cost, mark experiment as finished if its too expensive
    8) For each input id, run it on the model
    9) record results
    """
    log.info(f'Running task {task_type_int_to_str[self.type]} on date {date} with cost limit {cost_limit}')

    experiment_id, model, config, already_completed_outputs = self.recover_or_create_experiment(
      db_connection=db_connection,
      date=date)
    if experiment_id is None:
      return
    
    prompts_dict, total_token_count, total_cut_prompts, validation_data = self._load_and_prepare_prompts_for_task(
      excluded_input_ids=set([output['input_code'] for output in already_completed_outputs]),
      configuration=config,
      model=model)
    runner = ModelRunner(model, config)

    # process the cost
    cost_is_acceptable = self.process_experiment_cost(cost_limit, total_token_count, model, experiment_id, db_connection, cost_callback)
    if not cost_is_acceptable:
      return

    # runs the models
    evaluation_callback = EvaluationResultsCallback(
      db_connection=db_connection,
      experiment_id=experiment_id,
      task=self.type,
      existing_processed_outputs=already_completed_outputs,
      validation_data=validation_data,
      db_enabled=True,
      db_cache_limit=db_cache_limit,
      path_to_save_db_on_update=path_to_save_db_on_update)
    try:
      runner.run_model(
        prompts_dict,
        callback=evaluation_callback,
        max_new_tokens=new_tokens_limit_per_task_type_int[self.type])
      evaluation_callback.finalize_evaluation()
    except Exception as e:
      log.error(e.with_traceback(None))
      raise e
    
    # checks if all evaluations were completed, save metrics if so, complain and die if not
    log.info(f'Checking if all input codes were tested on')
    all_input_codes = set(validation_data.keys())
    processed_input_codes = set([
      output['input_code'] for output in db_connection.get_experiment_from_id(experiment_id)['outputs']])
    if all_input_codes == processed_input_codes:
      log.info(f'Yes, all inputs were processed ({len(processed_input_codes)}) out of ({len(all_input_codes)})')
      evaluation_callback.compute_and_save_metrics()
      db_connection.mark_experiment_as_finished(experiment_id, too_expensive=False)
      if path_to_save_db_on_update is not None:
        log.info(f'Saving finalized DB')
        db_connection.save_data_to_file(path_to_save_db_on_update)
    else:
      log.error(f'Not all input codes were processed, the experiment was not marked as finished: {len(all_input_codes.difference(processed_input_codes))} unprocessed inputs')
