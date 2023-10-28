from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest, Http404

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .frontend_data_handling import (
  DatabaseConnector, get_unique_config_params_in_evaluations, create_metrics_df, get_possible_config_combinations_in_evaluations, get_unique_input_codes_from_evaluations,
  count_interpreted_answers_for_input_code,
  filter_experiments_by_filters,
  aggregate_metrics_from_experiments,
  prettify_config_dict,
  RC_QUESTIONS, RC_TEXTS, BOT_DETECTION_DATASET, MULTIPLICATION_DATASET
)
from model_running.prompt_constructor import PromptConstructor, UniversalTokenizer
from model_running.run_config import Config
from model_running.task_type import task_type_str_to_int, TaskType
from model_running.runnable_model_data import RunnableModel
from .inference_runner import InferenceRunner

import random
from typing import *
import pandas as pd
import copy
import json
import logging
import string
from collections import Counter

log = logging.getLogger("views.py")
logging.basicConfig(level=logging.DEBUG)


conn = DatabaseConnector(
  fill_with_testing_stuff=False,
  path_to_preload_data='./db_dump')
inferer = InferenceRunner(conn)

# this will hold requested universal tokenizers for each model ID
tokenizers = {} # type: Dict[str, UniversalTokenizer]

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def single_model_graphs_page(request: HttpRequest):
  models_list = conn.get_unique_model_ids_with_finished_evaluations()

  if len(models_list) == 0:
    selected_model = None
    no_models = True
  else:
    selected_model = models_list[0]
    no_models = False
  
  context = {'general_data': {'models_list': models_list, 'selected_model': selected_model},
             'no_models': no_models}
  return render(request, "frontendapp/single_model_graphs_page.html", context)
  #return HttpResponse("A?")


# adds "all" and "none" to the filter options, adds fitlering by dates
def get_config_filters_for_ui(filters_list: List[Dict], evaluations: List[Dict]) -> List[Dict]:
  filters_list = copy.deepcopy(filters_list)
  for filter_dict in filters_list:
    filter_dict['values'].extend(['all', 'none'])
    filter_dict['default'] = 'all'
  
  unique_dates = list(set([evaluation['date'] for evaluation in evaluations]))
  filters_list.append({'name': 'Date', 'values': unique_dates + ['all'], 'default': 'all'})

  return filters_list


"""
'task_ratings':
      [
        {
          'task_name': 'Idk',
          'metrics':
          [
            {
              'name': 'Accuracy',
              'value': 0.64
            },
            {
              'name': 'False Positives',
              'value': 0.48
            }
          ]
        },
"""
def get_task_metrics_for_ui_from_computed_data(computed_df: pd.DataFrame) -> List[Dict]:
  """computed_df columns: model_id, task_type, metric_name, value
  """
  print('sdfsdf'*20)
  print(computed_df)
  final_list = []

  # will need another function for multiple models
  assert len(computed_df['model_id'].unique()) < 2
  
  # todo: do i really need nests here? Rework
  task_types = computed_df['task_type'].unique()
  for task_type in task_types:
    metrics_for_task = []
    for _, row in computed_df[computed_df['task_type'] == task_type].iterrows():
      print('A'*20)
      print(row)
      metrics_for_task.append({'name': row['metric_name'], 'value': row['value']})
    final_list.append({
      'task_name': task_type,
      'metrics': metrics_for_task
    })
  return final_list


def convert_to_numeric_if_possible(value: Any) -> Union[float, Any]:
  try:
    return float(value)
  except:
    return value


def create_single_model_graphs_data(model_experiments: List[Dict]) -> List[Dict]:
  """Creates a list of dicts describing a task and necessary graphs for it

  Return format:
  [
    {
      task_name: str,
      graphs:
      [
        {
          graph_name: str,
          score_label: str,
          values:
          [
            {
              name: str,
              value: float
            }
          ]
        }
      ]
    }
  ]

  Reading Comprehension task will add a graph with interpreted answers distribution
  """
  log.info(f'Creating graphs for the model')

  def create_metrics_graph_data_dict(metrics: Dict[str, float]) -> Dict[str, Union[str, Dict]]:
    log.info(f'Creating metric graph')
    data_dict = {'graph_name': 'Metrics', 'score_label': 'Metric Score', 'values': []}
    data_values = data_dict['values'] # type: list
    for metric_name, metric_value in metrics.items():
      data_values.append({'name': metric_name, 'value': metric_value})
    return data_dict
  
  def create_RC_answer_distribution_graph_data_dict(experiments: List[Dict]) -> Dict[str, Union[str, Dict]]:
    log.info(f'Creating RC answer distrigution graph')
    final_data_dict = {'graph_name': 'Answer Distribution', 'score_label': 'Answer Count', 'values': []}
    data_values_list = final_data_dict['values'] # type: list

    interpreted_answer_counter = Counter()
    for experiment in experiments:
      interpreted_answers_in_experiment = [output['interpreted_output'] for output in experiment['outputs']]

      # Changes answers that aren't A/B/C/D into "Others"
      corrected_interpreted_answers = [
        answer if answer in ['A', 'B', 'C', 'D'] else 'Other'
        for answer in interpreted_answers_in_experiment]

      interpreted_answer_counter.update(corrected_interpreted_answers)
      log.info(f'Got the following counts: {interpreted_answer_counter}')

    for interpreted_value_name, occurence_count in dict(interpreted_answer_counter).items():
      if interpreted_value_name is None:
        interpreted_value_name = '-'
      data_values_list.append({'name': interpreted_value_name, 'value': occurence_count})

    log.info(f'Got the following counts: {data_values_list}')
    final_data_dict['values'] = sorted(data_values_list, key=lambda answer_count: answer_count['name'])
    
    return final_data_dict


  # fills it up by tasks
  final_data_list= []
  unique_task_names_in_experiments = set([experiment['task_type'] for experiment in model_experiments])
  for current_task_name in unique_task_names_in_experiments:
    log.info(f'Creating graphs for the task {current_task_name}')
    experiments_matching_the_task = [experiment for experiment in model_experiments if experiment['task_type'] in current_task_name]
    task_record = {'task_name': current_task_name, 'graphs': []}
    task_graphs = task_record['graphs'] # type: List[Dict]

    # metrics graph
    metrics_for_the_task = aggregate_metrics_from_experiments(experiments_matching_the_task)
    task_graphs.append(create_metrics_graph_data_dict(metrics_for_the_task))

    # special graphs based on the task
    if current_task_name == 'Reading Comprehension':
      task_graphs.append(create_RC_answer_distribution_graph_data_dict(experiments_matching_the_task))

    final_data_list.append(task_record)
  
  return final_data_list


def _create_notes_list_from_experiments(model_experiments: List[Dict]) -> List[str]:
  """Currently this just sums up "Culled prompts count" notes"""
  final_notes_list = []
  total_culled_prompts = 0
  for experiment in model_experiments:
    notes = experiment['notes'] # type: Dict[str, int]
    if not isinstance(notes, dict):
      log.info(f'Skipping notes for experiment {experiment["_id"]} as they are not in dict format ({type(notes)})')
      continue
    culled_prompts_in_experiment = notes.get('Culled prompts count', 0)
    log.info(f'Read {culled_prompts_in_experiment} culled prompt count in experiment {experiment["_id"]}')
    total_culled_prompts += culled_prompts_in_experiment
  final_notes_list.append(f'Culled prompts count: {total_culled_prompts}')
  return final_notes_list


def get_graphs_data_for_one_model(request: HttpRequest):
  log.info('\n'*4)
  model_id = request.GET.get('single_model_id', None)
  model_evaluations = conn.get_finished_evaluations_for_model(model_id)
  if len(model_evaluations) == 0:
    return Http404()

  filters_in_request = dict()
  date_filter = 'all'
  for parameter in request.GET:
    filter_prefix = 'filter_'
    if parameter == 'filter_Date':
      date_filter = request.GET.get(parameter, 'all')
    elif parameter.startswith(filter_prefix):
      filter_name = parameter[len(filter_prefix):]
      filters_in_request[filter_name] = convert_to_numeric_if_possible(request.GET.get(parameter, 'all'))
  log.info(f'Model ID: {model_id}')
  log.info(f'Filters: {filters_in_request}')
  log.info(f'Date filter: {date_filter}')

  # todo: consider moving it to after experiment filtering. Why not? What will this break?
  all_possible_parameters = get_unique_config_params_in_evaluations(model_evaluations)

  filtered_model_experiments = filter_experiments_by_filters(
    experiments=model_evaluations,
    config_filter_values=filters_in_request,
    date_filter=date_filter)
  
  tasks_graphs_data = create_single_model_graphs_data(filtered_model_experiments)

  data = {
    'filters': get_config_filters_for_ui(all_possible_parameters, model_evaluations),
    'notes': _create_notes_list_from_experiments(filtered_model_experiments),
    'evaluation_error_message': 'There were errors in model evaluation' if False else '',
    'tasks_graphs_data': tasks_graphs_data
  }

  return HttpResponse(json.dumps(data))
  #return HttpResponse("A?")


def task_results_ui(request: HttpRequest):
  models_list = conn.get_unique_model_ids_with_finished_evaluations()
  assert len(models_list) != 0

  context = {
    'models_ids': models_list,
    'task_types': conn.get_unique_task_types()
  }
  return render(request, "frontendapp/task_results.html", context)


def _get_prompt_for_model_config_combination(
    task_type_str: str,
    config: Dict,
    model: RunnableModel,
    **kwargs
):
  tokenizer = tokenizers.get(model._id, UniversalTokenizer(model))
  tokenizers[model._id] = tokenizer
  try:
    prompt = PromptConstructor(
      task_type=task_type_str_to_int[task_type_str],
      configuration_dict=config,
      model=model,
      existing_tokenizer=tokenizer
      ).construct_prompt(**kwargs)[0]
  except Exception as e:
    log.error(f'Exception when constructing prompt: {e}')
    prompt = 'No prompt available for this combination'
  return prompt


def _get_task_specific_dataset_entry_for_input_code(task_type_str: str, input_code: str) -> Dict:
  """
  Returns an entry, which can be used in place of optional kwargs for prompt constructor's construct_prompt()
  """
  task_type_int = task_type_str_to_int[task_type_str]
  if task_type_int == TaskType.READING_COMPREHENSION:
    question_entry = RC_QUESTIONS[input_code]
    return {'question_dict': question_entry, 'context_text': RC_TEXTS[question_entry['text_id']]}
  elif task_type_int == TaskType.BOT_DETECTION:
    _, post_history = BOT_DETECTION_DATASET[input_code]
    return {'post_history': post_history}
  elif task_type_int == TaskType.MULTIPLICATION:
    equation, answer = MULTIPLICATION_DATASET[input_code]
    return {'math_expression': equation, 'answer': answer}
  else:
    raise Exception(f'Unknown task type "{task_type_str}"')


def _get_prompts_and_answer_counts_for_llm_config_combinations_for_dataset_entry(
    input_code: str,
    task_type_str: str,
    llm_configs: List[Dict]) -> Dict[str, Dict[str, Union[str, Dict]]]:
  model_id_to_runnable_models = {
    model._id:model for model in conn.model_ids_to_runnable_models([combo['model_id'] for combo in llm_configs])}
  prompt_and_output_counts_per_llm_config_combination = dict()
  for combination in llm_configs:
    model_id, config = combination['model_id'], combination['config']
    runnable_model = model_id_to_runnable_models[model_id]
    log.info(f'Counting for combination {model_id, config}')
    evals = conn.get_evaluations_for_llm_config_task_combination(task_type_str, runnable_model, config)
    log.info(f'Got {len(evals)} evaluations')
    interpreted_output_counts_for_model = count_interpreted_answers_for_input_code(evals, input_code)
    log.info(f'Counts: {interpreted_output_counts_for_model}')
    readable_combination_name = model_id + ' : ' + prettify_config_dict(config)

    tokenizer = tokenizers.get(model_id, UniversalTokenizer(runnable_model))
    tokenizers[model_id] = tokenizer
    task_specific_prompt_kwargs = _get_task_specific_dataset_entry_for_input_code(task_type_str, input_code)
    prompt_and_output_counts_per_llm_config_combination[readable_combination_name] = {
      'prompt': _get_prompt_for_model_config_combination(
        task_type_str, config, runnable_model,
        **task_specific_prompt_kwargs),
      'counts': interpreted_output_counts_for_model
    }
  return prompt_and_output_counts_per_llm_config_combination


idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}
def _get_task_specific_context(input_code:str, task_type_str: str, llm_configs: List[Dict]) -> Dict:
  context = {
    'input_code': input_code
  }
  task_type_as_enum = task_type_str_to_int[task_type_str]
  if task_type_as_enum == TaskType.READING_COMPREHENSION:
    question_details = RC_QUESTIONS[input_code]
    context['question_context'] = RC_TEXTS[question_details['text_id']]
    context['question'] = question_details['question']
    context['options'] = [f'{idx2alphabet.get(i)}) {option}' for i, option in enumerate(question_details['options'])]
    context['answer'] = question_details['answer']
  elif task_type_as_enum == TaskType.BOT_DETECTION:
    question_details = BOT_DETECTION_DATASET[input_code]
    is_bot, post_history = question_details
    context['post_history'] = post_history
    context['is_bot'] = is_bot
  elif task_type_as_enum == TaskType.MULTIPLICATION:
    equation, answer = MULTIPLICATION_DATASET[input_code]
    context['equation'] = equation
    context['answer'] = answer
  else:
    raise Exception(f'Unknown task type "{task_type_str}"')
  
  context['prompt_and_interpreted_output_counts_per_readable_llm_config_combination'] = (
    _get_prompts_and_answer_counts_for_llm_config_combinations_for_dataset_entry(
      input_code=input_code,
      task_type_str=task_type_str,
      llm_configs=llm_configs))
  return context


def _draw_single_test_results(request) -> HttpResponse:
  input_code = request.GET.get('input_code', None) # type: str
  task_type_str = request.GET.get('task_type', None) # type: str
  llm_configs = json.loads(request.GET.get('llm_configs', None)) # type: List[Dict]
  log.info(f'Drawing results for input code {input_code} and task type {task_type_str}')

  context = _get_task_specific_context(
    input_code=input_code,
    task_type_str=task_type_str,
    llm_configs=llm_configs)

  appropriate_render_template = {
    TaskType.READING_COMPREHENSION: 'frontendapp/rc_results_ui.html',
    TaskType.BOT_DETECTION: 'frontendapp/bd_results_ui.html',
    TaskType.MULTIPLICATION: 'frontendapp/multiplication_results_ui.html'
  }[task_type_str_to_int[task_type_str]]

  return render(request, appropriate_render_template, context)


def task_results_data(request: HttpRequest):
  final_data = dict()

  requested_data_type = request.GET.get('requested_data_type', None)
  if requested_data_type == 'config_combinations':
    model_id = request.GET.get('selected_model', None)
    log.info(f'Returning config combinations for model {model_id}')
    model_evaluations = conn.get_finished_evaluations_for_model(model_id)
    combinations = get_possible_config_combinations_in_evaluations(model_evaluations)
    log.info(f'Got combinations: {combinations}')
    final_data['config_combinations'] = combinations
  elif requested_data_type == 'evaluations':
    task_type = request.GET.get('task_type', None)
    combinations = json.loads(request.GET.get('llm_configs', None))
    log.info(f'Returning tests for task {task_type} and combos {combinations}')
    evaluations = conn.get_evaluations_for_llm_config_task_combinations(task_type, combinations)
    input_codes = get_unique_input_codes_from_evaluations(evaluations)
    final_data['input_codes'] = input_codes
  elif requested_data_type == 'evaluation_graphic':
    return _draw_single_test_results(request)
  elif requested_data_type == 'prompt_text':
    #task_type = request.GET.get('task_type', None)
    pass
  else:
    raise Exception(f'Unknown requested_data_type of {requested_data_type}')

  return HttpResponse(json.dumps(final_data))


def custom_inference_page(request: HttpRequest):
  """This returns the custom inference page filled with task types and model ID's"""

  task_options = conn.get_unique_task_types()
  model_ids = conn.get_unique_model_ids()

  context = {
    'task_options': task_options,
    'model_ids': model_ids
  }
  return render(request, 'frontendapp/custom_inference_page.html', context)


def process_inference_request(request: HttpRequest):
  """This runs the requested model with specified parameters, task type and task inputs"""

  model_id = request.GET.get('model_id', None)
  task_type_str = request.GET.get('task_type', None)
  input_fields = json.loads(request.GET.get('input_fields', None))
  config = json.loads(request.GET.get('config', None))
  log.info(f"Received inference request")
  log.info(f"model_id = {model_id}")
  log.info(f"task_type = {task_type_str}")
  log.info(f"input_fields = {input_fields}")
  log.info(f"config = {config}")

  task_type_int = task_type_str_to_int[task_type_str]
  config = Config(parameters_dict=config)
  runnable_model = conn.get_model_from_id(model_id)
  tokenizer = tokenizers.get(model_id, UniversalTokenizer(runnable_model))
  tokenizers[model_id] = tokenizer
  output, microseconds = inferer.infer(
    model=runnable_model,
    task_type=task_type_int,
    input_fields=input_fields,
    config=config,
    tokenizer=tokenizer)

  inference_result = {'output': output, 'microseconds': microseconds}
  return HttpResponse(json.dumps(inference_result))

