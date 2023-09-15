from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest, Http404

from .data_handling import (
  DatabaseConnector, get_unique_config_params_in_evaluations, create_metrics_df, get_possible_config_combinations_in_evaluations, get_unique_input_codes_from_evaluations,
  count_interpreted_answers_for_input_code,
  filter_experiments_by_filters,
  aggregate_metrics_from_experiments,
  prettify_config_dict,
  RC_QUESTIONS, RC_TEXTS
)
from .prompt_constructor import PromptConstructor

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


conn = DatabaseConnector()

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
    data_dict = {'graph_name': 'Answer Distribution', 'score_label': 'Answer Count', 'values': []}
    data_values_list = data_dict['values'] # type: list

    interpreted_answer_counter = Counter()
    for experiment in experiments:
      all_interpreted_answers_in_experiment = [output['interpreted_output'] for output in experiment['outputs']]
      interpreted_answer_counter.update(all_interpreted_answers_in_experiment)
    log.info(f'Got the following counts: {interpreted_answer_counter}')

    for interpreted_value_name, occurence_count in dict(interpreted_answer_counter).items():
      data_values_list.append({'name': interpreted_value_name, 'value': occurence_count})
    
    return data_dict


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


idx2alphabet = {i:letter for i, letter in enumerate(string.ascii_uppercase)}
def _draw_rc_results(request) -> HttpResponse:
  input_code = request.GET.get('input_code', None)
  task_type = request.GET.get('task_type', None)
  log.info(f'Drawing results for input code {input_code} and task type {task_type}')
  llm_configs = json.loads(request.GET.get('llm_configs', None)) # type: List[Dict]

  log.warning(f'Configs being tested: {llm_configs}')

  question_details = RC_QUESTIONS[input_code]
  question_context = RC_TEXTS[question_details['text_id']]

  # todo: rework prompts to be sent on-request
  prompt_and_interpreted_output_counts_per_readable_llm_config_combination = dict()
  for llm_config_combination in llm_configs:
    log.info(f'Counting for combination {llm_config_combination}')
    evals = conn.get_evaluations_for_llm_config_task_combination(task_type, llm_config_combination)
    log.info(f'Got {len(evals)} evaluations')
    interpreted_output_counts_for_model = count_interpreted_answers_for_input_code(evals, input_code)
    log.info(f'Counts: {interpreted_output_counts_for_model}')
    readable_combination_name = llm_config_combination['model_id'] + ' : ' + prettify_config_dict(llm_config_combination['config'])
    prompt_and_interpreted_output_counts_per_readable_llm_config_combination[readable_combination_name] = {
      'prompt': PromptConstructor(task_type, llm_config_combination['config']).construct_prompt(text=question_context, question_dict=question_details),
      'counts':interpreted_output_counts_for_model
    }

  question_details = RC_QUESTIONS[input_code]
  context = {
    'question_context': question_context,
    'input_code': input_code,
    'question': question_details['question'],
    'options': [f'{idx2alphabet.get(i)}) {option}' for i, option in enumerate(question_details['options'])],
    'answer': question_details['answer'],
    'prompt_and_interpreted_output_counts_per_readable_llm_config_combination': prompt_and_interpreted_output_counts_per_readable_llm_config_combination
  }
  print(context)
  return render(request, "frontendapp/rc_results_ui.html", context)


def _draw_results(request) -> HttpResponse:
  task_type = request.GET.get('task_type', None)
  function_selector_by_task = {
    'Reading Comprehension': _draw_rc_results
  }
  return function_selector_by_task[task_type](request)


def _draw_rc_prompt(request: HttpRequest) -> HttpResponse:
  input_code = request.GET.get('input_code', None)
  task_type = request.GET.get('task_type', None)


def _draw_prompt_text_window(request: HttpRequest) -> HttpResponse:
  task_type = request.GET.get('task_type', None)
  function_selector_by_task = {
    'Reading Comprehension': _draw_rc_prompt
  }
  return function_selector_by_task[task_type](request)


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
    return _draw_results(request)
  elif requested_data_type == 'prompt_text':
    #task_type = request.GET.get('task_type', None)
    pass
  else:
    raise Exception(f'Unknown requested_data_type of {requested_data_type}')

  return HttpResponse(json.dumps(final_data))

