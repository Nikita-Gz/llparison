from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest, Http404

from .data_handling import (DatabaseConnector, get_unique_config_params_in_evaluations, create_evaluations_df, get_possible_config_combinations_in_evaluations, get_unique_input_codes_from_evaluations,
                            count_interpreted_answers_for_input_code,
                            RC_QUESTIONS, RC_TEXTS
)

import random
from typing import *
import pandas as pd
import copy
import json
import logging
import string

log = logging.getLogger("views.py")
logging.basicConfig(level=logging.INFO)


conn = DatabaseConnector()

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def comparisons(request: HttpRequest):
  models_list = conn.get_unique_model_ids_with_finished_evaluations()
<<<<<<< HEAD

  if len(models_list) == 0:
    selected_model = None
    no_models = True
  else:
    selected_model = models_list[0]
    no_models = False
=======
  assert len(models_list) != 0
  selected_model = models_list[0]
>>>>>>> c4c96fb7a4dfd15a94dfe5c6c15b9eaa5333332c
  
  context = {'general_data': {'models_list': models_list, 'selected_model': selected_model},
             'no_models': no_models}
  return render(request, "frontendapp/comparisons.html", context)
  #return HttpResponse("A?")


# adds "all" and "none" to the filter options, adds fitlering by dates
def get_filters_for_ui(filters_list: List[Dict], evaluations: List[Dict]) -> List[Dict]:
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
def get_task_ratings_for_ui_from_computed_data(computed_df: pd.DataFrame) -> List[Dict]:
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


def just_task_rating_data(request: HttpRequest):
  log.info('\n'*4)
  model_id = request.GET.get('single_model_id', None)
  model_evaluations = conn.get_finished_evaluations_for_model(model_id)
  if len(model_evaluations) == 0:
    return Http404()
  
  all_possible_parameters = get_unique_config_params_in_evaluations(model_evaluations)

  filters_in_request = dict()
  date_filter = 'all'
  for parameter in request.GET:
    filter_prefix = 'filter_'
    if parameter == 'filter_Date':
      date_filter = request.GET.get(parameter, 'all')
    elif parameter.startswith(filter_prefix):
      filter_name = parameter[len(filter_prefix):]
      filters_in_request[filter_name] = convert_to_numeric_if_possible(request.GET.get(parameter, 'all'))
  log.info(f'Filters: {filters_in_request}')
  log.info(f'Date filter: {date_filter}')
  
  computed_eval_data = create_evaluations_df(
    model_evaluations,
    all_possible_parameters,
    filters_in_request,
    date_filter=date_filter)
  task_ratings = get_task_ratings_for_ui_from_computed_data(computed_eval_data)

  data = {
    'filters': get_filters_for_ui(all_possible_parameters, model_evaluations),
    'evaluation_error_message': 'There were errors in model evaluation' if False else '',
    'task_ratings': task_ratings
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
  llm_configs = json.loads(request.GET.get('llm_configs', None))

  interpreted_output_counts_per_model = dict()
  for llm_config_combination in llm_configs:
    evals = conn.get_evaluations_for_llm_config_task_combination(task_type, llm_config_combination)
    interpreted_output_counts_for_model = count_interpreted_answers_for_input_code(evals, input_code)
    interpreted_output_counts_per_model[llm_config_combination['model_id']] = interpreted_output_counts_for_model

  question_details = RC_QUESTIONS[input_code]
  context = {
    'question_context': RC_TEXTS[question_details['text_id']],
    'input_code': input_code,
    'question': question_details['question'],
    'options': [f'{idx2alphabet.get(i)}) {option}' for i, option in enumerate(question_details['options'])],
    'answer': question_details['answer'],
    'interpreted_output_counts_per_model': interpreted_output_counts_per_model
  }
  print(context)
  return render(request, "frontendapp/rc_results_ui.html", context)


def _draw_results(request) -> HttpResponse:
  task_type = request.GET.get('task_type', None)
  function_selector_by_task = {
    'Reading Comprehension': _draw_rc_results
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
  else:
    raise Exception(f'Unknown requested_data_type of {requested_data_type}')

  return HttpResponse(json.dumps(final_data))

