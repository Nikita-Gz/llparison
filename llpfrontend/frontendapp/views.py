from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest, Http404

from .fake_run_data import get_processed_dict_for_output
from .data_handling import DatabaseConnector, get_unique_config_params_in_evaluations, create_evaluations_df

import random
from typing import *
import pandas as pd
import copy
import json
import logging

log = logging.getLogger("views.py")
logging.basicConfig(level=logging.INFO)


conn = DatabaseConnector()

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def comparisons(request: HttpRequest):
  models_list = conn.get_unique_models_with_evaluations()
  assert len(models_list) != 0
  selected_model = models_list[0]
  
  general_data, _ = get_processed_dict_for_output()

  context = {'general_data': {'models_list': models_list, 'selected_model': selected_model}}
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


def just_data(request: HttpRequest):
  log.info('\n'*4)
  model_id = request.GET.get('single_model_id', None)
  model_evaluations = conn.get_evaluations_for_model(model_id)
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

