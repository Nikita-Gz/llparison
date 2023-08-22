from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest, Http404

from .fake_run_data import get_processed_dict_for_output

import random
import json

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def comparisons(request: HttpRequest):
  general_data, _ = get_processed_dict_for_output()

  context = {'general_data': general_data}
  return render(request, "frontendapp/comparisons.html", context)
  #return HttpResponse("A?")


def check_if_model_exists(model_id):
  # todo
  return True


def just_data(request: HttpRequest):
  model_id = request.GET.get('single_model_id', None)
  if model_id is None or not check_if_model_exists(model_id):
    return Http404()
  
  filters = dict()
  for parameter in request.GET:
    filter_prefix = 'filter_'
    if parameter.startswith(filter_prefix):
      filter_name = parameter[len(filter_prefix):]
      filters[filter_name] = request.GET.get(parameter)
  

  _, model_data = get_processed_dict_for_output()
  model_data = model_data[model_id]

  # for tests
  if len(filters) != 0:
    for task_rating in model_data['task_ratings']:
      for metric in task_rating['metrics']:
        metric['value'] = random.random()

  return HttpResponse(json.dumps(model_data))
  #return HttpResponse("A?")

