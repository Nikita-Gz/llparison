from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest

from .fake_run_data import get_processed_dict_for_output

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def comparisons(request: HttpRequest):
  data = {'persons': ['mumu', 'uu', 'mm'],
          'mesgs': [50, 15, 34]}
  
  processed_data = get_processed_dict_for_output()

  context = {'qs': data, 'model_data': processed_data}
  return render(request, "frontendapp/comparisons.html", context)
  #return HttpResponse("A?")

