from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpRequest

def index(request: HttpRequest):
  return render(request, "frontendapp/index.html", {})


def comparisons(request: HttpRequest):
  data =  [
        {'person': 'mumu', 'mesgs': 50},
        {'person': 'uu', 'mesgs': 20},
        {'person': 'mm', 'mesgs': 34},
        {'person': 'mu', 'mesgs': 80},
        {'person': 'mimi', 'mesgs': 2},
    ]
  data = {'persons': ['mumu', 'uu', 'mm', 'mu', 'mi'],
          'mesgs': [50, 20, 34, 80, 2]}
  context = {'qs': data}
  return render(request, "frontendapp/comparisons.html", context)
  #return HttpResponse("A?")

