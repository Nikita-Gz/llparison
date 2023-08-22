from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("comparisons/", views.comparisons, name="comparisons"),
    path("comparisons/data/", views.just_data, name="comparisons_single_model_data")
]

