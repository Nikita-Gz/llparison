from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("comparisons/", views.comparisons, name="comparisons"),
    path("comparisons/data/", views.just_task_rating_data, name="comparisons_single_model_data"),
    path("task_inspection/", views.task_results_ui, name='task_data_ui'),
    path("task_inspection/data/", views.task_results_data, name='task_data_json'),
]

