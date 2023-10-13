from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("comparisons/", views.single_model_graphs_page, name="comparisons"),
    path("comparisons/data/", views.get_graphs_data_for_one_model, name="comparisons_single_model_data"),
    path("task_inspection/", views.task_results_ui, name='task_data_ui'),
    path("task_inspection/data/", views.task_results_data, name='task_data_json'),
    path("custom_inference/", views.custom_inference_page, name="custom_inference_page")
]

