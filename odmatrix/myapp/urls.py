from django.urls import path
from . import views

urlpatterns = [
    # path('', views.visualize_od_matrix, name='visualize_od_matrix'),
    path('', views.index, name='index'),
]
