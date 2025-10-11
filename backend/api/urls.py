from django.urls import path
from . import views

urlpatterns = [
    path('ping/', views.ping, name='ping'),
    path('predict/', views.predict_view, name='predict'),
    path('results/', views.results_view, name='results'),
]
