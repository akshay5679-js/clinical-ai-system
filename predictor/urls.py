

from django.urls import path
from . import views

urlpatterns = [

    # HOME PAGE
    path('', views.home, name='home'),

    # HEART PREDICTION
    path('predict/', views.index, name="predict"),

    # CHATBOT
    path('chat/', views.chat, name='chat'),

    # REPORT ANALYZER
    path('summarize/', views.summarize, name="summarize"),

    # ABOUT PAGE
    path('about/', views.about, name="about"),

    path('report/', views.report_upload, name='report'),

]