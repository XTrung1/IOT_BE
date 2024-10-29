from django.contrib import admin

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('pc_cam/', views.pc_cam, name='pc_cam'),
    path('get_pc_cam/', views.get_pc_cam, name='get_pc_cam'),

    path('esp_cam/', views.esp_cam, name='esp_cam'),
    path('get_esp_cam/', views.get_esp_cam, name='get_esp_cam'),
    path('get_esp_cam_no_detect/', views.get_esp_no_detect, name='get_esp_cam_no_detect'),

    path('toggle_pc_cam/', views.toggle_pc_cam, name='toggle_pc_cam'),

    path('stream/', views.get_esp_cam, name='get_esp_cam'),
    path('stream_pc/', views.get_pc_cam, name='get_pc_cam'),

    path('capture/', views.capture, name="capture"),
    path('attendance/', views.attendance, name="attendance"),
    path('train/', views.train_faces, name="train"),
]
