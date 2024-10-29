from django.contrib import admin

from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name="BE Home"),
    path('video/', include("video_processing.urls")),
    path('admin/', admin.site.urls),

]
