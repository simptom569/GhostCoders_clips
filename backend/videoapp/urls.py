# videoapp/urls.py


from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_and_process_video, name='upload_video'),
    path('result/<int:video_id>/', views.video_result, name='video_result'),
]