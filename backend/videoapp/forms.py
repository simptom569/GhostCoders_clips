# videoapp/forms.py

from django import forms
from .models import VideoFile

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoFile
        fields = ['video']
