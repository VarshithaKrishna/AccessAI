from django import forms
from .models import Data

class DataForm(forms.ModelForm):
    class Meta:
        model = Data
        fields = ['name','Text']
       
class VideoUploadForm(forms.Form):
    video_file = forms.FileField()