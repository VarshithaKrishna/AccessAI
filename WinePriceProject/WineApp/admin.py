from django.contrib import admin
from .models import Data

class DataAdmin(admin.ModelAdmin):
    list_display = ('name','Text','predictions')

admin.site.register(Data, DataAdmin)
