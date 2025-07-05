from django.urls import path
from WineApp import views
urlpatterns =[
    path('',views.loginpage,name='login'),
    path('register/',views.registerpage,name='register'),
    path('home/',views.home,name='home'),
    path('genai/',views.genai,name='genai'),
    path('ml/',views.ml,name='ml'),
    path('dl',views.dl,name='dl'),
    path('logout/',views.logoutpage,name='logout')
]