from django.urls import path
from WineApp import views

# Path can be given as per your wish, you can give path however you like, it's just the url that appears when you navigate through those html pages
# Name should be equal to the one given to your html pages in templates folder
# and views.name should be equal to the method name given in views.
# so you can give any name to those three parameters a or you can also give same names as well.

urlpatterns =[
    path('',views.loginpage,name='login'),
    path('chat/', views.chatbot_view, name='chatbot'),
    path('register/',views.registerpage,name='register'),
    path('home/',views.home,name='home'),
    path('genai/',views.genai,name='genai'),
    path('ml/',views.ml,name='ml'),
    path('dl/', views.dl, name='dl'),
    path('dl_upload/', views.dl_upload, name='dl_upload'),
    path('logout/',views.logoutpage,name='logout'),
    path('index/', views.index, name='dashboard-index'),
    path('predictions/', views.predictions, name='dashboard-predictions'),
    path('result/', views.result, name='result'),
    path('homegen/', views.genHome, name='genhome'),
    path('vsd/', views.vsd, name='vsd'),



]