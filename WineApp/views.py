from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.shortcuts import render, redirect
from .models import predict

def loginpage(request):
    if request.method == 'POST':
        username = request.POST.get('num1')
        password = request.POST.get('num2')
        user=authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('home')
    return render(request,'login.html')
    
def registerpage(request):
    if request.method == 'POST':
        username = request.POST.get('num1')
        password = request.POST.get('num2')
        conform = request.POST.get('num3')
        if password != conform:
            return render(request,'register.html',{'result':'ERROR'})
        user=User.objects.create_user(username=username,password=password)
        return redirect('login')
    return render(request,'register.html')
def home(request):
    return render (request,'home.html')
def logoutpage(request):
    logout(request)
    return redirect('login')
def genai(request):
    return render(request,'genai.html')

def ml(request):
    if request.method == "POST":
        wine_data = {
            "body": request.POST.get("body"),
            "acidity": request.POST.get("acidity"),
            "rating": request.POST.get("rating"),
            "country": request.POST.get("country"),
            "winery": request.POST.get("winery"),
            "type": request.POST.get("type"),
            "region": request.POST.get("region"),
            "year": request.POST.get("year"),
            "wine": request.POST.get("wine"),
            "num_reviews": request.POST.get("num_reviews"),
        }

        predicted_price=predict(wine_data)
        return render(request, "ML/prediction.html", {
        "price": round(predicted_price[0],4),
        "wine_data": wine_data
        })
    return render(request, "ml.html")


def dl(request):
    
    return render(request,'dl.html')

