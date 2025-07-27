from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.shortcuts import render, redirect
from .models import *
from django.conf import settings
from django.contrib.auth.models import User
from .forms import DataForm
from .models import Data
from PIL import Image
import pytesseract
import fitz
import os
from django.core.files.storage import default_storage


def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"

# 🧩 Image text extraction
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"[Error processing image: {str(e)}]"

# 🏠 Home view — renders genhome.html
def genHome(request):
    return render(request, "AI/genhome.html")

# 📄 Result view — handles form submission
def result(request):
    extracted_text = ""

    if request.method == "POST":
        input_type = request.POST.get("input_type", "").strip()
        voice_text = request.POST.get("voice_text", "").strip()

        # Priority: voice input if present
        if voice_text:
            extracted_text = voice_text

        elif input_type == "text":
            extracted_text = request.POST.get("input_text", "").strip()

        elif input_type == "pdf":
            pdf_file = request.FILES.get("pdf_file")
            if pdf_file:
                extracted_text = extract_text_from_pdf(pdf_file)

        elif input_type == "image":
            image_file = request.FILES.get("image_file")
            if image_file:
                extracted_text = extract_text_from_image(image_file)

        else:
            extracted_text = "[No valid input type provided.]"

    return render(request, "AI/result.html", {"extracted_text": extracted_text})



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

def dl_upload(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        result = predictDL(uploaded_image)

        # Pass result to prediction.html
        return render(request, 'DL/prediction.html', {'result': result})

    return render(request, 'DL/dl_upload.html')

chat_history = []

def chatbot_view(request):
    global chat_history

    if request.method == "POST":
        user_input = request.POST.get("user_input")

        if user_input.lower() == "exit":
            chat_history.clear()
            return redirect('genai')

        response = send_message(user_input)
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Gemini: {response}")

    return render(request, "AI/chatbot.html", {"chat_history": chat_history})

def index(request):
    if request.method == 'POST':
        form = DataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('dashboard-predictions')
    else:
        form = DataForm()
    context = {
        'form': form,
    }
    return render(request, 'AI/dashboard/index.html', context)

def predictions(request):
    predicted_tweets = Data.objects.all()
    context = {
        'predicted_tweets': predicted_tweets
    }
    return render(request, 'AI/dashboard/predictions.html', context)

import cv2
import time
from datetime import datetime
import os
import numpy as np
from .forms import VideoUploadForm
from django.conf import settings
from scipy.spatial import distance as dist

def blackout(image, WIDTH):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array([[0,0], [xBlack,0], [0,yBlack]])
    triangle_cnt2 = np.array([[WIDTH,0], [WIDTH-xBlack,0], [WIDTH,yBlack]])
    cv2.drawContours(image, [triangle_cnt], 0, (0,0,0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0,0,0), -1)
    return image

def process_video(video_path, classifier_path, output_dir, speed_limit=20):
    WIDTH, HEIGHT, cropBegin = 1280, 720, 240
    mark1, mark2, markGap, fpsFactor = 120, 360, 15, 3
    carCascade = cv2.CascadeClassifier(classifier_path)
    video = cv2.VideoCapture(video_path)
    frameCounter = 0
    overspeed_cars = []

    # Simple centroid tracker state
    nextCarID = 0
    cars = {}  # carID: {'centroid': (x, y), 'bbox': (x, y, w, h), 'start_time': None, 'end_time': None}
    disappeared = {}  # carID: frames since last seen

    MAX_DISAPPEARED = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        rc, image = video.read()
        if not rc or image is None:
            break

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = image[cropBegin:720, 0:1280]
        image = np.ascontiguousarray(image)
        if image is None or image.size == 0 or len(image.shape) != 3 or image.shape[2] != 3:
            continue

        resultImage = blackout(image.copy(), WIDTH)
        frameCounter += 1

        # Detect cars every frame (or every N frames for speed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

        # Compute centroids for new detections
        inputCentroids = []
        inputBBoxes = []
        for (x, y, w, h) in detections:
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            inputCentroids.append((cX, cY))
            inputBBoxes.append((x, y, w, h))

        # If no cars are currently tracked, register all detections
        if len(cars) == 0:
            for i in range(len(inputCentroids)):
                cars[nextCarID] = {
                    'centroid': inputCentroids[i],
                    'bbox': inputBBoxes[i],
                    'start_time': None,
                    'end_time': None
                }
                disappeared[nextCarID] = 0
                nextCarID += 1
        else:
            # Match input centroids to existing cars using Euclidean distance
            carIDs = list(cars.keys())
            carCentroids = [cars[carID]['centroid'] for carID in carIDs]
            if len(inputCentroids) > 0 and len(carCentroids) > 0:
                D = dist.cdist(np.array(carCentroids), np.array(inputCentroids))
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                assignedRows = set()
                assignedCols = set()
                for (row, col) in zip(rows, cols):
                    if row in assignedRows or col in assignedCols:
                        continue
                    carID = carIDs[row]
                    cars[carID]['centroid'] = inputCentroids[col]
                    cars[carID]['bbox'] = inputBBoxes[col]
                    disappeared[carID] = 0
                    assignedRows.add(row)
                    assignedCols.add(col)
                # Register new detections as new cars
                for col in range(len(inputCentroids)):
                    if col not in assignedCols:
                        cars[nextCarID] = {
                            'centroid': inputCentroids[col],
                            'bbox': inputBBoxes[col],
                            'start_time': None,
                            'end_time': None
                        }
                        disappeared[nextCarID] = 0
                        nextCarID += 1
                # Mark disappeared cars
                for row in range(len(carCentroids)):
                    if row not in assignedRows:
                        carID = carIDs[row]
                        disappeared[carID] += 1
                        if disappeared[carID] > MAX_DISAPPEARED:
                            del cars[carID]
                            del disappeared[carID]
            else:
                # No detections: mark all as disappeared
                for carID in carIDs:
                    disappeared[carID] += 1
                    if disappeared[carID] > MAX_DISAPPEARED:
                        del cars[carID]
                        del disappeared[carID]

        # Speed estimation and snapshot
        for carID in list(cars.keys()):
            x, y, w, h = cars[carID]['bbox']
            cY = y + h
            now_time = time.time()
            # Start timing when car enters between mark1 and mark2
            if cars[carID]['start_time'] is None and mark2 > cY > mark1 and y < mark1:
                cars[carID]['start_time'] = now_time
            # End timing when car leaves mark2
            elif cars[carID]['start_time'] is not None and cars[carID]['end_time'] is None and cY > mark2:
                cars[carID]['end_time'] = now_time
                timeDiff = cars[carID]['end_time'] - cars[carID]['start_time']
                speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
                if speed > speed_limit:
                    now = datetime.now()
                    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
                    link = os.path.join(output_dir, f"{nameCurTime}.jpeg")
                    cropped_image = image[y:y+h, x:x+w]
                    if cropped_image is None or cropped_image.size == 0:
                        continue
                    if cropped_image.dtype != np.uint8:
                        cropped_image = cropped_image.astype(np.uint8)
                    if len(cropped_image.shape) == 2:
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(link, cropped_image)
                    overspeed_cars.append({'car_id': carID, 'speed': speed, 'image': link})

    video.release()
    return overspeed_cars

def vsd(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['video_file']
            video_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
            with open(video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            classifier_path = os.path.join(settings.BASE_DIR, 'DL/HaarCascadeClassifier.xml')
            output_dir = os.path.join(settings.MEDIA_ROOT, 'DL/overspeeding_cars/')
            results = process_video(video_path, classifier_path, output_dir)
            return render(request, 'DL/result.html', {'results': results})
    else:
        form = VideoUploadForm()
    return render(request, 'DL/vsd.html', {'form': form})