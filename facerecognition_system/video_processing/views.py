from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import urllib.request
import json
import os
import requests
import face_recognition
from pathlib import Path
from django.conf import settings
import time
from .models import FaceRecognition
from django.core.exceptions import ObjectDoesNotExist
import shutil

camera_status = True
camera = None
url = "http://192.168.1.15/cam-mid.jpg"

@csrf_exempt
def toggle_pc_cam(request):
    global camera_status
    if request.method == 'POST':
        data = json.loads(request.body)
        status = data.get('status')
        
        if status == 'on':
            start_camera()  
            camera_status = True
            print("Camera turned ON")
        elif status == 'off':
            stop_camera()
            camera_status = False
            print("Camera turned OFF")
        
        return JsonResponse({'status': camera_status})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

recognized_names_global = []

def genCam():
    global recognized_names_global
    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        recognized_names_global = []

        for (x, y, w, h) in face_locations:
            face_img = frame[y:y+h, x:x+w]
            rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_face_img)

            if face_encoding:
                best_match_percentage = 0
                matched_name = "Unknown"
                frame_color = (0, 0, 255)

                for db_face in FaceRecognition.objects.all():
                    db_encoding = np.frombuffer(db_face.encoding, dtype=np.float64)
                    distance = face_recognition.face_distance([db_encoding], face_encoding[0])[0]
                    match_percentage = max(0, 100 * (1 - distance / 0.6))

                    if match_percentage > best_match_percentage:
                        best_match_percentage = match_percentage
                        matched_name = db_face.name
                        frame_color = (255, 0, 0) if match_percentage < 70 else (0, 255, 0)

                if best_match_percentage >= 30:
                    recognized_names_global.append(matched_name)

                label = f"{matched_name} ({best_match_percentage:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), frame_color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, frame_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    camera.release()

def get_pc_cam(request):
    return StreamingHttpResponse(genCam(), content_type='multipart/x-mixed-replace; boundary=frame')

def pc_cam(request):
    return render(request, "video_processing/pc_cam.html", {'camera_status': camera_status})

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def genESP():
    global recognized_names_global

    while True:
        img = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_names_global = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matched_name = "Unknown"
            frame_color = (0, 0, 255) 
            best_match_percentage = 0 

            for db_face in FaceRecognition.objects.all():
                db_encoding = np.frombuffer(db_face.encoding, dtype=np.float64)
                
                distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                
                match_percentage = max(0, 100 * (1 - distance / 0.6))

                if match_percentage > best_match_percentage:
                    best_match_percentage = match_percentage
                    matched_name = db_face.name
                    frame_color = (255, 0, 0) if match_percentage < 70 else (0, 255, 0)

            if best_match_percentage >= 30:
                recognized_names_global.append(matched_name)

            label = f"{matched_name} ({best_match_percentage:.1f}%)"
            cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, frame_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@csrf_exempt
def attendance(request):
    global recognized_names_global

    if request.method == 'POST':
        return JsonResponse({'recognized_names': recognized_names_global})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def genESP_no_detect():
    global url

    while True:
        img = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_esp_no_detect(request):
    return StreamingHttpResponse(genESP_no_detect(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_esp_cam(request):
    return StreamingHttpResponse(genESP(), content_type='multipart/x-mixed-replace; boundary=frame')

def esp_cam(request):
    return render(request, "video_processing/esp_cam.html")

def index(request):
    return render(request, "video_processing/home.html")

@csrf_exempt
def train_faces(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        name_FE = body.get('name', '')
        mssv = body.get('id', '')
        try:
            existing_user = FaceRecognition.objects.get(mssv=mssv)
            print(f"Người dùng với MSSV {mssv} đã tồn tại.")
            return 
        except ObjectDoesNotExist:
            pass 

        face_encodings = []
        for i in range(1, 31):
            img_path = os.path.join("dataset", f"{mssv}_{i}.jpg")
            if os.path.exists(img_path):
                img = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(img)
                if encoding:
                    face_encodings.append(encoding[0])

        if not face_encodings:
            print("Không tìm thấy khuôn mặt nào trong các bức ảnh.")
            return

        average_encoding = np.mean(face_encodings, axis=0)

        new_face = FaceRecognition(mssv=mssv, name=name_FE, encoding=average_encoding.tobytes())
        new_face.save()
        print(f"Đã lưu thông tin người dùng: {name_FE} với MSSV {mssv} vào cơ sở dữ liệu.")
        return JsonResponse({'status': 'OK'})
    return JsonResponse({'status': 'FAIL'})

@csrf_exempt
def capture(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        name_FE = body.get('name', '')
        mssv = body.get('id', '')

        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir) 
        os.makedirs(dataset_dir)

        stream_url = "http://127.0.0.1:8000/video/get_esp_cam_no_detect/"
        stream = requests.get(stream_url, stream=True)

        face_count = 0  
        byte_data = b''

        while face_count < 30:
          
            for chunk in stream.iter_content(chunk_size=1024):
                byte_data += chunk
                a = byte_data.find(b'\xff\xd8')
                b = byte_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg_data = byte_data[a:b+2]
                    byte_data = byte_data[b+2:]
                    img_np = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) == 0:
                        print("Không phát hiện khuôn mặt nào.")
                    
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_count += 1
                        face_filename = os.path.join("dataset", f"{mssv}_{face_count}.jpg")
                        cv2.imwrite(face_filename, face_img)
                        if face_count >= 30:
                            break

                if face_count >= 30:
                    break
            time.sleep(1)

        return JsonResponse({'status': 'Da Nhan'})
    
    return JsonResponse({'status': 'Phương thức không hợp lệ'}, status=405)
