import numpy as np
import os
from django.shortcuts import render, redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, StreamingHttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm, LoginForm
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
import threading
from django.views.decorators import gzip

# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            form = LoginForm(request, data=request.POST)
            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')
                user = authenticate(request, username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('home')
                else:
                    messages.info(request, 'Username or password is incorrect')
            else:
                 messages.info(request, 'Username or password is incorrect')
        else:
            form = LoginForm()

        context = {'form': form}
        return render(request, 'accounts/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'accounts/home.html')

def home2(request):
    return render(request, 'accounts/index1.html')

def home3(request):
    return render(request, 'accounts/index.html')

@login_required(login_url='login')
def predictImage(request):
    KNOWN_DISTANCE = 30.0  # Known distance to the object (in cm)
    KNOWN_DIAMETER = 10.0  # Known diameter of the object (in cm)
    FOCAL_LENGTH = None  # Will be calculated

    def calculate_focal_length(known_distance, known_diameter, pixel_width):
        return (pixel_width * known_distance) / known_diameter

    def calculate_distance(real_diameter, focal_length, pixel_width):
        return (real_diameter * focal_length) / pixel_width

    # Load the YOLOv8 model
    model = YOLO('yolov8_best.pt')

    # Capture a frame for focal length calculation
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Perform inference on the frame to get the bounding box of the known object
    results = model(frame)

    # Assuming the object is detected
    for result in results:
        if result.boxes:
            # Extract the width of the bounding box for the known object
            x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
            pixel_width = x2 - x1

            # Calculate the focal length
            FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_DIAMETER, pixel_width)
            print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} pixels")
            break

    # List to store apple count per frame
    apple_count_list = []

    # Now use the calculated focal length for distance measurement
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame, conf=0.1)
        apple_count = 0  # Count apples per frame

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    # Extract the width of the bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    pixel_width = x2 - x1

                    # Calculate the distance
                    distance = calculate_distance(KNOWN_DIAMETER, FOCAL_LENGTH, pixel_width)

                    # Draw the bounding box and distance on the frame
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    frame = cv2.putText(frame, f'Object: {distance:.2f} cm', (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    apple_count += 1  # Increment apple count

        # Store the apple count in the list
        apple_count_list.append(apple_count)
        # print(f"Apples detected: {apple_count}")

        # Display the frame with detections
        cv2.imshow('YOLOv8 Inference', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Get the last apple count safely
    last_count = apple_count_list[-1] if apple_count_list else 0
    print("Apple count per frame:", last_count)

    return render(request, "accounts/result.html", {'class': last_count})

@login_required(login_url='login')
def predictImage1(request):
    if "document" not in request.FILES:
        return HttpResponse("Error: No file uploaded")
    
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    file_extension = os.path.splitext(fileObj.name)[1].lower()
    
    # Constants
    KNOWN_DISTANCE = 30.0
    KNOWN_DIAMETER = 10.0
    FOCAL_LENGTH = 500.0
    
    def calculate_focal_length(known_distance, known_diameter, pixel_width):
        return (pixel_width * known_distance) / known_diameter
    
    def calculate_distance(real_diameter, focal_length, pixel_width):
        # Enhanced distance calculation with error correction
        distance = (real_diameter * focal_length) / pixel_width
        # Add error correction factor based on distance
        if distance < 50:  # Close objects
            return distance * 0.95  # Slight reduction for close objects
        elif distance > 200:  # Far objects
            return distance * 1.05  # Slight increase for far objects
        return distance

    def draw_text_with_background(img, text, position, font_scale, color, thickness, bg_color=(0, 0, 0)):
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calculate background rectangle coordinates
        x, y = position
        bg_rect = (x, y - text_height - baseline, text_width, text_height + 2 * baseline)
        
        # Draw background rectangle
        cv2.rectangle(img, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), bg_color, -1)
        
        # Draw text
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return img

    model = YOLO('yolov8_best.pt')
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    if file_extension in image_extensions:
        filename = f"uploaded_image{file_extension}"
        filePathName = fs.save(filename, fileObj)
        filePathName = fs.url(filePathName)
        image_path = "." + filePathName
        
        print("Uploaded image:", fileObj.name)
        
        image = cv2.imread(image_path)
        if image is None:
            return HttpResponse("Error: Could not read image file")
        
        # Preprocess the image
        # Resize image if it's too large
        max_dimension = 1280
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Enhance image contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Perform inference with lower confidence threshold
        results = model(image, conf=0.1)  # Lower confidence threshold
        apple_count = 0
        
        for result in results:
            if result.boxes:
                if apple_count == 0 and len(result.boxes) > 0:
                    x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
                    pixel_width = x2 - x1
                    FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_DIAMETER, pixel_width)
                    print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} pixels")
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    pixel_width = x2 - x1
                    distance = calculate_distance(KNOWN_DIAMETER, FOCAL_LENGTH, pixel_width)
                    
                    # Draw bounding box with thicker lines
                    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    # Add apple count to each detection
                    image = cv2.putText(image, f'Apple {apple_count + 1}: {distance:.2f} cm', 
                                      (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    apple_count += 1
        
        # Add total count to the image
        cv2.putText(image, f'Total Apples: {apple_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert the processed image to base64 for display
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return render(request, "accounts/result.html", {
            'class': apple_count,
            'image_data': img_str,
            'is_image': True
        })
        
    elif file_extension in video_extensions:
        filename = f"uploaded_video{file_extension}"
        filePathName = fs.save(filename, fileObj)
        filePathName = fs.url(filePathName)
        video_path = "." + filePathName
        
        print("Uploaded video:", fileObj.name)
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return HttpResponse("Error: Could not read video file")
        
        # Initialize tracker
        tracker = AppleTracker(max_disappeared=10, max_distance=100)
        
        # Calculate focal length from first frame
        results = model(frame)
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
                pixel_width = x2 - x1
                FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_DIAMETER, pixel_width)
                print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} pixels")
                break
        
        processed_frames = []
        max_apples_detected = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            height, width = frame.shape[:2]
            if max(height, width) > 1280:
                scale = 1280 / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # Enhance frame contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            results = model(frame, conf=0.1)
            current_boxes = []
            
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        current_boxes.append((x1, y1, x2, y2))
            
            # Update tracker
            tracked_objects = tracker.update(current_boxes)
            
            # Draw tracked objects
            for object_id, (centroid, box) in tracked_objects.items():
                x1, y1, x2, y2 = box
                pixel_width = x2 - x1
                distance = calculate_distance(KNOWN_DIAMETER, FOCAL_LENGTH, pixel_width)
                
                # Draw bounding box with thicker lines
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                
                # Draw object ID and distance
                text = f"Apple {object_id + 1}: {distance:.1f} cm"
                frame = draw_text_with_background(
                    frame,
                    text,
                    (int(x1), int(y1) - 10),
                    0.7,
                    (255, 255, 255),
                    2,
                    (0, 128, 0)
                )
                
                # Draw centroid
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
            
            # Update max count
            current_count = len(tracked_objects)
            max_apples_detected = max(max_apples_detected, current_count)
            
            # Add live count to frame with background
            frame = draw_text_with_background(
                frame,
                f'Current Count: {current_count}',
                (10, 40),
                1.0,
                (255, 255, 255),
                2,
                (0, 128, 0)
            )
            
            frame = draw_text_with_background(
                frame,
                f'Max Count: {max_apples_detected}',
                (10, 80),
                1.0,
                (255, 255, 255),
                2,
                (0, 0, 128)
            )
            
            # Add average distance information
            if current_count > 0:
                avg_distance = sum(calculate_distance(KNOWN_DIAMETER, FOCAL_LENGTH, x2-x1) 
                                 for _, (_, (x1, y1, x2, y2)) in tracked_objects.items()) / current_count
                frame = draw_text_with_background(
                    frame,
                    f'Average Distance: {avg_distance:.1f} cm',
                    (10, 120),
                    0.8,
                    (255, 255, 255),
                    2,
                    (128, 0, 0)
                )
            
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_str = base64.b64encode(buffer).decode('utf-8')
            
            # Add frame data with count information
            frame_data = {
                'image': frame_str,
                'count': current_count
            }
            processed_frames.append(frame_data)
        
        cap.release()
        
        # Use the maximum count detected as the final count
        final_count = max_apples_detected
        print("Maximum unique apples detected in video:", final_count)
        
        return render(request, "accounts/result.html", {
            'class': final_count,
            'frames': processed_frames,
            'is_video': True,
            'max_count': max_apples_detected
        })
    
    else:
        return HttpResponse(f"Error: Unsupported file type {file_extension}. Please upload an image or video file.")

class AppleTracker:
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object IDs and their centroids
        self.disappeared = {}  # Dictionary to store how long each object has been missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counted_apples = set()  # Set to store IDs of apples that have been counted

    def register(self, centroid, box):
        self.objects[self.next_object_id] = (centroid, box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        input_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))
            input_boxes.append((x1, y1, x2, y2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.sqrt(
                        (object_centroids[i][0] - input_centroids[j][0]) ** 2 +
                        (object_centroids[i][1] - input_centroids[j][1]) ** 2
                    )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], input_boxes[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_boxes[col])

        return self.objects

@gzip.gzip_page
@login_required(login_url='login')
def live_detection(request):
    from ultralytics import YOLO
    import cv2
    import base64
    import time

    model = YOLO('yolov8_best.pt')
    cap = cv2.VideoCapture(0)
    max_count = 0

    def gen():
        nonlocal max_count
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.1)
            apple_count = 0
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        # Only count apples (class 0)
                        if hasattr(box, 'cls'):
                            class_id = int(box.cls[0].cpu().numpy())
                        else:
                            class_id = 0  # fallback if not available
                        if class_id == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            apple_count += 1
            max_count = max(max_count, apple_count)
            # Draw counts
            cv2.putText(frame, f'Current: {apple_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f'Max: {max_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

@login_required(login_url='login')
def live_detection_page(request):
    return render(request, 'accounts/live_detection.html')