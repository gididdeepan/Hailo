import subprocess
import degirum as dg
import sys
import cv2
import time
import threading
import json
import socket
import os
import signal
from flask import Flask, Response, jsonify, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Authentication configuration
VALID_USERNAME = "admin"
VALID_PASSWORD = "password"
AUTH_REQUIRED = True

# Global variables
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
last_frame_time = time.time()
shutdown_flag = threading.Event()

# Thread management
running_threads = []

# Performance monitoring
performance_stats = {
    'fps': 0,
    'inference_time': 0,
    'processing_time': 0
}

# UDP Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def get_sys_info():
    try:
        result = subprocess.run(["degirum", "sys-info"], capture_output=True, text=True, check=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting system info: {e}")

def generate_frames():
    global latest_frame, last_frame_time
    target_fps = 15
    min_frame_interval = 1 / target_fps
    
    while not shutdown_flag.is_set():
        current_time = time.time()
        time_since_last = current_time - last_frame_time
        
        if time_since_last < min_frame_interval:
            time.sleep(min_frame_interval - time_since_last)
            continue
            
        with frame_lock:
            if latest_frame is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', latest_frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 75,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
            ])
            
            if not ret:
                continue
                
        last_frame_time = current_time
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def send_detections_udp(detections):
    try:
        data = json.dumps({
            'detections': detections,
            'timestamp': time.time(),
            'stats': performance_stats
        }).encode('utf-8')
        udp_socket.sendto(data, (UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"Error sending UDP data: {e}")

def monitor_performance():
    global performance_stats
    frame_count = 0
    start_time = time.time()
    
    while not shutdown_flag.is_set():
        time.sleep(5)
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        with frame_lock:
            performance_stats = {
                'fps': round(fps, 1),
                'frame_count': frame_count,
                'uptime': round(elapsed, 1),
                'detection_count': len(latest_detections),
                'inference_time': performance_stats.get('inference_time', 0),
                'processing_time': performance_stats.get('processing_time', 0)
            }
            
        frame_count = 0
        start_time = time.time()

def process_video_stream():
    global latest_frame, latest_detections, performance_stats
    
    print("Initializing system...")
    get_sys_info()

    try:
        # Initialize camera
        video_path = "rtsp://192.168.136.100:554/live/0"
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        # Load model
        model = dg.load_model(
            model_name="yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1",
            inference_host_address="@local",
            zoo_url="degirum/hailo",
            device_type="HAILORT/HAILO8L"
        )
        model.image_backend = 'opencv'
        model.overlay_show_prob = True
        model.overlay_show_bbox = True
        model.overlay_line_width = 2

        print(f"Processing video: {video_path}")
        frame_counter = 0
        skip_frames = 1
        
        while not shutdown_flag.is_set():
            start_processing = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_counter += 1
            if frame_counter % (skip_frames + 1) != 0:
                continue

            model_frame = cv2.resize(frame, (640, 640))
            
            inference_start = time.time()
            results = model(model_frame)
            print(results)
            inference_time = (time.time() - inference_start) * 1000
            performance_stats['inference_time'] = round(inference_time, 1)
            
            detections = []
            display_frame = cv2.resize(frame, (1280, 720))
            
            for det in results.results:
                x1 = int(det["bbox"][0] * (1280/640))
                y1 = int(det["bbox"][1] * (720/640))
                x2 = int(det["bbox"][2] * (1280/640))
                y2 = int(det["bbox"][3] * (720/640))
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{det['label']} {det['score']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                detections.append({
                    "label": det["label"],
                    "score": float(det["score"]),
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": time.time()
                })

            with frame_lock:
                latest_frame = display_frame
                latest_detections = detections
            
            threading.Thread(
                target=send_detections_udp, 
                args=(detections,), 
                daemon=True
            ).start()
            
            performance_stats['processing_time'] = round(
                (time.time() - start_processing) * 1000, 1
            )

    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        cap.release()
        udp_socket.close()
        print("Video processing stopped")

def start_thread(target, daemon=True):
    t = threading.Thread(target=target, daemon=daemon)
    t.start()
    running_threads.append(t)
    return t

def shutdown_server():
    print("Initiating shutdown...")
    shutdown_flag.set()
    
    # Wait for threads to finish
    for t in running_threads:
        t.join(timeout=1)
    
    # Close all resources
    cv2.destroyAllWindows()
    
    # Exit the application
    os.kill(os.getpid(), signal.SIGINT)

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template_string('''
    <!DOCTYPE html>
<html>
<head>
    <title>Video Stream</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        nav {
            background-color: #333;
            padding: 10px;
            display: flex;
        }
        nav img {
            height: 50px;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        .stream-wrapper {
            display: inline-block;
            margin: 20px auto;
            position: relative;
        }
        .stream-container {
            width: 1280px;
            height: 720px;
            overflow: hidden;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
            background-color: #000;
        }
        img.stream {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .logout-link {
            display: inline-block;
            margin-top: 20px;
            padding: 8px 16px;
            background-color: #f44336;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
        }
        .logout-link:hover {
            background-color: #d32f2f;
        }
    </style>
</head>
<body>
    <nav>
        <img src="{{ url_for('static', filename='ERL-logo.png') }}" alt="ERL Logo">
    </nav>

    <div class="container">
        <h1>ERL SPECTRA Live Person Detection Stream</h1>
        
        <div class="stream-wrapper">
            <div class="stream-container">
                <img class="stream" src="{{ url_for('video_feed') }}">
            </div>
        </div>
        
        <a class="logout-link" href="{{ url_for('logout') }}" onclick="return confirm('Are you sure you want to shutdown the system?')">Shutdown System</a>
    </div>
</body>
</html>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if (request.form.get('username') == VALID_USERNAME and 
            request.form.get('password') == VALID_PASSWORD):
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Login</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .login-container {
                        background-color: white;
                        padding: 30px;
                        border-radius: 8px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                        width: 300px;
                    }
                    h1 {
                        text-align: center;
                        color: #333;
                    }
                    .form-group {
                        margin-bottom: 15px;
                    }
                    label {
                        display: block;
                        margin-bottom: 5px;
                        font-weight: bold;
                    }
                    input[type="text"],
                    input[type="password"] {
                        width: 100%;
                        padding: 8px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        box-sizing: border-box;
                    }
                    input[type="submit"] {
                        width: 100%;
                        padding: 10px;
                        background-color: #333;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    input[type="submit"]:hover {
                        background-color: #555;
                    }
                    .error-message {
                        color: red;
                        text-align: center;
                        margin-top: 10px;
                    }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <h1>Login</h1>
                    <form method="POST">
                        <div class="form-group">
                            <label for="username">Username:</label>
                            <input type="text" id="username" name="username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password:</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <input type="submit" value="Login">
                        <div class="error-message">Invalid username or password</div>
                    </form>
                </div>
            </body>
            </html>
        ''')
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .login-container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    width: 300px;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input[type="text"],
                input[type="password"] {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                input[type="submit"] {
                    width: 100%;
                    padding: 10px;
                    background-color: #333;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #555;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h1>Login</h1>
                <form method="POST">
                    <div class="form-group">
                        <label for="username">Username:</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password:</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <input type="submit" value="Login">
                </form>
            </div>
        </body>
        </html>
    ''')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    threading.Thread(target=shutdown_server).start()
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Shutdown</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    text-align: center;
                }
                .message-container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    max-width: 500px;
                }
                h1 {
                    color: #333;
                }
                p {
                    margin: 20px 0;
                    font-size: 18px;
                }
            </style>
        </head>
        <body>
            <div class="message-container">
                <h1>System Shutting Down</h1>
                <p>All processes are being terminated...</p>
                <p>Please wait a moment before restarting the system.</p>
            </div>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return Response("Unauthorized", status=401)
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    with frame_lock:
        return jsonify({
            'detections': latest_detections,
            'stats': performance_stats
        })

if __name__ == "__main__":
    # Start all threads through our thread manager
    start_thread(monitor_performance)
    start_thread(process_video_stream)
    
    # Start Flask server
    app.run(
        host='0.0.0.0', 
        port=5000, 
        threaded=True, 
        processes=1,
        debug=False
    )
