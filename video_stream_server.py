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
from flask import Flask, Response, jsonify, request, redirect, url_for, session, render_template_string, send_from_directory

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Authentication configuration
VALID_USERNAME = "Deepan"
VALID_PASSWORD = "erlspectra"
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

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERL SPECTRA | Live Surveillance</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --success-color: #27ae60;
            --warning-color: #f39c12;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f7fa;
            color: var(--dark-color);
            line-height: 1.6;
        }
        
        .navbar {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo img {
            height: 40px;
        }
        
        .logo h1 {
            font-size: 1.5rem;
            font-weight: 500;
        }
        
        .nav-links {
            display: flex;
            gap: 1.5rem;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links a:hover {
            color: var(--secondary-color);
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 2rem;
        }
        
        .video-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            position: relative;
        }
        
        .video-header {
            padding: 1rem;
            background-color: var(--primary-color);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .video-wrapper {
            width: 100%;
            height: 720px;
            background-color: black;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .video-wrapper img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }
        
        .stats-card, .detections-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }
        
        .card-header h3 {
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--primary-color);
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.8rem;
        }
        
        .stat-label {
            font-weight: 500;
            color: #7f8c8d;
        }
        
        .stat-value {
            font-weight: 700;
        }
        
        .detection-item {
            padding: 0.8rem 0;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        
        .detection-item:last-child {
            border-bottom: none;
        }
        
        .detection-label {
            font-weight: 500;
        }
        
        .detection-confidence {
            font-weight: 700;
            color: var(--success-color);
        }
        
        .btn {
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 4px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background-color: var(--secondary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #2980b9;
        }
        
        .btn-danger {
            background-color: var(--accent-color);
            color: white;
        }
        
        .btn-danger:hover {
            background-color: #c0392b;
        }
        
        .btn-sm {
            padding: 0.4rem 0.8rem;
            font-size: 0.9rem;
        }
        
        .notification {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 1rem;
            background-color: var(--success-color);
            color: white;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            transform: translateX(120%);
            transition: transform 0.3s;
            z-index: 1000;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Scrollable detections container */
        .detections-container {
            max-height: 360px;
            overflow-y: auto;
            padding-right: 8px;
        }
        
        /* Custom scrollbar styling */
        .detections-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .detections-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .detections-container::-webkit-scrollbar-thumb {
            background: #bdc3c7;
            border-radius: 4px;
        }
        
        .detections-container::-webkit-scrollbar-thumb:hover {
            background: #7f8c8d;
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .video-wrapper {
                height: auto;
                aspect-ratio: 16/9;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">
            <img src="{{ url_for('static', filename='ERL-logo.png') }}" alt="ERL Logo">
            
        </div>
        <div class="nav-links">
            <a href="#" class="tooltip">
                <i class="fas fa-user"></i> {{ session.get('username', 'Admin') }}
                <span class="tooltiptext">Logged in as {{ session.get('username', 'Admin') }}</span>
            </a>
            <a href="{{ url_for('logout') }}" onclick="return confirm('Are you sure you want to shutdown the system?')">
                <i class="fas fa-power-off"></i> Shutdown
            </a>
        </div>
    </nav>
    
    <div class="container">
        <div class="dashboard">
            <div class="video-container">
                <div class="video-header">
                    <h2><i class="fas fa-video"></i> Live Camera Feed</h2>
                    <div id="connection-status" style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="status-dot" style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%;"></span>
                        <span>Connected</span>
                    </div>
                </div>
                <div class="video-wrapper">
                    <img src="{{ url_for('video_feed') }}" id="video-feed">
                </div>
            </div>
            
            <div class="sidebar">
                <div class="stats-card">
                    <div class="card-header">
                        <h3><i class="fas fa-chart-line"></i> Performance Stats</h3>
                        <i class="fas fa-sync-alt" id="refresh-stats" style="cursor: pointer;"></i>
                    </div>
                    <div id="stats-content">
                        <div class="stat-item">
                            <span class="stat-label">FPS:</span>
                            <span class="stat-value" id="stat-fps">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Inference Time:</span>
                            <span class="stat-value" id="stat-inference">0 ms</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Processing Time:</span>
                            <span class="stat-value" id="stat-processing">0 ms</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Uptime:</span>
                            <span class="stat-value" id="stat-uptime">0s</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Frame Count:</span>
                            <span class="stat-value" id="stat-frames">0</span>
                        </div>
                    </div>
                </div>
                
                <div class="detections-card">
                    <div class="card-header">
                        <h3><i class="fas fa-bell"></i> Recent Detections</h3>
                        <span class="badge" id="detection-count">0</span>
                    </div>
                    <div class="detections-container" id="detections-content">
                        <!-- Detections will be populated here by JavaScript -->
                        <div style="text-align: center; padding: 1rem; color: #7f8c8d;">
                            <i class="fas fa-spinner fa-spin"></i> Loading detections...
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="notification" id="notification">
        <span id="notification-message"></span>
    </div>
    
    <script>
        // Function to update stats
        function updateStats() {
            fetch('/detections')
                .then(response => response.json())
                .then(data => {
                    // Update performance stats
                    document.getElementById('stat-fps').textContent = data.stats.fps;
                    document.getElementById('stat-inference').textContent = data.stats.inference_time + ' ms';
                    document.getElementById('stat-processing').textContent = data.stats.processing_time + ' ms';
                    document.getElementById('stat-uptime').textContent = data.stats.uptime + 's';
                    document.getElementById('stat-frames').textContent = data.stats.frame_count;
                    
                    // Update detections
                    const detectionsContent = document.getElementById('detections-content');
                    const detectionCount = document.getElementById('detection-count');
                    
                    if (data.detections && data.detections.length > 0) {
                        detectionCount.textContent = data.detections.length;
                        detectionCount.style.backgroundColor = data.detections.length > 0 ? '#e74c3c' : '#2ecc71';
                        
                        let html = '';
                        data.detections.slice(0, 10).forEach(det => {
                            html += `
                                <div class="detection-item">
                                    <span class="detection-label">${det.label}</span>
                                    <span class="detection-confidence">${(det.score * 100).toFixed(1)}%</span>
                                </div>
                            `;
                        });
                        
                        detectionsContent.innerHTML = html;
                    } else {
                        detectionCount.textContent = '0';
                        detectionCount.style.backgroundColor = '#2ecc71';
                        detectionsContent.innerHTML = `
                            <div style="text-align: center; padding: 1rem; color: #7f8c8d;">
                                No detections
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                    document.getElementById('connection-status').innerHTML = `
                        <span class="status-dot" style="height: 10px; width: 10px; background-color: #e74c3c; border-radius: 50%;"></span>
                        <span>Connection Error</span>
                    `;
                });
        }
        
        // Function to show notification
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            const notificationMessage = document.getElementById('notification-message');
            
            notification.style.backgroundColor = type === 'success' ? '#27ae60' : '#e74c3c';
            notificationMessage.textContent = message;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        
        // Check video feed connection
        function checkVideoFeed() {
            const videoFeed = document.getElementById('video-feed');
            videoFeed.onerror = function() {
                document.getElementById('connection-status').innerHTML = `
                    <span class="status-dot" style="height: 10px; width: 10px; background-color: #e74c3c; border-radius: 50%;"></span>
                    <span>Video Feed Error</span>
                `;
                showNotification('Video feed connection lost', 'error');
            };
            
            videoFeed.onload = function() {
                document.getElementById('connection-status').innerHTML = `
                    <span class="status-dot" style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%;"></span>
                    <span>Connected</span>
                `;
            };
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Update stats immediately and then every 2 seconds
            updateStats();
            setInterval(updateStats, 2000);
            
            // Set up refresh button
            document.getElementById('refresh-stats').addEventListener('click', updateStats);
            
            // Check video feed connection
            checkVideoFeed();
            
            // Show welcome notification
            setTimeout(() => {
                showNotification('Successfully connected to surveillance system');
            }, 1000);
        });
    </script>
</body>
</html>
    ''')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return jsonify({'success': True, 'redirect': url_for('index')})
        
        return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    if session.get('logged_in'):
        return redirect(url_for('index'))
    
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERL SPECTRA | Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --success-color: #27ae60;
            --warning-color: #f39c12;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f7fa;
            color: var(--dark-color);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        }
        
        .login-container {
            width: 100%;
            max-width: 400px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .login-header {
            background-color: var(--primary-color);
            color: white;
            padding: 2rem;
            text-align: center;
            position: relative;
        }
        
        .login-header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        
        .login-header p {
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            margin: 0 auto 1rem;
            display: block;
        }
        
        .login-form {
            padding: 2rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--primary-color);
        }
        
        .input-with-icon {
            position: relative;
        }
        
        .input-with-icon i {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #7f8c8d;
        }
        
        .form-control {
            width: 100%;
            padding: 12px 15px 12px 45px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .form-control:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
            outline: none;
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn:hover {
            background-color: #2980b9;
        }
        
        .btn:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .error-message {
            color: var(--accent-color);
            font-size: 0.9rem;
            margin-top: 0.5rem;
            text-align: center;
            display: none;
        }
        
        .footer {
            text-align: center;
            padding: 1rem;
            font-size: 0.8rem;
            color: #7f8c8d;
            border-top: 1px solid #eee;
        }
        
        .footer a {
            color: var(--secondary-color);
            text-decoration: none;
        }
        
        .particles {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
    </style>
</head>
<body>
    <div class="particles" id="particles-js"></div>
    
    <div class="login-container">
        <div class="login-header">
            
            <h1>ERL SPECTRA</h1>
            <p>Surveillance System Login</p>
        </div>
        
        <div class="login-form">
            <form id="loginForm">
                <div class="form-group">
                    <label for="username">Username</label>
                    <div class="input-with-icon">
                        <i class="fas fa-user"></i>
                        <input type="text" id="username" name="username" class="form-control" placeholder="Enter your username" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <div class="input-with-icon">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="password" name="password" class="form-control" placeholder="Enter your password" required>
                    </div>
                </div>
                
                <div class="error-message" id="errorMessage">
                    Invalid username or password
                </div>
                
                <button type="submit" class="btn" id="loginBtn">
                    <span id="btnText">Login</span>
                    <div class="spinner" id="spinner" style="display: none;"></div>
                </button>
            </form>
        </div>
        
        <div class="footer">
            <p>ERL SPECTRA Surveillance System &copy; 2025</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
        // Initialize particles.js
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: "#ffffff" },
                shape: { type: "circle" },
                opacity: { value: 0.5, random: true },
                size: { value: 3, random: true },
                line_linked: { enable: true, distance: 150, color: "#ffffff", opacity: 0.4, width: 1 },
                move: { enable: true, speed: 2, direction: "none", random: true, straight: false, out_mode: "out" }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: { enable: true, mode: "repulse" },
                    onclick: { enable: true, mode: "push" }
                }
            }
        });
        
        // Handle form submission
        document.getElementById('loginForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const loginBtn = document.getElementById('loginBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            const errorMessage = document.getElementById('errorMessage');
            
            // Show loading state
            loginBtn.disabled = true;
            btnText.textContent = 'Authenticating...';
            spinner.style.display = 'block';
            errorMessage.style.display = 'none';
            
            // Simulate API call
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.href = data.redirect;
                } else {
                    errorMessage.textContent = data.message || 'Invalid username or password';
                    errorMessage.style.display = 'block';
                    loginBtn.disabled = false;
                    btnText.textContent = 'Login';
                    spinner.style.display = 'none';
                    
                    // Shake animation for error
                    document.querySelector('.login-container').style.animation = 'shake 0.5s';
                    setTimeout(() => {
                        document.querySelector('.login-container').style.animation = '';
                    }, 500);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                errorMessage.textContent = 'Connection error. Please try again.';
                errorMessage.style.display = 'block';
                loginBtn.disabled = false;
                btnText.textContent = 'Login';
                spinner.style.display = 'none';
            });
        });
        
        // Add shake animation to CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                20%, 40%, 60%, 80% { transform: translateX(5px); }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
    ''')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    threading.Thread(target=shutdown_server).start()
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Shutdown</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --success-color: #27ae60;
            --warning-color: #f39c12;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f7fa;
            color: var(--dark-color);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            text-align: center;
            background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        }
        
        .shutdown-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            padding: 3rem;
            max-width: 500px;
            width: 90%;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .shutdown-icon {
            font-size: 4rem;
            color: var(--accent-color);
            margin-bottom: 1.5rem;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        p {
            margin-bottom: 2rem;
            color: #7f8c8d;
        }
        
        .progress-container {
            width: 100%;
            background-color: #f1f1f1;
            border-radius: 5px;
            margin-bottom: 2rem;
        }
        
        .progress-bar {
            width: 0%;
            height: 10px;
            background-color: var(--accent-color);
            border-radius: 5px;
            transition: width 0.3s;
        }
        
        .footer {
            margin-top: 2rem;
            font-size: 0.8rem;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="shutdown-container">
        <div class="shutdown-icon">
            <i class="fas fa-power-off"></i>
        </div>
        <h1>System Shutting Down</h1>
        <p>All surveillance processes are being terminated safely. Please wait...</p>
        
        <div class="progress-container">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        
        <div class="footer">
            <p>ERL SPECTRA Surveillance System &copy; 2023</p>
        </div>
    </div>
    
    <script>
        // Animate progress bar
        let progress = 0;
        const progressBar = document.getElementById('progressBar');
        const interval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    window.location.href = '/login';
                }, 500);
            }
        }, 200);
    </script>
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
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
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
