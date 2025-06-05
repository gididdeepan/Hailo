
import subprocess
import degirum as dg
import sys
import cv2

import time
def get_sys_info():
    try:
        # Run the CLI command and capture the output
        result = subprocess.run(
            ["degirum", "sys-info"], capture_output=True, text=True, check=True
        )
        print(result.stdout)  # Print the command output
    except subprocess.CalledProcessError as e:
        print("Error executing 'degirum sys-info':", e.stderr)
    except FileNotFoundError:
        print(
            "Error: 'degirum' command not found. Make sure DeGirum PySDK is installed."
        )
    except Exception as e:
        print(f"Unexpected error while getting system info: {e}")


if __name__ == "__main__":
    try:
        print("System information:")
        get_sys_info()

        # Check supported devices
        try:
            supported_devices = dg.get_supported_devices(inference_host_address="@local")
        except Exception as e:
            print(f"Error fetching supported devices: {e}")
            sys.exit(1)

        print("Supported RUNTIME/DEVICE combinations:", list(supported_devices))

        # Determine appropriate device_type
        if "HAILORT/HAILO8L" in supported_devices:
            device_type = "HAILORT/HAILO8L"
        elif "HAILORT/HAILO8" in supported_devices:
            device_type = "HAILORT/HAILO8"
        else:
            print(
                "Hailo device is NOT supported or NOT recognized properly. Please check the installation."
            )
            sys.exit(1)

        print(f"Using device type: {device_type}")

        print("Loading model...")
        inference_host_address = "@local"
        zoo_url = "degirum/hailo"
        token = ""

        model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"

        try:
            model = dg.load_model(
                model_name=model_name,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url,
                token=token,
                device_type=device_type,
            )
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            sys.exit(1)

        # Open video file
        #video_path = "assets/Traffic.mp4"
        video_path = "rtsp://192.168.136.100:554/live/0"
        cap = cv2.VideoCapture(video_path)
        #cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            sys.exit(1)

        print(f"Running inference on video: {video_path}")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot fetch the frame.")
                break
            frame_count += 1

            # Process every 2nd frame (for approx 15 FPS if original is 30 FPS)
            if frame_count % 2 == 0:
                try:
                    resized_frame = cv2.resize(frame, (1920, 1080))
                    start = time.time()
                    results = model(resized_frame)
                    print(results)
                    print("processing time", time.time() - start)
            
                    # Option 1: Access via .results
                    for det in results.results:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        label = det["label"]
                        score = det["score"]
            
                        # Draw bounding box
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(resized_frame, f"{label} {score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                    cv2.imshow("RTSP Stream with Inference", resized_frame)
                    
        
                    # Quit if user presses 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User pressed 'q'. Exiting.")
                        break
        
                except Exception as e:
                    print(f"Error during inference: {e}")
                    break
            
        cap.release()
        print("Video processing completed.")

    
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
