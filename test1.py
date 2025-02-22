import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Directory to save screenshots
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)

# Assume average shoulder width (in cm)
AVERAGE_SHOULDER_WIDTH_CM = 40

def convert_pixels_to_cm(pixel_distance, scaling_factor):
    """Convert pixels to real-world centimeters using a scaling factor."""
    return pixel_distance * scaling_factor

def extract_body_points(landmarks, image_shape):
    """Extract key body points for shoulders, chest, waist, and hips."""
    height, width, _ = image_shape
    key_points = {}

    def get_coordinates(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    # Shoulders
    key_points['left_shoulder'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
    key_points['right_shoulder'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])

    # Chest (use shoulders and hips to estimate)
    key_points['left_chest'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
    key_points['right_chest'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])

    # Waist (use hips as a proxy)
    key_points['left_waist'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
    key_points['right_waist'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])

    # Hips
    key_points['left_hip'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
    key_points['right_hip'] = get_coordinates(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])

    return key_points

def calculate_body_measurements(frame, scaling_factor):
    """Compute shoulder width, chest, waist, and hip measurements in cm."""
    # Process the frame with MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None  # No pose detected

    # Extract key body points
    key_points = extract_body_points(results.pose_landmarks, frame.shape)

    # Calculate pixel distances
    shoulder_width_pixels = abs(key_points['left_shoulder'][0] - key_points['right_shoulder'][0])
    chest_width_pixels = abs(key_points['left_chest'][0] - key_points['right_chest'][0])
    waist_width_pixels = abs(key_points['left_waist'][0] - key_points['right_waist'][0])
    hip_width_pixels = abs(key_points['left_hip'][0] - key_points['right_hip'][0])

    # Convert pixels to cm
    shoulder_cm = convert_pixels_to_cm(shoulder_width_pixels, scaling_factor)
    chest_cm = convert_pixels_to_cm(chest_width_pixels, scaling_factor)
    waist_cm = convert_pixels_to_cm(waist_width_pixels, scaling_factor)
    hip_cm = convert_pixels_to_cm(hip_width_pixels, scaling_factor)

    # Manually adjust measurements for better accuracy
    shoulder_cm += 2  # Add 2 cm to shoulder width
    chest_cm += 5     # Add 5 cm to chest width
    waist_cm += 5     # Add 5 cm to waist width
    hip_cm += 5       # Add 5 cm to hip width

    return {
        'shoulder_width': round(shoulder_cm, 2),
        'chest_width': round(chest_cm, 2),
        'waist_width': round(waist_cm, 2),
        'hip_width': round(hip_cm, 2)
    }

def classify_body_type(measurements):
    """Classify body type based on shoulder-to-hip and waist-to-hip ratios."""
    shoulder_to_hip_ratio = measurements['shoulder_width'] / measurements['hip_width']
    waist_to_hip_ratio = measurements['waist_width'] / measurements['hip_width']

    # Classify based on the provided threshold
    if 0.85 <= shoulder_to_hip_ratio <= 1.15 and 0.60 <= waist_to_hip_ratio <= 0.90:
        return "Hourglass"
    elif shoulder_to_hip_ratio < 0.85 and waist_to_hip_ratio > 0.90:
        return "Pear (Triangle)"
    elif shoulder_to_hip_ratio > 1.15 and waist_to_hip_ratio > 0.90:
        return "Apple"
    elif 0.85 <= shoulder_to_hip_ratio <= 1.15 and waist_to_hip_ratio > 0.90:
        return "Rectangle"
    elif shoulder_to_hip_ratio > 1.15 and waist_to_hip_ratio < 0.90:
        return "Inverted Triangle"
    elif 0.85 <= shoulder_to_hip_ratio <= 1.15 and waist_to_hip_ratio > 0.85:
        return "Oval"
    else:
        return "Undefined"

def classify_body_size(measurements):
    """Classify body size into Small, Medium, Large, Extra Large, or Extra Extra Large."""
    shoulder = measurements['shoulder_width']
    chest = measurements['chest_width']
    waist = measurements['waist_width']
    hip = measurements['hip_width']

    # Determine size based on shoulder width (can use other measurements as well)
    if shoulder < 40:
        return "Small (S)"
    elif 40 <= shoulder < 45:
        return "Medium (M)"
    elif 45 <= shoulder < 50:
        return "Large (L)"
    elif 50 <= shoulder < 55:
        return "Extra Large (XL)"
    else:
        return "Extra Extra Large (XXL)"

def are_landmarks_visible(landmarks):
    """Check if all required landmarks are visible."""
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    for landmark in required_landmarks:
        if landmarks.landmark[landmark].visibility < 0.5:  # Threshold for visibility
            return False
    return True

def is_optimal_pose(measurements):
    """Define criteria for optimal pose."""
    if 30 <= measurements['shoulder_width'] <= 60:
        return True
    return False

# Main loop for real-time processing
optimal_measurements = None
optimal_frame = None
stable_frames = 0
required_stable_frames = 30

scaling_factor = None

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks and are_landmarks_visible(results.pose_landmarks):
        if scaling_factor is None:
            shoulder_width_pixels = abs(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            ) * frame.shape[1]
            scaling_factor = AVERAGE_SHOULDER_WIDTH_CM / shoulder_width_pixels

        measurements = calculate_body_measurements(frame, scaling_factor)

        if measurements:
            body_type = classify_body_type(measurements)
            body_size = classify_body_size(measurements)  # Get body size

            # Display measurements, body type, body size, and waist size on screen
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 255)
            font_thickness = 2

            cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width']} cm", (10, 30), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Chest Width: {measurements['chest_width']} cm", (10, 70), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Waist Width: {measurements['waist_width']} cm", (10, 110), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Hip Width: {measurements['hip_width']} cm", (10, 150), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Body Type: {body_type}", (10, 190), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Body Size: {body_size}", (10, 230), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f"Waist Size: {measurements['waist_width']} cm", (10, 270), font, font_scale, font_color, font_thickness)

            if is_optimal_pose(measurements):
                stable_frames += 1
                if stable_frames >= required_stable_frames:
                    optimal_measurements = measurements
                    print("Final Measurements:", optimal_measurements)
                    print("Body Type:", body_type)
                    print("Body Size:", body_size)  # Print body size
                    print("Waist Size:", measurements['waist_width'], "cm")  # Print waist size
                    break
            else:
                stable_frames = 0
    else:
        stable_frames = 0

    cv2.imshow("Real-Time Body Measurements", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()