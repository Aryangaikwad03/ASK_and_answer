from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

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

def is_optimal_pose(measurements):
    """Check if the pose is optimal based on predefined criteria."""
    shoulder_width = measurements['shoulder_width']
    waist_width = measurements['waist_width']
    hip_width = measurements['hip_width']

    # Example criteria for optimal pose
    return (
        shoulder_width >= 40 and shoulder_width <= 50 and
        waist_width / hip_width <= 0.85
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze body measurements from a base64-encoded image."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Calculate scaling factor
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return jsonify({"error": "No pose detected in the image"}), 400

        shoulder_width_pixels = abs(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        ) * image.shape[1]
        scaling_factor = AVERAGE_SHOULDER_WIDTH_CM / shoulder_width_pixels

        # Calculate body measurements
        measurements = calculate_body_measurements(image, scaling_factor)
        if not measurements:
            return jsonify({"error": "Failed to calculate measurements"}), 400

        # Classify body type and size
        body_type = classify_body_type(measurements)
        body_size = classify_body_size(measurements)

        # Check for optimal pose
        is_optimal = is_optimal_pose(measurements)

        # Return results as JSON
        return jsonify({
            "measurements": measurements,
            "body_type": body_type,
            "body_size": body_size,
            "is_optimal_pose": is_optimal
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)