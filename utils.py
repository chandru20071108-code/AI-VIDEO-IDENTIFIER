import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

# MediaPipe may expose solutions at different import paths depending on install
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    MEDIA_PIPE_AVAILABLE = True
except Exception:
    face_mesh = None
    MEDIA_PIPE_AVAILABLE = False


def analyze_ai_anomalies(frames):
    """
    Analyzes visual and temporal anomalies to detect AI generation (e.g. Sora, Gen-2).
    Returns a probability (0.0 to 1.0) of being AI-generated based on structural physics.
    """
    import numpy as np
    if len(frames) < 2:
        return 0.5

    laplacian_vars = []
    structural_diffs = []
    
    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacian_vars.append(lap_var)
        
        if i > 0:
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            # Find the absolute pixel shift across 1 second
            diff = cv2.absdiff(gray, prev_gray)
            structural_diffs.append(np.mean(diff))

    avg_lap = np.mean(laplacian_vars)
    avg_diff = np.mean(structural_diffs)
    
    anomaly_score = 0.0
    
    # 1. Sharpness/Noise Anomaly
    # Generative AI often hallucinates universally mathematically perfect smoothness or sharp noise.
    if avg_lap < 50: 
        anomaly_score += 0.3 # Unnaturally smooth
    elif avg_lap > 1000:
        anomaly_score += 0.2 # Unnaturally noisy
        
    # 2. Temporal Morphing
    # At 1 Frame-Per-Second, real physics constrain light shifting. AI generation often 
    # exhibits physical hallucination ("morphing"), shifting extreme amounts of raw pixel geometry.
    if avg_diff > 45:
        # High likelihood of hallucinated motion physics (like the alligator video)
        anomaly_score += 0.6
    elif avg_diff > 30:
        anomaly_score += 0.3
        
    # Scale mathematically to 0-1 bounds (Base assumption of Real starts at ~10% uncertainty)
    final_prob = min(0.95, max(0.05, anomaly_score + 0.10))
    
    return final_prob


def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from video at specified frame rate.
    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Number of frames to extract per second.
    Returns:
        list: List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate) if frame_rate > 0 else 1

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess a frame: resize, normalize, convert to tensor.
    Args:
        frame (np.array): Input frame.
        target_size (tuple): Target size for resizing.
    Returns:
        torch.Tensor: Preprocessed frame tensor with ImageNet normalization.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize
    frame_resized = cv2.resize(frame_rgb, target_size)
    # Normalize to [0, 1]
    frame_normalized = frame_resized / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame_normalized = (frame_normalized - mean) / std
    
    # Convert to tensor and add batch dimension
    frame_tensor = torch.from_numpy(frame_normalized).float().permute(2, 0, 1).unsqueeze(0)
    return frame_tensor


def detect_facial_landmarks(frame):
    """
    Detect facial landmarks using MediaPipe, if available.
    """
    if not MEDIA_PIPE_AVAILABLE or face_mesh is None:
        return None
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    return None

def detect_blinking(landmarks):
    """
    Detect blinking based on eye aspect ratio.
    Args:
        landmarks (list): Facial landmarks.
    Returns:
        bool: True if blinking detected.
    """
    if not landmarks:
        return False

    # Eye landmarks indices (approximate for left and right eyes)
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(eye_points):
        # Calculate EAR
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    left_ear = eye_aspect_ratio([landmarks[i] for i in left_eye_indices])
    right_ear = eye_aspect_ratio([landmarks[i] for i in right_eye_indices])

    # Threshold for blinking
    ear_threshold = 0.25
    return left_ear < ear_threshold or right_ear < ear_threshold

def visualize_frame(frame, landmarks=None, blinking=False):
    """
    Visualize frame with landmarks and blinking indicator.
    Args:
        frame (np.array): Input frame.
        landmarks (list): Facial landmarks.
        blinking (bool): Blinking status.
    Returns:
        np.array: Visualized frame.
    """
    vis_frame = frame.copy()
    if landmarks:
        h, w, _ = vis_frame.shape
        for x, y, z in landmarks:
            cv2.circle(vis_frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
    if blinking:
        cv2.putText(vis_frame, "BLINKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return vis_frame