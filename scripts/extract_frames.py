import cv2
import os

def extract_frames(video_path, output_dir, stride, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % stride == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count}_stride_{stride}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()

# Example usage
video_path = "/home/wesleyferreiramaia/data/infoRates/data/UCF101_data/UCF-101/YoYo/v_YoYo_g01_c01.avi"
output_dir = "/home/wesleyferreiramaia/data/infoRates/docs/figures"
os.makedirs(output_dir, exist_ok=True)

# Dense sampling (stride 1)
extract_frames(video_path, output_dir, stride=1, max_frames=5)

# Sparse sampling (stride 16, aliasing)
extract_frames(video_path, output_dir, stride=16, max_frames=5)