import cv2
import os
import random


def extract_frames(video_path, output_dir, num_frames=5):
    """
    Extract frames from a video and save them as images.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return

    # Select indices for extraction
    indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    count = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count in indices:
            # Resize for the model (224x224)
            frame_resized = cv2.resize(frame, (224, 224))
            filename = f"{os.path.basename(video_path).split('.')[0]}_{frame_idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame_resized)
            frame_idx += 1
            if frame_idx >= num_frames:
                break
        
        count += 1
    
    cap.release()

def main():
    input_dir = "FaceForensics"
    output_base_dir = "data"
    
    # Create directory structure
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(output_base_dir, split, label), exist_ok=True)
    
    # Get list of videos
    videos = sorted([f for f in os.listdir(input_dir) if f.endswith(".mp4")])
    
    # Define labels based on the discussed convention (0-499 Real, 500-999 Fake)
    # Adjust this if labels are different!
    real_videos = [v for v in videos if int(v.split(".")[0]) < 500]
    fake_videos = [v for v in videos if int(v.split(".")[0]) >= 500]
    
    # Shuffle for train/val split
    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    # 80/20 split
    split_idx_real = int(len(real_videos) * 0.8)
    split_idx_fake = int(len(fake_videos) * 0.8)
    
    dataset_splits = {
        "train": {
            "real": real_videos[:split_idx_real],
            "fake": fake_videos[:split_idx_fake]
        },
        "val": {
            "real": real_videos[split_idx_real:],
            "fake": fake_videos[split_idx_fake:]
        }
    }
    
    # Process each split
    for split, labels in dataset_splits.items():
        for label, video_list in labels.items():
            print(f"Processing {split}/{label} ({len(video_list)} videos)...")
            output_dir = os.path.join(output_base_dir, split, label)
            for i, video_name in enumerate(video_list):
                if i % 50 == 0:
                    print(f"  ...processed {i}/{len(video_list)} videos")
                video_path = os.path.join(input_dir, video_name)
                extract_frames(video_path, output_dir, num_frames=5)


    print("Data preprocessing complete! Folder 'data/' is ready for training.")

if __name__ == "__main__":
    main()
