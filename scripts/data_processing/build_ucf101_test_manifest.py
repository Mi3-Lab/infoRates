import os
import csv

# Path to the UCF-101 video root directory
ucf101_root = "data/UCF101_data/UCF-101"
# Path to official test list
testlist_path = "data/UCF101_data/ucfTrainTestlist/testlist01.txt"
# Path to output manifest
manifest_path = "data/UCF101_data/manifests/ucf101_testlist01.csv"

# Read test video filenames
with open(testlist_path, 'r') as f:
    test_videos = [line.strip() for line in f if line.strip()]

rows = []
for rel_path in test_videos:
    # rel_path format: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
    class_name = rel_path.split('/')[0]
    video_path = os.path.join(ucf101_root, rel_path)
    rows.append([video_path, class_name])

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'label'])
    writer.writerows(rows)

print(f"Manifest written to {manifest_path} with {len(rows)} entries.")
