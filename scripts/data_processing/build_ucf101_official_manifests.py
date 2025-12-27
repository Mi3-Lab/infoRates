import os
import csv

def load_ucf101_split(split_file):
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

ucf101_root = "data/UCF101_data/UCF-101"
trainlist_path = "data/UCF101_data/ucfTrainTestlist/trainlist01.txt"
testlist_path = "data/UCF101_data/ucfTrainTestlist/testlist01.txt"

# Train manifest
train_manifest_path = "data/UCF101_data/manifests/ucf101_trainlist01.csv"
train_rows = []
for line in load_ucf101_split(trainlist_path):
    parts = line.split()
    rel_path = parts[0]  # e.g., ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
    class_name = rel_path.split('/')[0]
    video_path = os.path.join(ucf101_root, rel_path)
    train_rows.append([video_path, class_name])
os.makedirs(os.path.dirname(train_manifest_path), exist_ok=True)
with open(train_manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'label'])
    writer.writerows(train_rows)
print(f"Train manifest written to {train_manifest_path} with {len(train_rows)} entries.")

# Test manifest
test_manifest_path = "data/UCF101_data/manifests/ucf101_testlist01.csv"
test_rows = []
for rel_path in load_ucf101_split(testlist_path):
    class_name = rel_path.split('/')[0]
    video_path = os.path.join(ucf101_root, rel_path)
    test_rows.append([video_path, class_name])
os.makedirs(os.path.dirname(test_manifest_path), exist_ok=True)
with open(test_manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'label'])
    writer.writerows(test_rows)
print(f"Test manifest written to {test_manifest_path} with {len(test_rows)} entries.")
