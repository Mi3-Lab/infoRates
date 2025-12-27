import os
import csv

# Path to the UCF-101 video root directory
ucf101_root = "data/UCF101_data/UCF-101"
# Path to output manifest
manifest_path = "data/UCF101_data/manifests/ucf101_full.csv"

rows = []
for class_name in sorted(os.listdir(ucf101_root)):
    class_dir = os.path.join(ucf101_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in sorted(os.listdir(class_dir)):
        if fname.endswith('.avi'):
            video_path = os.path.join(ucf101_root, class_name, fname)
            rows.append([video_path, class_name])

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'label'])
    writer.writerows(rows)

print(f"Manifest written to {manifest_path} with {len(rows)} entries.")
