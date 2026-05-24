import os
from pathlib import Path
import av
import csv

# Paths
ucf101_root = "data/UCF101_data/UCF-101"
testlist_path = "data/UCF101_data/ucfTrainTestlist/testlist01.txt"
out_dir = "data/UCF101_data/UCF101_50f_testset"
manifest_path = "data/UCF101_data/manifests/ucf101_50f_testset.csv"
target_frames = 50

# Read test video filenames
with open(testlist_path, 'r') as f:
    test_videos = [line.strip() for line in f if line.strip()]

rows = []
for rel_path in test_videos:
    class_name = rel_path.split('/')[0]
    video_path = os.path.join(ucf101_root, rel_path)
    try:
        container = av.open(video_path)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        n_frames = len(frames)
        segments = n_frames // target_frames
        out_label_dir = Path(out_dir) / class_name
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(segments):
            seg_frames = frames[i * target_frames:(i + 1) * target_frames]
            seg_path = out_label_dir / f"{Path(video_path).stem}_seg{i:02d}.mp4"
            out = av.open(str(seg_path), "w")
            stream = out.add_stream("mpeg4", rate=25)
            stream.width, stream.height = seg_frames[0].shape[1], seg_frames[0].shape[0]
            for frame in seg_frames:
                frame_av = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame_av):
                    out.mux(packet)
            # Flush stream
            for packet in stream.encode():
                out.mux(packet)
            out.close()
            rows.append([str(seg_path), class_name])
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'label'])
    writer.writerows(rows)

print(f"Manifest written to {manifest_path} with {len(rows)} entries.")
