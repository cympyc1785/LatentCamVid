import os
import json
import csv
from config import cfg

def make_csv(input_root, output_csv):
    rows = []

    for idx, data_name in enumerate(sorted(os.listdir(input_root))):
        data_dir = os.path.join(input_root, data_name)
        if not os.path.isdir(data_dir):
            continue

        # video_path = os.path.join(data_dir, "video_input.mp4")
        video_path = os.path.join(data_dir, "static_video_input.mp4")
        scene_path = os.path.join(data_dir, "scene.pt")
        camera_path = os.path.join(data_dir, "cameras.json")
        prompt_json = os.path.join(data_dir, "prompt.json")

        if not all(os.path.exists(p) for p in [video_path, scene_path, camera_path, prompt_json]):
            print(f"[SKIP] missing file in {data_name}")
            continue

        with open(prompt_json, "r", encoding="utf-8") as f:
            prompt = json.load(f)["prompt_camera"]

        rows.append({
            "video": f"{data_name}/video_input.mp4",
            # "video": f"{data_name}/static_video_input.mp4",
            "prompt": prompt,
            "input_scene": f"{data_name}/scene.pt",
            "input_camera": f"{data_name}/cameras.json",
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "prompt", "input_scene", "input_camera"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")


make_csv(cfg.dataset_dir, '../dataset/metadata_static_1K_camera.csv')
