from model_evaluator.bbox_generator.keyframe_interpolator import KeyframeInterpolator
from model_evaluator.utils.json_file_reader import read_json, write_json
import glob
from pathlib import Path

def create_bboxes_from_keyframes_file(file_path:str):
    bboxes = read_json(f"keyframes/{file_path}.json")

    all_boxes = []
    for ped_path in bboxes:
        ki = KeyframeInterpolator(ped_path)
        all_boxes += ki.all_frames

    all_boxes.sort(key=lambda x: x["frame"])

    write_json(f"scene_boxes/{file_path}.json", all_boxes)


if __name__ == "__main__":
    paths = glob.glob("keyframes/*")

    for p in paths:
        create_bboxes_from_keyframes_file(Path(p).stem)