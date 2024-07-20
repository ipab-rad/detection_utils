import json
from model_evaluator.bbox_generator.keyframe_interpolator import KeyframeInterpolator


def create_bboxes_from_keyframes_file(file_path:str):
    with open(f"keyframes/{file_path}") as f:
        d = json.load(f)

    all_boxes = []
    for ped_path in d:
        ki = KeyframeInterpolator(ped_path)
        all_boxes += ki.all_frames

    all_boxes.sort(key=lambda x: x["frame"])

    with open(f"scene_boxes/{file_path}", 'w', encoding='utf-8') as f:
        json.dump(all_boxes, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    create_bboxes_from_keyframes_file("10m_1_ped_0.json")