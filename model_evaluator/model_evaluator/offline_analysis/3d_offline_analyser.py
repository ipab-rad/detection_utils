from model_evaluator.interfaces.labels import ALL_LABELS
from model_evaluator.utils.json_file_reader import read_json

import numpy as np

from model_evaluator.utils.metrics_calculator import calculate_ap


def analyse():
    file_path = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/results/kb"
    file_name = "10m_1_ped_0"

    per_class_results = read_json(f"{file_path}/{file_name}.json")

    for label in ALL_LABELS:
        label_results = per_class_results[label.name]

        gt_count = label_results["gt_count"]
        predictions = label_results["results"]
        pred_count = len(predictions)

        if gt_count == 0 or pred_count == 0:
            print(f"Cannot calculate AP for label {label.name} {gt_count=} {pred_count=}")

        predictions.sort(key=lambda x: x["score"])

        tp = np.array([int(x["true_positive"]) for x in predictions])
        fp = np.ones(tp.shape) - tp
        
        ap = calculate_ap(tp,fp,gt_count)

        print(f"AP={ap} for label {label.name}")


if __name__ == "__main__":
    analyse()