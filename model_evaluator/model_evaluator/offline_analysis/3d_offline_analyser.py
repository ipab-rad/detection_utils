from model_evaluator.interfaces.labels import ALL_LABELS, WAYMO_LABELS
from model_evaluator.utils.json_file_reader import read_json

import numpy as np
from pathlib import Path
import glob

from model_evaluator.utils.metrics_calculator import calculate_ap

def results_file_listing():
    results_files = [
        "5m_1_ped_0",           # 0
        "5m_1_ped_1",           # 1
        "5m_1_ped_bike_0",      # 2
        "5m_2_ped_0",           # 3
        "5m_2_ped_same_way_0",  # 4
        "10m_1_ped_0",          # 5
        "10m_1_ped_bike_2",     # 6
        "10m_2_ped_0",          # 7
        "10m_2_ped_same_way_1", # 8
        "20m_1_ped_1",          # 9
        "20m_1_ped_bike_0",     # 10
        "20m_2_ped_0",          # 11
        "20m_2_ped_same_way_1", # 12
        "40m_1_ped_0",          # 13
        "40m_1_ped_bike_1",     # 14
        "40m_2_ped_0",          # 15
        "40m_2_ped_same_way_0"  # 16
    ]

    return results_files


def results_files_groupings():
    range_groups = {"5m": [0, 1, 2, 3, 4], "10m": [5, 6, 7, 8], "20m": [9, 10, 11, 12], "40m": [13, 14, 15, 16]}

    vru_type_groups = {"1_ped": [0, 1, 5, 9, 13], "1_ped_bike": [2, 6, 10, 14], "2_ped": [3, 7, 11, 15],
                       "2_ped_same_way": [4, 8, 12, 16]}

    return range_groups, vru_type_groups


def append_results(all_results, results_per_class):
    for result_label in results_per_class:
        overall_dict_entry = all_results[result_label]
        result_dict_entry = results_per_class[result_label]

        overall_dict_entry["gt_count"] += result_dict_entry["gt_count"]
        overall_dict_entry["results"] += result_dict_entry["results"]


def analyse():
    # Waymo
    # file_path = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/results/waymo/has_ground"

    # labels_to_use = WAYMO_LABELS

    # files_to_combine = [Path(p).stem for p in glob.glob(f"{file_path}/*")]

    # KB
    file_path = "/opt/ros_ws/src/deps/external/detection_utils/model_evaluator/model_evaluator/results/kb/no_ground"

    results_files = results_file_listing()
    range_groups, vru_type_groups = results_files_groupings()

    chosen_range_names = []
    chosen_vru_type_names = ["1_ped_bike"]
    chosen_range_files = [rf for n in chosen_range_names for rf in range_groups[n]]
    chosen_vru_type_files = [vtf for n in chosen_vru_type_names for vtf in vru_type_groups[n]]

    labels_to_use = ALL_LABELS

    files_to_combine = [results_files[x] for x in range(len(results_files)) if x in chosen_range_files
                        or x in chosen_vru_type_files]

    print(f"Grouping by range {chosen_range_names}")
    print(f"Grouping by vru type {chosen_vru_type_names}")

    print(files_to_combine)

    all_results = {label.name: {"gt_count": 0, "results": []} for label in labels_to_use}

    for file_name in files_to_combine:
        per_class_results = read_json(f"{file_path}/{file_name}.json")
        append_results(all_results, per_class_results)

    for label in labels_to_use:
        label_results = all_results[label.name]

        gt_count = label_results["gt_count"]
        predictions = label_results["results"]
        pred_count = len(predictions)

        if gt_count == 0 or pred_count == 0:
            print(f"Cannot calculate AP for label {label.name} {gt_count=} {pred_count=}")
            continue

        predictions.sort(key=lambda x: x["score"])

        tp = np.array([int(x["true_positive"]) for x in predictions])
        fp = np.ones(tp.shape) - tp

        ap = calculate_ap(tp, fp, gt_count)

        print(f"AP={ap} for label {label.name}")


if __name__ == "__main__":
    analyse()