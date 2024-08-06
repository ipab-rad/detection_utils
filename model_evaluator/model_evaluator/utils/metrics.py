import numpy as np
import scipy

from model_evaluator.interfaces.detection2D import BBox2D, Detection2D, DifficultyLevel
from model_evaluator.interfaces.labels import Label


def calculate_ious_2d(
    pred_bboxes: list[BBox2D], gt_bboxes: list[BBox2D]
) -> np.ndarray:
    ious = np.empty((len(pred_bboxes), len(gt_bboxes)))

    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            ious[i, j] = pred_bbox.iou(gt_bbox)

    return ious

def calculate_ious(
    detections: list[Detection2D], gts: list[Detection2D]
) -> np.ndarray:
    detections = detections.copy()
    detections.sort(key=lambda x: x.score, reverse=True)

    ious = calculate_ious_2d(
        [detection.bbox for detection in detections],
        [gt.bbox for gt in gts],
    )

    return ious

def filter_detections(detections: list[Detection2D], threshold: float):
    return [i for i, detection in enumerate(detections) if detection.score >= threshold]

def filter_detections_labels(detections: list[Detection2D], label: Label):
    return [detection for detection in detections if detection.label is label]

def correct_matches(ious: np.ndarray, threshold: float):
    num_detections = ious.shape[0]
    num_gts = ious.shape[1]

    correct = np.zeros(num_detections, dtype=np.bool8)

    for i in range(num_gts):
        max_iou = 0.0
        max_idx = -1

        for j in range(num_detections):
            iou = ious[j, i]

            if iou > max_iou and not correct[j]:
                max_iou = iou
                max_idx = j

        if max_iou >= threshold:
            correct[max_idx] = True

    return correct

def log_average_miss_rate(all_detections_gts: list[tuple[list[Detection2D], list[Detection2D]]]) -> float:
    thresholds = np.linspace(0.0, 1.0, num=100)

    nums_false_positives = np.zeros_like(thresholds)
    nums_false_negatives = np.zeros_like(thresholds)

    num_gts = 0

    for detections, gts in all_detections_gts:
        detections = filter_detections_labels(detections, Label.PEDESTRIAN)
        gts = filter_detections_labels(gts, Label.PEDESTRIAN)

        num_gts += len(gts)

        ious = calculate_ious(detections, gts)

        for i, threshold in enumerate(thresholds):
            filtered_indices = filter_detections(detections, threshold)

            filtered_ious = ious[filtered_indices, :]

            correct = correct_matches(filtered_ious, 0.5)

            true_positives = correct.sum()
            false_positives = (~correct).sum()
            false_negatives = len(gts) - true_positives

            nums_false_positives[i] += false_positives
            nums_false_negatives[i] += false_negatives

    false_positives_per_image = nums_false_positives / len(all_detections_gts)
    miss_rates = nums_false_negatives / num_gts

    np.finfo(np.float64).eps

    ts = np.logspace(-2, 0, num=9)
    sum = 0

    for t in ts:
        sum += np.log(miss_rates[np.argmax(false_positives_per_image <= t)])

    lamr = np.exp(sum / 9)

    return lamr


class Metrics:
    frames: list[tuple[list[Detection2D], list[Detection2D], np.ndarray]]

    def __init__(self, iou_threshold: float):
        self.frames = []
        self.iou_threshold = iou_threshold

    @staticmethod
    def filter_score(detections: list[Detection2D], threshold: float):
        return [i for i, detection in enumerate(detections) if detection.score >= threshold]

    @staticmethod
    def filter_label(detections: list[Detection2D], label: Label):
        return [i for i, detection in enumerate(detections) if detection.label is label]


    def log_average_miss_rate(self) -> float:
        thresholds = np.linspace(0.0, 1.0, num=100)

        nums_false_positives = np.zeros_like(thresholds)
        nums_false_negatives = np.zeros_like(thresholds)

        num_all_gts = 0

        for detections, gts, ious in self.frames:
            detections_indices = self.filter_label(detections, Label.PEDESTRIAN)
            gts_indices = self.filter_label(gts, Label.PEDESTRIAN)

            detections = [detections[i] for i in detections_indices]
            gts = [gts[i] for i in gts_indices]

            ious = ious[detections_indices, :]
            ious = ious[:, gts_indices]

            num_all_gts += len(gts)

            for i, threshold in enumerate(thresholds):
                filtered_indices = self.filter_score(detections, threshold)

                filtered_ious = ious[filtered_indices, :]

                correct = correct_matches(filtered_ious, 0.5)

                true_positives = correct.sum()
                false_positives = (~correct).sum()
                false_negatives = len(gts) - true_positives

                nums_false_positives[i] += false_positives
                nums_false_negatives[i] += false_negatives

        false_positives_per_image = nums_false_positives / len(self.frames)
        miss_rates = nums_false_negatives / num_all_gts

        ts = np.logspace(-2, 0, num=9)
        sum = 0

        for t in ts:
            sum += np.log(miss_rates[np.argmax(false_positives_per_image <= t)])

        return np.exp(sum / 9)

    def mean_average_precision(self, threshold: float) -> float:
        aps = []

        for detections, gts, ious in self.frames:
            detections_indices = self.filter_label(detections, Label.PEDESTRIAN)
            gts_indices = self.filter_label(gts, Label.PEDESTRIAN)

            detections = [detections[i] for i in detections_indices]
            gts = [gts[i] for i in gts_indices]

            if len(gts) == 0:
                # Use NaN
                pass

            ious = ious[detections_indices, :]
            ious = ious[:, gts_indices]

            num_gts = len(gts)

            correct = correct_matches(ious, threshold)

            true_positives_cumsum = np.cumsum(correct)
            false_positives_cumsum = np.cumsum(~correct)

            precisions = true_positives_cumsum / (true_positives_cumsum + false_positives_cumsum)
            recalls = true_positives_cumsum / num_gts

            precisions = np.concatenate(([0], precisions, [0]))
            recalls = np.concatenate(([0], recalls, [1]))

            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])

            indices = np.where(recalls[1:] != recalls[:-1])[0]

            ap = np.sum(
                (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
            )

            aps.append(ap)

        return np.nanmean(aps)

    def add_frame(self, detections: list[Detection2D], ground_truths: list[Detection2D]):
        ious = calculate_ious(detections, ground_truths)

        # labels = [Label.PEDESTRIAN, Label.BICYCLE]
        # label_map = {}

        # for label in labels:
        #     label_map[label] = ([], [])


        # for detection in detections:
        #     index_lists = label_map.get(detection.label)

        #     index_lists[0].append(detection)

        # for ground_truth in ground_truths:
        #     index_lists = label_map.get(ground_truth.label)

        #     index_lists[1].append(ground_truth)

        self.frames.append((detections, ground_truths, ious))

    def frame_metrics(self, detections: list[Detection2D], ground_truths: list[Detection2D]):
        num_ground_truths = len(ground_truths)
        num_detections = len(detections)

        ious = calculate_ious(detections, ground_truths)

        correct = correct_matches(ious, self.iou_threshold)

        num_correct = correct.sum()

        return num_correct, num_detections, num_ground_truths

class KBMetrics(Metrics):
    def compare_expectations(self, 
        detections: list[Detection2D], expectations: list[Detection2D]
    ):
        expectations = expectations.copy()

        for detection in detections:
            matched = False

            for expectation in expectations:
                if detection.label in expectation.label:
                    expectations.remove(expectation)
                    matched = True

                    break

            if not matched:
                return False

        return len(expectations) == 0

    def accuracy(self):
        incorrect = np.zeros(3, dtype=np.int32)

        for detections, expectations, _ in self.frames:
            detections = [detection for detection in detections if detection.label in Label.VRU]

            correct = self.compare_expectations(detections, expectations)

            if not correct:
                difficulty_levels = set([expectation.difficulty_level for expectation in expectations])

                if DifficultyLevel.LEVEL_2 in difficulty_levels:
                    incorrect[2] += 1
                elif DifficultyLevel.LEVEL_1 in difficulty_levels:
                    incorrect[1] += 1
                else:
                    incorrect[0] += 1

        return incorrect / len(self.frames)