import math
from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt


from model_evaluator.interfaces.detection2D import BBox2D, Detection2D, DifficultyLevel
from model_evaluator.interfaces.labels import Label


class MetricsCalculator:
    frames: list[tuple[list[Detection2D], list[Detection2D]]]
    labels: list[Label]

    def __init__(self, labels: list[Label] = [Label.PEDESTRIAN, Label.BICYCLE]):
        self.frames = []

        self.labels = labels

    @staticmethod
    def filter_score(detections: list[Detection2D], threshold: float):
        return [i for i, detection in enumerate(detections) if detection.score >= threshold]

    @staticmethod
    def filter_label(detections: list[Detection2D], label: Label):
        return [i for i, detection in enumerate(detections) if detection.label in label]

    @staticmethod
    def calculate_ious(
        detections: list[Detection2D], ground_truths: list[Detection2D]
    ) -> np.ndarray:
        detections = detections.copy()
        detections.sort(key=lambda x: x.score, reverse=True)

        detection_bboxes = [detection.bbox for detection in detections]
        ground_truth_bboxes = [ground_truth.bbox for ground_truth in ground_truths]

        ious = np.empty((len(detection_bboxes), len(ground_truth_bboxes)))

        for i, detection_bbox in enumerate(detection_bboxes):
            for j, ground_truth_bbox in enumerate(ground_truth_bboxes):
                ious[i, j] = detection_bbox.iou(ground_truth_bbox)

        return ious

    @staticmethod
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


    def log_average_miss_rate(self, plot_file=None) -> dict[Label, float]:
        lamr_dict = {}

        fig = None
        if plot_file:
            fig, axs = plt.subplots(3, 1, figsize=(15,15))

        thresholds = np.linspace(0.0, 1.0, num=100, endpoint=False)

        false_positives_sum = np.zeros_like(thresholds)
        false_negatives_sum = np.zeros_like(thresholds)

        num_all_gts = 0

        # fppis = np.empty((len(self.frames), thresholds.shape[0]))
        # mrs = np.empty((len(self.frames), thresholds.shape[0]))

        for label in self.labels:
            for frame_idx, (detections, gts) in enumerate(self.frames):
                detections_indices = self.filter_label(detections, label)
                gts_indices = self.filter_label(gts, label)

                detections = [detections[i] for i in detections_indices]
                gts = [gts[i] for i in gts_indices]

                ious = self.calculate_ious(detections, gts)

                num_all_gts += len(gts)

                for i, threshold in enumerate(thresholds):
                    filtered_indices = self.filter_score(detections, threshold)

                    filtered_ious = ious[filtered_indices, :]

                    correct = self.correct_matches(filtered_ious, 0.5)

                    true_positives = correct.sum()
                    false_positives = (~correct).sum()
                    false_negatives = len(gts) - true_positives

                    false_positives_sum[i] += false_positives
                    false_negatives_sum[i] += false_negatives

            #         fppi = false_positives
            #         fppis[frame_idx, i] = fppi

            #         if len(gts) == 0:
            #             mrs[frame_idx, i] = math.nan
            #             continue

            #         mr = false_negatives / len(gts)

            #         mrs[frame_idx, i] = mr

            # mean_fppis = np.nanmean(fppis, axis=0)
            # mean_mrs = np.nanmean(mrs, axis=0)

            fppis = false_positives_sum / len(self.frames)

            if num_all_gts == 0:
                mrs = np.full_like(false_negatives_sum, math.nan)
            else:
                mrs = false_negatives_sum / num_all_gts
            
            if fig:
                axs[1].plot(thresholds, fppis, label=f'{label} FPPI')
                axs[2].plot(thresholds, mrs, label=f'{label} MR')

            # nums_true_positives = np.zeros_like(thresholds)
            # nums_false_positives = np.zeros_like(thresholds)
            # nums_false_negatives = np.zeros_like(thresholds)

            # num_all_gts = 0

            # for detections, gts, ious in self.frames:
            #     detections_indices = self.filter_label(detections, label)
            #     gts_indices = self.filter_label(gts, label)

            #     detections = [detections[i] for i in detections_indices]
            #     gts = [gts[i] for i in gts_indices]

            #     ious = ious[detections_indices, :]
            #     ious = ious[:, gts_indices]

            #     num_all_gts += len(gts)

            #     for i, threshold in enumerate(thresholds):
            #         filtered_indices = self.filter_score(detections, threshold)

            #         filtered_ious = ious[filtered_indices, :]

            #         correct = self.correct_matches(filtered_ious, 0.5)

            #         true_positives = correct.sum()
            #         false_positives = (~correct).sum()
            #         false_negatives = len(gts) - true_positives

            #         nums_true_positives[i] += true_positives
            #         nums_false_positives[i] += false_positives
            #         nums_false_negatives[i] += false_negatives

            # if fig:
            #     axs[1].hlines(num_all_gts, thresholds[0], thresholds[-1], label=f'{label} GTs')
            #     axs[1].plot(thresholds, nums_true_positives, label=f'{label} TP')
            #     axs[1].plot(thresholds, nums_false_positives, label=f'{label} FP')
            #     axs[1].plot(thresholds, nums_false_negatives, label=f'{label} FN')

            # if num_all_gts == 0:
            #     # No ground truth for this label in all frames, use NaN
            #     lamr_dict[label] = math.nan

            #     continue

            # false_positives_per_image = nums_false_positives / len(self.frames)
            # miss_rates = nums_false_negatives / num_all_gts

            false_positives_per_image = fppis
            miss_rates = mrs
            
            if fig:
                axs[0].loglog(false_positives_per_image, miss_rates, label=f'{label} LAMR')
            
            ts = np.logspace(-2, 0, num=9)
            sum = 0

            for i, t in enumerate(ts):
                print(f'fppi threshold = {t} -> [{np.argmax(false_positives_per_image <= t)}] score threshold = {thresholds[np.argmax(false_positives_per_image <= t)]}  mr = {miss_rates[np.argmax(false_positives_per_image <= t)]}')
                sum += np.log(miss_rates[np.argmax(false_positives_per_image <= t)])

                if fig:
                    axs[0].vlines(t, 0, 1, color=f'C{i}', label=f'{i}')
                    axs[2].hlines(thresholds[[np.argmax(false_positives_per_image <= t)]], 0, 1, color=f'C{i}')

            lamr_dict[label] = np.exp(sum / 9)

        lamr = np.nanmean(list(lamr_dict.values()))

        if fig:
            axs[0].set_xlabel('FPPI')
            axs[0].set_ylabel('MR')
            axs[0].legend(self.labels)
            axs[1].set_yscale('log')
            axs[1].set_xlabel('Threshold')
            axs[1].set_ylabel('FPPI')
            axs[1].legend()
            axs[2].set_xlabel('Threshold')
            axs[2].set_ylabel('MR')
            axs[2].legend()

            fig.suptitle(f'LAMR = {lamr:.4f}')

            fig.savefig(plot_file)

        return lamr, lamr_dict

    def mean_average_precision(self, threshold: float) -> tuple[float, dict[Label, float]]:
        ap_dict = {}

        for label in self.labels:
            aps = []

            for detections, gts in self.frames:
                detections_indices = self.filter_label(detections, label)
                gts_indices = self.filter_label(gts, label)

                detections = [detections[i] for i in detections_indices]
                gts = [gts[i] for i in gts_indices]

                if len(gts) == 0:
                    aps.append(math.nan)
                    continue

                ious = self.calculate_ious(detections, gts)

                num_gts = len(gts)

                correct = self.correct_matches(ious, threshold)

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

            ap_dict[label] = np.nanmean(aps)

        return np.nanmean(list(ap_dict.values())), ap_dict

    def add_frame(self, detections: list[Detection2D], ground_truths: list[Detection2D]):
        self.frames.append((detections, ground_truths))

    # def frame_metrics(self, detections: list[Detection2D], ground_truths: list[Detection2D]):
    #     num_ground_truths = len(ground_truths)
    #     num_detections = len(detections)

    #     ious = calculate_ious(detections, ground_truths)

    #     correct = correct_matches(ious, self.iou_threshold)

    #     num_correct = correct.sum()

    #     return num_correct, num_detections, num_ground_truths

class KBMetricsCalculator(MetricsCalculator):
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

    def error(self):
        error_dict = {}

        for label in self.labels:
            incorrect = np.zeros(3, dtype=np.int32)

            for detections, expectations in self.frames:
                detections = [detection for detection in detections if detection.label in label]

                correct = self.compare_expectations(detections, expectations)

                if not correct:
                    difficulty_levels = set([expectation.difficulty_level for expectation in expectations])

                    if DifficultyLevel.HARD in difficulty_levels:
                        incorrect[2] += 1
                    elif DifficultyLevel.EASY in difficulty_levels:
                        incorrect[1] += 1
                    else:
                        incorrect[0] += 1

            error_dict[label] = incorrect / len(self.frames)

        return error_dict