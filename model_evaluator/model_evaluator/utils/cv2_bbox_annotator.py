import numpy as np
import cv2 as cv

from model_evaluator.interfaces.detection2D import BBox2D

def to_cv_pts(bbox: BBox2D):
    return (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2))


def draw_bboxes(image, gts, tps, fps, text_height = 25, offset = 10, thickness = 2):
    scale = cv.getFontScaleFromHeight(
        cv.FONT_HERSHEY_SIMPLEX, text_height, thickness
    )

    for gt in gts:
        pt1, pt2 = to_cv_pts(gt.bbox)
        cv.rectangle(image, pt1, pt2, (255, 0, 0), thickness)

        cv.putText(
            image,
            str(gt.label.name),
            (pt1[0], pt1[1] - offset),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 0, 0),
            thickness,
        )

    for tp in tps:
        pt1, pt2 = to_cv_pts(tp.bbox)
        cv.rectangle(image, pt1, pt2, (0, 255, 0), thickness)

        cv.putText(
            image,
            str(tp.label.name),
            (pt1[0], pt1[1] - offset),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 0),
            thickness,
        )

    for fp in fps:
        pt1, pt2 = to_cv_pts(fp.bbox)
        cv.rectangle(image, pt1, pt2, (0, 0, 255), thickness)

        cv.putText(
            image,
            str(fp.label.name),
            (pt1[0], pt1[1] - offset),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 255),
            thickness,
        )

def draw_metrics(image, threshold, mean_ap, labels, num_gts_per_label, tps_per_label, fps_per_label, mrs_per_label, aps_per_label, text_height = 25, offset = 10, thickness = 2):
    scale = cv.getFontScaleFromHeight(
        cv.FONT_HERSHEY_SIMPLEX, text_height, thickness
    )

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv.putText(
        mask,
        f'@{threshold:.2f}',
        (0, text_height),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (255),
        thickness,
    )
    cv.putText(
        mask,
        f'mAP: {mean_ap:.2f}',
        (150, text_height),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (255),
        thickness,
    )

    for i, label in enumerate(labels):
        cv.putText(
            mask,
            f'{label.name}',
            (0, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )
        cv.putText(
            mask,
            f'{num_gts_per_label[i]:2}',
            (250, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )
        cv.putText(
            mask,
            f'TP: {tps_per_label[i]:2}',
            (350, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )
        cv.putText(
            mask,
            f'FP: {fps_per_label[i]:2}',
            (500, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )
        cv.putText(
            mask,
            f'MR: {mrs_per_label[i]:2}',
            (650, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )
        cv.putText(
            mask,
            f'AP: {aps_per_label[i]:.2f}',
            (800, (i + 2) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255),
            thickness,
        )

    cv.bitwise_not(image, image, mask)


def write_png_img(filename, image):
    cv.imwrite(f'{filename}.png', image)

def create_video_writer(file, size=(960, 640), fps=10):
    if not file.endswith('.mp4'):
            file += '.mp4'

    fourcc = cv.VideoWriter_fourcc(*'avc1')
    return cv.VideoWriter(
        file, fourcc, fps, size
    )