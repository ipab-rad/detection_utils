import numpy as np
import cv2 as cv

from model_evaluator.interfaces.detection2D import BBox2D, Detection2D


DEFAULT_TEXT_HEIGHT = 25
DEFAULT_TEXT_MARGIN = 10
DEFAULT_TEXT_THICKNESS = 2
DEFAULT_BBOX_THICKNESS = 2


def draw_frame_number(image: np.ndarray, frame_number: int, colour: tuple[int, int, int] = (0, 255, 0), height: float = DEFAULT_TEXT_HEIGHT, margin: float = DEFAULT_TEXT_MARGIN, thickness: float = DEFAULT_TEXT_THICKNESS):
    scale = cv.getFontScaleFromHeight(
        cv.FONT_HERSHEY_SIMPLEX, height, thickness
    )

    ((width, _), _) = cv.getTextSize(str(frame_number), cv.FONT_HERSHEY_SIMPLEX, scale, thickness)

    cv.putText(
        image,
        str(frame_number),
        (image.shape[1] - width, height + margin),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
    )

def draw_bboxes(image: np.ndarray, detections: list[Detection2D], colour: tuple[int, int, int] = (0, 255, 0), text_height: float = DEFAULT_TEXT_HEIGHT, text_margin: float = DEFAULT_TEXT_MARGIN, text_thickness: float = DEFAULT_TEXT_THICKNESS, bbox_thickness: float = DEFAULT_BBOX_THICKNESS):
    scale = cv.getFontScaleFromHeight(
        cv.FONT_HERSHEY_SIMPLEX, text_height, text_thickness
    )

    for detection in detections:
        left = round(detection.bbox.x1)
        top = round(detection.bbox.y1)
        right = round(detection.bbox.x2)
        bottom = round(detection.bbox.y2)

        cv.rectangle(image, (left, top), (right, bottom), colour, bbox_thickness)

        cv.putText(
            image,
            str(detection.label.name),
            (left, top - text_margin),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            text_thickness,
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

def draw_matches(image, matches_dict, text_height = 25, offset = 10, thickness = 2):
    scale = cv.getFontScaleFromHeight(
        cv.FONT_HERSHEY_SIMPLEX, text_height, thickness
    )

    for i, label in enumerate(matches_dict):
        if matches_dict[label][0] == matches_dict[label][1]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv.putText(
            image,
            f'{label.name}',
            (0, (i + 1) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
        cv.putText(
            image,
            f'{matches_dict[label][1]}/{matches_dict[label][0]}',
            (250, (i + 1) * (text_height + offset)),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

def write_png_img(filename, image):
    cv.imwrite(f'{filename}.png', image)

def create_video_writer(file, size=(960, 640), fps=10):
    if not file.endswith('.mp4'):
            file += '.mp4'

    fourcc = cv.VideoWriter_fourcc(*'avc1')
    return cv.VideoWriter(
        file, fourcc, fps, size
    )