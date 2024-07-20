import cv2
from model_evaluator.interfaces.detection2D import BBox2D, Label


def to_cv_pts(bbox: BBox2D):
    return (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2))


def draw_bboxes(image, gts, detections, included_labels=Label.ALL):
    for gt in gts:
        if gt.label not in included_labels:
            continue

        pt1, pt2 = to_cv_pts(gt.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(
            image,
            str(gt.label),
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    for detection in detections:
        if detection.label not in included_labels:
            continue

        pt1, pt2 = to_cv_pts(detection.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)

        cv2.putText(
            image,
            str(detection.label),
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )


def write_png_img(filename, image):
    cv2.imwrite(f'{filename}.png', image)
