import cv2
from model_evaluator.interfaces.detection2D import BBox2D


def to_cv_pts(bbox: BBox2D):
    return (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2))


def draw_bboxes(image, gts, tps, fps):
    text_height = 15
    offset = 10
    thickness = 2

    scale = cv2.getFontScaleFromHeight(
        cv2.FONT_HERSHEY_SIMPLEX, text_height, thickness
    )

    for gt in gts:
        pt1, pt2 = to_cv_pts(gt.bbox)
        cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness)

        cv2.putText(
            image,
            str(gt.label.name),
            (pt1[0], pt1[1] - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 0, 0),
            thickness,
        )

    for tp in tps:
        pt1, pt2 = to_cv_pts(tp.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness)

        cv2.putText(
            image,
            str(tp.label.name),
            (pt1[0], pt1[1] - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 0),
            thickness,
        )

    for fp in fps:
        pt1, pt2 = to_cv_pts(fp.bbox)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), thickness)

        cv2.putText(
            image,
            str(fp.label.name),
            (pt1[0], pt1[1] - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 255),
            thickness,
        )


def write_png_img(filename, image):
    cv2.imwrite(f'{filename}.png', image)
