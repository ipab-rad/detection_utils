def rosbags_run(connector):
    rosbags = match_rosbags_in_path(
        '/opt/ros_ws/rosbags/kings_buildings_data/'
    )

    rosbags.sort(key=lambda x: x.metadata.timestamp)

    print(rosbags)

    # rosbags = [KBRosbag('/opt/ros_ws/rosbags/kings_buildings_data/2024_07_12-11_05_14_5m_1_ped_bike_1')]


    for rosbag in rosbags:
        print(f'/opt/ros_ws/src/deps/external/detection_utils/kings_buildings_videos/{rosbag.metadata.name}')
        print(rosbag.metadata)

        matches = []

        video_writer = create_video_writer(f'/opt/ros_ws/src/deps/external/detection_utils/kings_buildings_videos/{rosbag.metadata.name}', (2272, 1088), 20)

        # sift = cv.SIFT_create()

        # prev_detections = []
        # prev_image = None

        # last_kp = None
        # last_des = None
        # last_roi = None

        # last_detections_kp_des_roi = []
        # last_image = None

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        # search_params = dict(checks=100) # or pass empty dictionary
        # flann = cv.FlannBasedMatcher(index_params,search_params)

        for frame, (image, _) in enumerate(rosbag.get_reader_2d().read_data()):
            detections = connector.run_inference(image)

            detections = [detection for detection in detections if detection.label in Label.VRU]

            

            # detections_kp_des_roi = []

            # for detection in detections:
            #     roi = image[detection.bbox.y1:detection.bbox.y2, detection.bbox.x1:detection.bbox.x2]
            #     kp, des = sift.detectAndCompute(roi, None)

            #     if kp is None or des is None:
            #         print(f'{detection.bbox}: no keypoints found')
            #         continue
            #     if len(kp) < 2:
            #         print(f'{detection.bbox}: less than 2 keypoints found')
            #         continue
            #     else:
            #         print(f'{detection.bbox}: {len(kp)}')

            #     detections_kp_des_roi.append((detection, kp, des, roi))

            # for j, (last_detection, last_kp, last_des, last_roi) in enumerate(last_detections_kp_des_roi):
            #     res = cv.matchTemplate(image,roi,cv.TM_SQDIFF_NORMED)
            #     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            #     x = image[min_loc[1]:min_loc[1] + last_detection.bbox.height(), min_loc[0]:min_loc[0] + last_detection.bbox.width()]

            #     cv.imwrite(f'{frame}_{j}_template.png', x)

            #     matched = False
            #     close = []

            #     for detection, kp, des, roi in detections_kp_des_roi:
            #         if is_close(detection.bbox, last_detection.bbox):
            #             close.append((detection, kp, des, roi))

            #             print(f'close bboxes {detection.bbox} ({len(des)}) {last_detection.bbox} ({len(last_des)})')

            #     for k, (detection, kp, des, roi) in enumerate(close):
            #         matches = flann.knnMatch(last_des,des,k=2)
            #         print(f'matches: {len(matches)}')

            #         if len(matches) == 0:
            #             continue

            #         matched = True

            #         matchesMask = [[0,0] for i in range(len(matches))]
            #         # ratio test as per Lowe's paper
            #         for i,(m,n) in enumerate(matches):
            #             if m.distance < 0.7*n.distance:
            #                 matchesMask[i]=[1,0]

            #         draw_params = dict(matchColor = (0,255,0),
            #             singlePointColor = (255,0,0),
            #             matchesMask = matchesMask,
            #             flags = cv.DrawMatchesFlags_DEFAULT)
                    
            #         out = cv.drawMatchesKnn(last_roi,last_kp,roi,kp,matches,None,**draw_params)
            #         cv.imwrite(f'{frame}_{j}_{k}.png', out)

            #     if not matched:
            #         close_bbox = near_area(last_detection.bbox)
            #         roi = image[close_bbox.y1:close_bbox.y2, close_bbox.x1:close_bbox.x2]
            #         kp, des = sift.detectAndCompute(roi, None)

            #         if kp is None or des is None:
            #             print(f'{close_bbox}: no keypoints found')
            #         if len(kp) < 2:
            #             print(f'{close_bbox}: less than 2 keypoints found')
            #         else:
            #             print(f'{close_bbox}: {len(kp)}')

            #             matches = flann.knnMatch(last_des,des,k=2)
            #             print(f'matches: {len(matches)}')

            #             if len(matches) > 0:
            #                 matched = True

            #                 matchesMask = [[0,0] for i in range(len(matches))]
            #                 # ratio test as per Lowe's paper
            #                 for i,(m,n) in enumerate(matches):
            #                     if m.distance < 0.7*n.distance:
            #                         matchesMask[i]=[1,0]

            #                 draw_params = dict(matchColor = (0,255,0),
            #                     singlePointColor = (255,0,0),
            #                     matchesMask = matchesMask,
            #                     flags = cv.DrawMatchesFlags_DEFAULT)
                            
            #                 out = cv.drawMatchesKnn(last_roi,last_kp,roi,kp,matches,None,**draw_params)
            #                 cv.imwrite(f'{frame}_{j}_X.png', out)


            
            # last_detections_kp_des_roi = detections_kp_des_roi
            match, dict = compare_expectations(detections, rosbag.get_expectations())
            matches.append(match)


            draw_frame_number(image, frame)

            if match:
                draw_bboxes(image, [], detections, [])
            else:
                draw_bboxes(image, [], [], detections)

            draw_matches(image, dict)

            video_writer.write(image)

        video_writer.release()

        print(f'Correct frames: {matches.count(True)} of {frame}')

        for match in matches:
            if match:
                print('X', end='')
            else:
                print('.', end='')

        print()


def is_close(bbox1: BBox2D, bbox2: BBox2D) -> bool:
    cx1, cy1 = bbox1.center()
    cx2, cy2 = bbox2.center()

    diff_x = abs(cx1 - cx2)
    diff_y = abs(cy1 - cy2)

    return diff_x < 20 and diff_y < 20

def near_area(bbox: BBox2D) -> BBox2D:
    return BBox2D.from_xyxy(bbox.x1 - 20, bbox.y1 - 20, bbox.x2 + 20, bbox.y2 + 20)




def lamr(all_detections: list[list[Detection2D]], all_gts: list[list[Detection2D]]):
    

    thresholds = np.linspace(0.0, 1.0, num=100)

    fppis = []
    mrs = []

    for threshold in thresholds:
        fppi, mr = calculate_fppi_mr(all_detections, all_gts, threshold, 0.5)

        fppis.append(fppi)
        mrs.append(mr)

    fppis = np.array(fppis)
    mrs = np.array(mrs)

    fppi_indices = np.logical_and(fppis >= 0.01, fppis <= 1)

    if not np.any(fppi_indices):
        return None

    log_fppis = np.log(fppis[fppi_indices])
    corr_mrs = mrs[fppi_indices]

    lamr = np.exp(np.mean(log_fppis) * np.mean(corr_mrs))
    return lamr

def filter_detections(detections: list[Detection2D], threshold: float):
    return [detection for detection in detections if detection.score >= threshold]

def calculate_fppi_mr(all_detections: list[list[Detection2D]], all_gts: list[list[Detection2D]], score_threshold: float, iou_threshold: float):
    all_false_positives = 0
    all_misses = 0
    num_gts = 0
    num_frames = len(all_detections)

    # For detections, gts for each image
    for detections, gts in zip(all_detections, all_gts):
        filtered_detections = filter_detections(detections, score_threshold)

        false_positives, misses = false_positives_misses(filtered_detections, gts, iou_threshold)

        all_false_positives += false_positives
        all_misses += misses
        num_gts += len(gts)

    fppi = all_false_positives / num_frames
    mr = all_misses / num_gts
    

    return fppi, mr

def false_positives_misses(detections: list[Detection2D], gts: list[Detection2D], iou_threshold: float):
    ious = calculate_ious(detections, gts)

    tps, fps = calculate_tps_fps(ious, iou_threshold)

    false_positives = fps.sum()
    misses = len(gts) - tps.sum()

    return false_positives, misses