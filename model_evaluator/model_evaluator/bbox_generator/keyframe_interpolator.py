import numpy as np

class KeyframeInterpolator:
    def __init__(self, keyframes:list[dict]):
        self.keyframes = keyframes

        self.all_frames = []

        self.extract_all_frames()

    def extract_all_frames(self):
        keyframe_count = len(self.keyframes)

        for keyframe_counter in range(keyframe_count-1):
            keyframe = self.keyframes[keyframe_counter]
            next_keyframe = self.keyframes[keyframe_counter+1]
            frames_elapsed = next_keyframe["frame"] - keyframe["frame"]

            center_start = np.array(keyframe["center"])
            center_end = np.array(next_keyframe["center"])
            center_diff = center_end - center_start
            print(center_diff)

            self.all_frames.append(keyframe)
            for i in range(1, frames_elapsed):
                frame_dict = {"frame":i+keyframe["frame"]}
                increment = i/frames_elapsed
                frame_dict["center"] = np.round(center_start + increment * center_diff,3).tolist()
                self.all_frames.append(frame_dict)
        self.all_frames.append(self.keyframes[-1])

        print(self.all_frames)
