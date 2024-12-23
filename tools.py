import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_video(video_path):
    """Load the video from the specified path"""
    cap = cv2.VideoCapture(video_path)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_len, video_fps


def smooth_trajectory(trajectory, sigma=0.7):
    """Smooth the trajectory using a Gaussian filter"""
    if len(trajectory) < 2:
        return trajectory
    smoothed_trajectory = gaussian_filter1d(trajectory,
                                            sigma,
                                            axis=0,
                                            mode='nearest')
    return smoothed_trajectory


def add_trajectory_from_results(results, trajectory_list):
    """Add the position of the first ID to the trajectory list"""
    ids = results[0].boxes.id
    boxes = results[0].boxes.xywh.cpu()
    if ids is None:
        return trajectory_list
    else:
        box = boxes[0]
        x, y, w, h = box
        y = y + h / 2
        trajectory_list.append([x, y])
    return trajectory_list


def create_mask_from_results(results, frame):
    """Create a mask from the detected boxes"""
    mask = np.ones_like(frame) * 255
    boxes = results[0].boxes.xywh.cpu()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(mask,
                      (int(x - w / 2), int(y - h / 2)),
                      (int(x + w / 2), int(y + h / 2)),
                      0, -1)
    return mask


def calculate_homography_matrix(kp1, des1, kp2, des2):
    """Calculate the homography matrix using RANSAC"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def draw_trajectory(frame, trajectory):
    """Draw the trajectory on the frame"""
    color = (255, 0, 0)
    for t in reversed(range(1, len(trajectory))):
        # Gradually change the color
        color = [x + y for x, y in zip(color, (0, 5, 0))]
        cv2.line(frame,
                 (int(trajectory[t - 1][0]), int(trajectory[t - 1][1])),
                 (int(trajectory[t][0]), int(trajectory[t][1])),
                 color, 2)
    return frame
