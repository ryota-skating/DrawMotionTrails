import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from tools import load_video, smooth_trajectory
from tools import add_trajectory_from_results, create_mask_from_results
from tools import calculate_homography_matrix, draw_trajectory


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',
                        type=str,
                        default='sample.mp4',
                        help='filename of the input video')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load video
    input_dir = "./inputs"
    input_video_path = os.path.join(input_dir, args.video)
    cap, video_len, video_fps = load_video(input_video_path)

    # Create tmp directory
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Load detection model
    model = YOLO("yolo11x.pt")

    # Load AKAZE feature detector
    akz = cv2.AKAZE_create()

    # Process each frame
    print("Processing frames...")
    trajectory_list = []
    kp1, des1 = None, None
    for i in tqdm(range(video_len)):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO11 tracking
        results = model.track(frame,
                              classes=[0],
                              conf=0.3,
                              persist=True,
                              verbose=False)

        # Update the trajectory list
        trajectory_list = add_trajectory_from_results(results, trajectory_list)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask
        mask = create_mask_from_results(results, gray)

        # Detect and compute features in the current frame
        kp, des = akz.detectAndCompute(gray, mask)
        
        if kp1 is None or des1 is None:
            kp1, des1 = kp, des
            continue
        else:
            kp2, des2 = kp, des

        # Calculate the homography matrix
        H = calculate_homography_matrix(kp1, des1, kp2, des2)

        # Transform the trajectory points using the homography matrix
        trajectory_array = np.array(trajectory_list).reshape(-1, 1, 2)
        transformed_trajectory = cv2.perspectiveTransform(trajectory_array, H)
        # Smooth the transformed trajectory
        transformed_trajectory = smooth_trajectory(transformed_trajectory.reshape(-1, 2))

        # Draw the trajectory on the frame
        overlay = frame.copy()
        overlay = draw_trajectory(overlay, transformed_trajectory)

        # Update the keypoint, descriptor and trajectory list
        kp1, des1 = kp2, des2
        trajectory_list = transformed_trajectory.tolist()
        if len(trajectory_list) > 50:
            trajectory_list.pop(0)

        # Save the frame with the trajectory
        alpha = 0.7  # Transparency factor
        temp_img_path = os.path.join(tmp_dir, f'{i:04d}.jpg')
        frame_overlayed = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imwrite(temp_img_path, frame_overlayed)

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Create a video from the frames
    tmp_video_path = os.path.join(tmp_dir, 'tmp.mp4')
    os.system(f'ffmpeg -y -framerate {video_fps} -i {tmp_dir}/%04d.jpg '
              f'-c:v libx264 -pix_fmt yuv420p {tmp_video_path}')

    # Add audio to the video
    output_dir = "./outputs"
    output_video_path = os.path.join(output_dir, args.video)
    os.system(f'ffmpeg -y -i {tmp_video_path} -i {input_video_path} -c copy '
              f'-map 0:v:0 -map 1:a:0 -shortest {output_video_path}')

    # delete temp directory
    os.system(f'rm -rf {tmp_dir}')
    print(f"Output video saved at {output_video_path}")


if __name__ == '__main__':
    main()