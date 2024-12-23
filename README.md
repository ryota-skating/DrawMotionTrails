# DrawMotionTrails
**DrawMotionTrails** is a tool designed to visualize the motion trails of a person in a video. It leverages **YOLO11** (with **BoT-SORT**) for object detection and tracking to extract the time-series coordinates of the main person in the footage. The motion trail is then rendered directly onto the video using **OpenCV** and feature point matching.

![sample gif](https://github.com/ryota-skating/DrawMotionTrails/blob/main/fig/sample.gif?raw=true)

## Features
- Accurate object detection and tracking using YOLO11 and BoT-SORT.
- Visualization of motion trails for a single main subject in the video.
- Feature point matching with OpenCV for seamless trail rendering.

## Installation

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Install `ffmpeg` (necessary for video processing):
   - **For macOS** (using Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - **For Ubuntu**:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - **For Windows**:
     Download and install from the [official ffmpeg site](https://ffmpeg.org/download.html).

## Usage

1. Place the video you want to process into the `inputs` directory.
2. Run the script to draw motion trails. For example, to process a video named `sample.mp4`:
   ```bash
   python main.py --video sample.mp4
   ```

## Notes

- The person whose motion trail is being visualized must be clearly visible throughout the video. If there are multiple people or if the target person is obscured at any point, detection and tracking may fail.
- Backgrounds that are overly uniform or simplistic can result in poor feature point matching, leading to inaccuracies in the rendered trail.

## License
This repository is licensed under the [Apache-2.0 License](LICENSE).

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

## Acknowledgments
This tool uses:
- [YOLO11](https://github.com/ultralytics/yolov5) for object detection.
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) for object tracking.
- [OpenCV](https://opencv.org/) for feature point matching and rendering.
- [ffmpeg](https://ffmpeg.org/) for video processing.
