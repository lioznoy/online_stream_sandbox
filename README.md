# online_stream_sandbox
playing with CV and DL on online video streaming.

Getting video stream from earthcam.com link and use YOLOv5 (cpu) to get predictions of different objects in the video and present them.

Requirements - chromedriver, python packages: selenium, torch, ultralytics, opencv, numpy, matplotlib.

How-To - enter video stream URL from earthcam and play. output.mp4 video will be created with the annotations.
example:
main.py --video_url https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=bourbonstreet --time_stop 30 <time in seconds to record video, defualt=inf>

