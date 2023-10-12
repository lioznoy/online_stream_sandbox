import os
import os.path as osp
from tqdm import tqdm
import time
from datetime import datetime
import threading

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

import numpy as np
import cv2
import argparse
import torch

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

existing_cmap = plt.get_cmap("Paired")
custom_colors = [(existing_cmap(p)[2], existing_cmap(p)[1], existing_cmap(p)[0]) for p in np.linspace(0, 1, 12)] * 7
current_time = datetime.now()
formatted_time = current_time.strftime("%d-%m-%Y-%H-%M-%S")


def get_args():
    # Define user input arguments
    parser = argparse.ArgumentParser(prog='video_streaming_sandbox', description='Playing with video online streaming')
    parser.add_argument('--video_url', '--video_url', type=str, required=True,
                        help='Url of online video streaming - currently adjusted to earthcam feeds')
    parser.add_argument('--time_stop', '--time_stop', type=float, default=np.inf,
                        help='Time to stop running script [seconds] - default inf,'
                             ' will stop and save the video on KeyboardInterrupt at any time')
    return parser.parse_args()


def configure_chrome_webdriver(video_url):
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # Run in headless mode (optional)
    chrome_service = ChromeService()  # Specify the path to chromedriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    # Open the website
    driver.get(video_url)
    return driver


def get_media_url(driver):
    network_requests = driver.execute_script("return window.performance.getEntries()")
    for request in network_requests[::-1]:
        url = request["name"]
        if url is not None and 'media' in url:
            if url is None:
                return None, None
            index = int(url.split(".ts")[0].split("_")[-1])
            return url, index


def create_output_video_obj(cap, fps):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create a VideoWriter object to save the video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # Use video codec
    return cv2.VideoWriter(f"output_{formatted_time}.mp4", fourcc, fps, (frame_width, frame_height))


def yolo_inference_on_image(model, frame):
    cv2.imwrite('tmp.jpg', frame)
    results = model(['tmp.jpg'])
    return results.xyxy[0], results.names


def get_yolo_prediction_and_annotate_frame(model, frame):
    predictions, label_dict = yolo_inference_on_image(model, frame)
    annotated_frame = frame.copy()
    for p in predictions:
        p_np = p.numpy().astype(int)
        cv2.rectangle(annotated_frame, (p_np[2], p_np[3]), (p_np[0], p_np[1]),
                      color=np.array(custom_colors[p_np[5]]) * 255, thickness=2)
        cv2.putText(annotated_frame, f"{label_dict[p_np[5]]} - {int(p[4] * 100)}%", (p_np[0], p_np[1]),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=np.array(custom_colors[p_np[5]]) * 255, thickness=2)
    return annotated_frame


def get_video_stream(driver, model):
    flag = False
    out = None
    frame_calc_time = 0
    frame_count = 0
    try:
        while True:
            if not flag:
                try:
                    video_url, index_url = get_media_url(driver)
                except TypeError:
                    continue
                cap = cv2.VideoCapture(video_url)
                print(video_url)
                if out is None:
                    # get estimated fps
                    now = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    _ = get_yolo_prediction_and_annotate_frame(model, frame)
                    fps = 1 / (time.time() + 0.01 - now)
                    out = create_output_video_obj(cap, fps)
                flag = True
            else:
                video_url = '_'.join(video_url.split(".ts")[0].split("_")[:-1]) + "_" + str(
                    int(video_url.split(".ts")[0].split("_")[-1]) + 1) + ".ts" + video_url.split(".ts")[-1]
                cap = cv2.VideoCapture(video_url)
                print(video_url)
                now = time.time()
                while not cap.isOpened():
                    time.sleep(0.1)
                    cap = cv2.VideoCapture(video_url)
                    if time.time() - now > 2:
                        driver = configure_chrome_webdriver(args.video_url)
                        break
            if not cap.isOpened():
                print("Error: Could not open video stream")
                flag = False
            else:
                current_video_array = []
                while True:
                    # Read a frame from the video
                    skip_frames = int(frame_calc_time / (1 / int(cap.get(5))))
                    for i in range(skip_frames):
                        _, _ = cap.read()
                    now = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break  # Break the loop when we reach the end of the video
                    annotated_frame = get_yolo_prediction_and_annotate_frame(model, frame)
                    out.write(annotated_frame)
                    frame_count += 1
                    cv2.imshow("Video", annotated_frame)
                    # Press 'q' to exit the video
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    frame_calc_time = time.time() - now
                cap.release()
            if (frame_count / fps) > args.time_stop:
                break
        # close cv2 window
        cv2.destroyAllWindows()
        # release output video
        print('Releasing the video...')
        out.release()
        print(f'Video ready - {formatted_time}')
        # Close the browser
        print('Closing the browser...')
        now = time.time()
        driver.quit()
        print(f'Browser closed - {round(time.time() - now)}s')
        exit()
    except KeyboardInterrupt:
        # close cv2 window
        cv2.destroyAllWindows()
        # release output video
        print('Releasing the video...')
        out.release()
        print(f'Video ready - {formatted_time}')
        # Close the browser
        print('Closing the browser...')
        now = time.time()
        driver.quit()
        print(f'Browser closed - {round(time.time() - now)}s')
        exit()


if __name__ == '__main__':
    args = get_args()
    if osp.exists('tmp.jpg'):
        os.remove('tmp.jpg')
    print("Configuring chrome webdriver...")
    now = time.time()
    driver = configure_chrome_webdriver(args.video_url)
    print(f"Chrome webdriver configured - {round(time.time() - now)}s")
    print("Loading YOLO model...")
    now = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print(f"YOLO model Loaded - {round(time.time() - now)}s")
    print(f"Streaming video from {args.video_url}")
    get_video_stream(driver, model)
