import os
import os.path as osp
from tqdm import tqdm
import time
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

matplotlib.use('Agg')

existing_cmap = plt.get_cmap("Paired")
custom_colors = [(existing_cmap(p)[2], existing_cmap(p)[1], existing_cmap(p)[0]) for p in np.linspace(0, 1, 12)] * 7


def get_args():
    # Define user input arguments
    parser = argparse.ArgumentParser(prog='video_streaming_sandbox', description='Playing with video online streaming')
    parser.add_argument('--video_url', '--video_url', type=str, required=True, help='Url of online video streaming - currently adjusted to earthcam feeds')
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


def create_output_video_obj(cap):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 format
    return cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))


def yolo_inference_on_image(model, frame):
    cv2.imwrite('tmp.jpg', frame)
    results = model(['tmp.jpg'])
    return results.xyxy[0], results.names


def get_video_stream(driver, model):
    flag = False
    counter = 10
    out = None
    all_video_array = []
    for i in tqdm(range(counter)):
        if not flag:
            try:
                video_url, index_url = get_media_url(driver)
            except TypeError:
                continue
            cap = cv2.VideoCapture(video_url)
            print(video_url)
            if out is None:
                out = create_output_video_obj(cap)
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
                    break
        if not cap.isOpened():
            print("Error: Could not open video stream")
            flag = False
        else:
            current_video_array = []
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break  # Break the loop when we reach the end of the video
                current_video_array.append(frame)
            current_video_array = np.asarray(current_video_array)
            all_video_array.append(current_video_array)
            cap.release()
    for vid in tqdm(all_video_array):
        for frame in vid:
            predictions, label_dict = yolo_inference_on_image(model, frame)
            for p in predictions:
                p_np = p.numpy().astype(int)
                cv2.rectangle(frame, (p_np[2], p_np[3]), (p_np[0], p_np[1]),
                              color=np.array(custom_colors[p_np[5]]) * 255, thickness=2)
                cv2.putText(frame, f"{label_dict[p_np[5]]} - {int(p[4] * 100)}%", (p_np[0], p_np[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1, color=np.array(custom_colors[p_np[5]]) * 255, thickness=2)
            out.write(frame)
            # # Display the frame
            cv2.imshow("Video", frame)
            # Press 'q' to exit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    out.release()
    # Close the browser
    driver.quit()


if __name__ == '__main__':
    args = get_args()
    if osp.exists('tmp.jpg'):
        os.remove('tmp.jpg')
    if osp.exists('output.mp4'):
        os.remove('output.mp4')
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
