# pre-requisites:
#
# pip install picamera
# pip install tensorflow==1.13.1
# pip install opencv-python
# pip install keras==2.2.4
# pip install numpy==1.16.1
# pip install imageai --upgrade
#
# download this file into current directory:
# https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
#
# code adapted from:
# https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

from picamera import PiCamera
from time import sleep
from datetime import datetime
from imageai.Detection import ObjectDetection
import os


def continuous_capture(interval: float = 30, duration: float = 60, attempts: int = 960, repeat: bool = True) -> list[str]:
    """
    Captures and detects images continuously with the given interval.
    After the given number of attempts, it gives up.
    When it detects a person, it starts recording for the given duration.
    If repeat is set to True, the function runs again after recording.
    :param interval: interval (in seconds) during which the function is paused
    :param duration: duration (in seconds) of the recorded video
    :param attempts: number of attempts at detecting a person (default is roughly 8h of attempts)
    :param repeat: whether or not to try and record again
    :return: file names of the recorded videos (emtpy list if nothing was recorded)
    """
    detected = False
    while not detected and attempts > 0:
        filename = capture_image()
        detected = detect_people(filename) > 0
        sleep(interval)
        attempts = attempts - 1
    if detected:
        videoname = record_video(duration)
        if repeat:
            names = continuous_capture(interval=interval, duration=duration, attempts=attempts, repeat=repeat)
        return [videoname]+names
    else:
        return []


def record_video(duration: float = 60, preview: bool = False) -> str:
    """
    Records a video for the given duration and saves it as video_<current date & time>.jpg
    :param duration: duration (in seconds) of the recorded video
    :param preview: whether or not to preview the video
    :return: the file name
    """
    now = str(datetime.now())
    now = now.split('.')[0]
    now = '_' + now.split(' ')[0] + '_' + now.split(' ')[1]
    camera = PiCamera()
    if preview:
        camera.start_preview()
    filename = 'video' + now + '.jpg'
    camera.start_recording(filename)
    sleep(duration)
    camera.stop_recording()
    print(filename + ' was captured.')
    if preview:
        camera.stop_preview
    return filename


def capture_image(preview: bool = False) -> str:
    """
    Captures a single image, and saves it as image_<current date & time>.jpg
    :param preview: whether or not to preview the image
    :return: the file name
    """
    sleep(1)  # sleeps for 1sec to make sure no two images have the same name
    now = str(datetime.now())
    now = now.split('.')[0]
    now = '_' + now.split(' ')[0] + '_' + now.split(' ')[1]
    camera = PiCamera()
    if preview:
        camera.start_preview()
    filename = 'image' + now + '.jpg'
    camera.capture_image(filename)
    print(filename + ' was captured.')
    if preview:
        camera.stop_preview()
    return filename


def detect_people(filename: str) -> int:
    """
    Runs tensorflow object detection in the image given by filename.
    Returns the number of people detected.
    :param filename: file name of the image whose objects to detect
    :return: number of people detected
    """
    filename_detected = filename
    filename_detected.replace('image', 'detected_image')
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, filename),
                                                 output_image_path=os.path.join(execution_path, filename_detected))
    # for eachObject in detections:
    #    print(eachObject["name"], " : ", eachObject["percentage_probability"])
    checks = [detection["name"] == "person" for detection in detections]
    return sum(checks)
