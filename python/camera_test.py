from picamera import PiCamera
from time import sleep

camera = PiCamera();

camera.start_preview()
camera.start_recording('/home/pi/Desktop/video_test.h264')
sleep(60)
camera.stop_recording()
camera.stop_preview()
print('done recording');

#Random change to test git
#another random change