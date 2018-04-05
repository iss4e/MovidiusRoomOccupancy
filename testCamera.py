import picamera
import picamera.array

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        camera.resolution = (1280,720)
        camera.capture(output, 'bgr')
        print(output.array)
