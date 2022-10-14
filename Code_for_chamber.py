import RPi.GPIO as GPIO

import serial

from smbus2 import SMBus
from mlx90614 import MLX90614

import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import imutils
import cv2

origin = (0, 50)
color = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 2
thickness = 2

# empty display of black color
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for scanning the card
scan_card = cv2.putText(black, "Scan RFID Card", origin, font, scale, color, thickness, cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for after scanning the card
allowed = cv2.putText(black, "Door Open!!", origin, font, scale, color, thickness, cv2.LINE_AA)
allowed = cv2.putText(allowed, "You can now enter!!", (origin[0], origin[1] + 100), font, scale, color, thickness,
                      cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for hand sanitization
hand_ultrasonic = cv2.putText(black, "Place hand in front of", origin, font, scale, color, thickness, cv2.LINE_AA)
hand_ultrasonic = cv2.putText(hand_ultrasonic, "ultrasonic sensor", (origin[0], origin[1] + 100), font, scale, color,
                              thickness, cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for temperature sensor
temp = cv2.putText(black, "Place hand in front of", origin, font, scale, color, thickness, cv2.LINE_AA)
temp = cv2.putText(temp, "Temperature sensor", (origin[0], origin[1] + 100), font, scale, color, thickness, cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for normal body temperature
normal_temp = cv2.putText(black, "Normal body", origin, font, scale, color, thickness, cv2.LINE_AA)
normal_temp = cv2.putText(normal_temp, "temperature", (origin[0], origin[1] + 100), font, scale, color, thickness,
                          cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for high temperature
high_temp = cv2.putText(black, "High body temperature!!", origin, font, scale, color, thickness, cv2.LINE_AA)
high_temp = cv2.putText(high_temp, "Can not enter", (origin[0], origin[1] + 100), font, scale, color, thickness,
                        cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for sanitization
sanit = cv2.putText(black, "Sanitizing!!!", origin, font, scale, color, thickness, cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# display for mask detection by facing in the camera
face_camera = cv2.putText(black, "Please face into", origin, font, scale, color, thickness, cv2.LINE_AA)
face_camera = cv2.putText(face_camera, "the camera", (origin[0], origin[1] + 100), font, scale, color, thickness,
                          cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# dispaly for door open
# door_open = cv2.putText(black,"Door Open!!",origin,font,scale,color,thickness,cv2.LINE_AA)
# door_open = cv2.putText(door_open,"You can now enter",(origin[0],origin[1]+100),font,scale,color,thickness,cv2.LINE_AA)

# dispaly for door close
close_door = cv2.putText(black, "Door Close!!", origin, font, scale, color, thickness, cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

# dispaly for without mask no entry
# not_allowed = cv2.putText(black,"Not allowed to enter",origin,font,scale,color,thickness,cv2.LINE_AA)
# not_allowed = cv2.putText(not_allowed,"without wearing MASK!",(origin[0],origin[1]+100),font,scale,color,thickness,cv2.LINE_AA)

# display for tain mask from the conveyer belt
conveyer = cv2.putText(black, "Take your mask from", origin, font, scale, color, thickness, cv2.LINE_AA)
conveyer = cv2.putText(conveyer, "Conveyer Belt", (origin[0], origin[1] + 100), font, scale, color, thickness,
                       cv2.LINE_AA)
black = np.zeros((720, 720, 3), dtype=np.uint8)

conv_belt = 15
sanitizer = 32
pulse_start = 0
pulse_end = 0
TRIG = 37
ECHO = 36
sanitize_delay = 2
door_open = 31
door_close = 33

details = []

rfid_list = ["0B0023E7CA05", "0B0023F10ED7", "0B0023EA9153", "0B0023F769B6"]

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(conv_belt, GPIO.OUT)
GPIO.setup(sanitizer, GPIO.OUT)
GPIO.setup(door_open, GPIO.OUT)
GPIO.setup(door_close, GPIO.OUT)

GPIO.output(door_open, True)
GPIO.output(door_close, True)
GPIO.output(sanitizer, True)
GPIO.output(conv_belt, True)


def check_dist():
    pulse_start = 0
    pulse_end = 0
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance + 1.15, 2)
    # print(distance)
    return distance


def check_temp():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    # print ("Ambient Temperature :", sensor.get_ambient())
    temp = sensor.get_object_1()
    # print ("Object Temperature :", temp)
    bus.close()
    temp = int((9 / 5 * temp) + 32)
    return temp


def sanitize():
    time.sleep(2)
    GPIO.output(sanitizer, False)
    time.sleep(sanitize_delay)
    GPIO.output(sanitizer, True)


def start_conveyer():
    GPIO.output(conv_belt, False)
    time.sleep(5)
    GPIO.output(conv_belt, True)


def detect_and_predict_mask(frame, faceNet, maskNet):
    print("checking mask")
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))


if len(faces) > 0:
    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=32)

return (locs, preds)


def read_rfid():
    print("rfid tag")
    ser = serial.Serial(port='/dev/ttyS0', baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS)
    data = ser.read(12)
    ser.close()
    read = str(data)
    read = read.replace("'", "").replace('b', '')
    return read


while True:

    cv2.imshow("Text", hand_ultrasonic)
    cv2.waitKey(1000)

    distance = check_dist()

    if distance <= 8:
        # Scan RFID Card
        cv2.imshow("Text", scan_card)
        cv2.waitKey(1000)
        rfid = read_rfid()
        if rfid in rfid_list:  # if RFID matched then check temperature
            cv2.imshow("Text", temp)
            cv2.waitKey(3000)
            temperature = check_temp()

            if temperature >= 99.00:  # if temperature greater than threshold do nothing
                cv2.imshow("Text", high_temp)
                cv2.waitKey(2000)
                # No need to sanitize

            elif temperature < 99.00:  # if temperature in normal range check for mask
                cv2.imshow("Text", normal_temp)
                cv2.waitKey(2000)

                cv2.imshow("Text", face_camera)
                cv2.waitKey(3000)

                camera = PiCamera()
                camera.resolution = (480, 320)  # (640, 480)
                rawCapture = PiRGBArray(camera, size=(480, 320))  # (640, 480))
                time.sleep(0.3)

                camera.capture(rawCapture, format="bgr")
                frame = rawCapture.array
                cv2.imshow("Text", frame)
                cv2.waitKey(1000)
                cv2.imwrite("image.jpg", frame)
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):

                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    pred = max(mask, withoutMask) * 100

                    print(pred, "  ", label)

                    if label == "Mask":
                        cv2.imshow("Text", sanit)  # first sanitize and then enter
                        cv2.waitKey(3000)
                        sanitize()
                        welcome = cv2.putText(frame, "Welcome user", origin, font, 1, (0, 0, 255), thickness,
                                              cv2.LINE_AA)
                        cv2.imshow("Text", welcome)
                        cv2.waitKey(3000)
                        GPIO.output(door_open, False)
                        GPIO.output(door_close, True)
                        time.sleep(5)
                        GPIO.output(door_open, True)
                        GPIO.output(door_close, True)

                        cv2.imshow("Text", allowed)
                        cv2.waitKey(3000)

                        GPIO.output(door_open, True)
                        GPIO.output(door_close, False)
                        time.sleep(5)
                        GPIO.output(door_open, True)
                        GPIO.output(door_close, True)
                        cv2.imshow("Text", close_door)
                        cv2.waitKey(3000)

                        # here send email of rfid, time, and temperature
                        send_email(rfid, temperature)



                    else:
                        print("No Mask")
                        not_allowed = cv2.putText(frame, "Not allowed to enter", origin, font, 1, (0, 0, 255),
                                                  thickness, cv2.LINE_AA)
                        not_allowed = cv2.putText(not_allowed, "without wearing MASK!", (origin[0], origin[1] + 100),
                                                  font, 1, (0, 0, 255), thickness, cv2.LINE_AA)
                        cv2.imshow("Text", not_allowed)
                        cv2.waitKey(3000)
                        cv2.imshow("Text", conveyer)
                        cv2.waitKey(3000)
                        start_conveyer()

                camera.close()
                i = 1

        else:
            print("Card not authorized")

    else:
        cv2.imshow("Text", black)
        cv2.waitKey(1000)

"""except KeyboardInterrupt:
    GPIO.cleanup()
"""