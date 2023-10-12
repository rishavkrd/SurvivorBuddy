#!/usr/bin/env python3
from __future__ import annotations
from abc import ABC, abstractmethod
from random import randrange
from typing import List

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage, Image
from moveit_msgs.msg import DisplayTrajectory
from threading import Thread
from geometry_msgs.msg import TwistStamped


# Python 2/3 compatibility imports
import sys
from globals import *
import globals
import copy
import random
import math
import rospy
import time
import geometry_msgs.msg
import moveit_commander
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading
import signal
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


class Subject(ABC):
    @abstractmethod
    def addObserver(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def removeObserver(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def notify(self) -> None:
        pass

    @abstractmethod
    def notify(self) -> None:
        pass


class CameraPerception(Subject):
    _state: CompressedImage = None
    _observers: List[Observer] = []

    def addObserver(self, observer: Observer) -> None:
        self._observers.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def setChanged(self, data) -> None:
        self._state = data
        self.notify()


class AudioPerception(Subject):
    _state: Float32MultiArray = None
    _observers: List[Observer] = []

    def addObserver(self, observer: Observer) -> None:
        # print("Subject: Attached an observer.")
        self._observers.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def setChanged(self, audioData) -> None:
        self._state = audioData
        self.notify()


class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject) -> None:
        pass


Counter_1 = 0
Counter_2 = 0
timer = -1
face_detected = True
clap_detected = False
current_frame = None

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model("./mp_hand_gesture")

# Load class names
f = open("./gesture.names", "r")
classNames = f.read().split("\n")
f.close()
# print(classNames)


class HandGestureCartographer(Observer, Subject):
    _state: Float32MultiArray = None
    _observers: List[Observer] = []
    previous_image = None
    new_image = None

    def __init__(self):
        self.message = String()
        self.rate = rospy.Rate(1)

        x = threading.Thread(target=self.handRecognition, args=())
        x.start()

    def addObserver(self, observer: Observer) -> None:
        print("Subject: Attached an observer.")
        self._observers.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def setChanged(self, classID) -> None:
        self._state = classID
        self.notify()

    def update(self, subject: Subject) -> None:
        self.new_image = subject._state

    def handRecognition(self):
        global timer, timer1, Counter_1, Counter_2, hands
        global current_frame, VideoData
        global mp_drawing, mp_drawing_styles, mp_hands
        cv2.namedWindow("MediaPipe Hands", 2)

        while not killer.kill_now:
            if self.previous_image != self.new_image:
                VideoData = self.new_image
                self.previous_image = self.new_image
                np_arr = np.frombuffer(VideoData.data, np.uint8)
                cv_frame = cv2.imdecode(np_arr, 1)
                image = cv2.resize(cv_frame, (120, 120))
                x, y, c = image.shape

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # print('inside video')

                # print(result)

                className = ""

                # post process the result
                if result.multi_hand_landmarks:
                    print("..............Hand recognition called")
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            # print(id, lm)
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)

                            landmarks.append([lmx, lmy])
                        # timer=int(time.time())
                        # Drawing landmarks on frames
                        mpDraw.draw_landmarks(image, handslms, mpHands.HAND_CONNECTIONS)

                        # Predict gesture  13n
                        prediction = model.predict([landmarks])
                        # print(prediction)
                        classID = np.argmax(prediction)

                        # className = classNames[classID]
                        # print(className)
                    self.setChanged(classID)
                cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
                cv2.waitKey(1)
            time.sleep(0.1)


class YoutubeBehavior(Observer):
    gestureID = 0
    className = ""

    def __init__(self):
        print("Youtube Behavior ", threading.current_thread())

    def update(self, subject: Subject) -> None:
        self.gestureID = subject._state
        self.youtube_output()

    def youtube_output(self):
        gestureID = self.gestureID
        if gestureID == 1:
            print("Play (Victiory Peace)")
            self.message = "PLAY"
            displayController.youtubeAction(self.message)

        elif gestureID == 2:
            print("Increase Volume(Thumbs up)")
            self.message = "RAISE"
            displayController.youtubeAction(self.message)

        elif gestureID == 3:
            print("Decrease Volume (Thumbs Down)")
            self.message = "LOwER"
            displayController.youtubeAction(self.message)

        elif gestureID == 5:
            print("Stop (Palm)")
            self.message = "STOP"
            displayController.youtubeAction(self.message)

        elif gestureID == 8:
            print("Pause (Fist)")
            self.message = "PAUSE"
            displayController.youtubeAction(self.message)
        pass


class Display_Schema:
    global killer

    def __init__(self, sb_guesture_publisher):
        print("Display Schema ", threading.current_thread())
        self.sb_guesture_publisher = sb_guesture_publisher
        self.display_command = ""
        x = threading.Thread(target=self.youtube_video_control, args=())
        x.start()

    def youtubeAction(self, action):
        self.display_command = action

    def youtube_video_control(self):
        print("youtube video control thread: ", threading.current_thread())
        while not killer.kill_now:
            if self.display_command != "":
                self.sb_guesture_publisher.publish(self.display_command)
                self.display_command = ""
            time.sleep(0.1)


class ClapCartographer(Observer, Subject):
    _state: CompressedImage = None
    previous_audio = None
    new_audio = None
    _observers: List[Observer] = []

    def __init__(self):
        x = threading.Thread(target=self.detectClap, args=())
        x.start()

    def addObserver(self, observer: Observer) -> None:
        print("ClapCartographer: Attached an observer.")
        self._observers.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def setChanged(self, clap_data) -> None:
        self._state = clap_data
        self.notify()

    def update(self, subject: Subject) -> None:
        self.new_audio = subject._state

    def isNoise(self, arr):
        thr = CLAP_THREASHOLD  # Threashold
        sum = 0
        for i in arr:
            sum += abs(i)
        if abs(sum) > thr:
            print(abs(sum))
            intensity = (abs(sum) - thr) / thr * 100
            # print("intensity: ", intensity)
            return True
            # print(abs(sum))
        return False

    def detectClap(self):
        while not killer.kill_now:
            if self.previous_audio != self.new_audio:
                self.previous_audio = self.new_audio
                audio_input_data = self.new_audio
                if self.isNoise(audio_input_data.data):
                    print(
                        "===========================================Clap detected from AudioPerception data: "
                    )
                    clap_detected = True
                    self.setChanged(clap_detected)
                else:
                    self.setChanged(False)
            time.sleep(0.1)


class Face_Params:
    nose = 0
    ear_distance = 0

    def __init__(self) -> None:
        self.nose = 0
        self.ear_distance = 0


def get_dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class CameraObserver(Observer):
    def update(self, subject: Subject) -> None:
        pass


class FaceCartographer(Observer, Subject):
    _face: bool = None
    previous_image = None
    new_image = None
    _state: CompressedImage = None  # Face_Params = None
    _face_params = Face_Params()  # = None
    _observers: List[Observer] = []

    def __init__(self):
        x = threading.Thread(target=self.detectFace, args=())
        x.start()

    def addObserver(self, observer: Observer) -> None:
        print("Subject: Attached an observer.")
        self._observers.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def setChanged(self, face_features) -> None:
        self._face_params.nose = face_features[0]
        self._face_params.ear_distance = face_features[1]
        self.notify()

    def update(self, subject: Subject) -> None:
        self.new_image = subject._state

    def detectFace(self):
        while not killer.kill_now:
            if self.previous_image != self.new_image:
                camera_input_data = self.new_image
                self.previous_image = self.new_image
                current_x = 0
                np_arr = np.frombuffer(camera_input_data.data, np.uint8)
                image = cv2.imdecode(np_arr, 1)
                face_features = []
                # cv2.namedWindow("Image Window",1)
                with mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                ) as face_detection:
                    results = face_detection.process(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )
                    if results.detections:
                        for detection in results.detections:
                            nose = mp_face_detection.get_key_point(
                                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
                            )
                            left = mp_face_detection.get_key_point(
                                detection,
                                mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION,
                            )
                            right = mp_face_detection.get_key_point(
                                detection,
                                mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION,
                            )
                            distance_between_ears = get_dist(left, right)
                            current_x = nose.x
                            face_features.append(current_x)
                            face_features.append(distance_between_ears)
                            self.setChanged(face_features)
                    else:
                        # self.setChanged([0, 0])
                        pass
                # cv2.imshow("Image Window",image)
                # cv2.waitKey(1)

            time.sleep(0.01)


class AttentionBehavior(Observer):
    isClap = False

    def __init__(self, sb_motor):
        self.sb_motor = sb_motor
        print("Attention Behavior ", threading.current_thread())
        x = threading.Thread(target=self.seek_attention, args=())
        x.start()

    def update(self, subject: Subject) -> None:
        if globals.dance_enabled > 0:
            return
        if subject._state == True:
            self.isClap = True
        else:
            self.isClap = False

    def seek_attention(self):
        while not killer.kill_now:
            if self.isClap:
                rand = random.randint(0, 9)
                print("---------------------------- seeking attention", rand)
                if rand % 2 == 0:
                    self.movement_1()
                else:
                    self.movement_2()

                self.isClap = False
            time.sleep(0.1)

    def movement_1(self):
        self.sb_motor.move([5, 15, 15, 5], 20)
        time.sleep(0.1)

        self.sb_motor.move([5, -15, -15, -5], 20)
        time.sleep(0.1)
        self.sb_motor.move([0, 0, 0, 0], 5)

    def movement_2(self):
        self.sb_motor.move([5, 5, 5, 0], 20)
        time.sleep(0.1)
        self.sb_motor.move([-5, -5, -5, 0], 20)
        time.sleep(0.1)
        self.sb_motor.move([0, 0, 0, 0], 5)


class MirrorBehavior(Observer):
    nose = 0
    ear_distance = 0

    def __init__(self, sb_motor):
        self.sb_motor = sb_motor
        print("Mirror Behavior ", threading.current_thread())
        x = threading.Thread(target=self.motor_output, args=())
        x.start()

    def update(self, subject: Subject) -> None:
        if globals.dance_enabled > 0:
            return

        self.nose = subject._face_params.nose
        self.ear_distance = subject._face_params.ear_distance

    def move_robot(self, axis_0, axis_1):
        self.sb_motor.move(
            [axis_0, axis_1, self.sb_motor.motor_pos[2], self.sb_motor.motor_pos[3]]
        )

    def motor_output(self):
        while not killer.kill_now:
            if self.nose != -1 and self.ear_distance != -1:
                right_boundary = 0.55
                left_boundary = 0.45
                left_movement_limit = -25
                right_movement_limit = 25

                delta = 1
                rest_axis_0 = -15

                axis_1 = self.sb_motor.motor_pos[1]
                axis_0 = self.sb_motor.motor_pos[0]

                if self.nose > right_boundary:
                    new_axis_1_position = axis_1 - delta
                    if new_axis_1_position >= left_movement_limit:
                        axis_1 = axis_1 - delta
                        self.move_robot(axis_0, axis_1)
                elif self.nose < left_boundary:
                    new_axis_1_position = axis_1 + delta
                    if new_axis_1_position <= right_movement_limit:
                        axis_1 = axis_1 + delta
                        self.move_robot(axis_0, axis_1)
                print("dist is ", self.ear_distance, "  state is ", globals.face_pos)

                if globals.face_pos == 2:
                    if self.ear_distance < 0.22:
                        globals.face_pos = 0

                if globals.face_pos == 0:
                    if self.ear_distance > 0.23:
                        axis_0 = +5
                        globals.face_pos = 1
                        self.move_robot(axis_0, axis_1)
                elif globals.face_pos == 1:
                    if self.ear_distance < 0.22:
                        axis_0 = rest_axis_0
                        globals.face_pos = 0
                        self.move_robot(axis_0, axis_1)
                    elif self.ear_distance > 0.4:
                        axis_0 = rest_axis_0
                        globals.face_pos = 2
                        self.move_robot(axis_0, axis_1)

                self.nose = -1
                self.ear_distance = -1

            time.sleep(0.1)


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


killer = GracefulKiller()

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String

twist = TwistStamped()
pause = True
my_time = time.time()
is_confused = 0

dance_clock = 0


class Motor_Schema:
    global killer

    def __init__(self, sb_motor_publisher):
        print("Motor Schema ", threading.current_thread())
        self.sb_motor_publisher = sb_motor_publisher
        self.motor_pos = [0, 0, 0, 0]
        self.global_goal = [0, 0, 0, 0]
        self.global_speed = 10
        self.move([-15, 0, 0, 0])
        x = threading.Thread(target=self.motor_control, args=())
        x.start()

    def move(self, goal, speed=globals.default_speed):
        self.global_goal = goal
        self.global_speed = speed

    def inProgress(self):
        return self.motor_pos != self.global_goal

    def motor_control(self):
        print("motor control thread: ", threading.current_thread())
        while not killer.kill_now:
            transient_pos = self.motor_pos
            goal = self.global_goal
            speed = self.global_speed
            if goal != self.motor_pos:
                if goal[0] > self.motor_pos[0]:
                    transient_pos[0] = transient_pos[0] + speed
                    if transient_pos[0] > goal[0]:
                        transient_pos[0] = goal[0]
                elif goal[0] < self.motor_pos[0]:
                    transient_pos[0] = transient_pos[0] - speed
                    if transient_pos[0] < goal[0]:
                        transient_pos[0] = goal[0]

                if goal[1] > self.motor_pos[1]:
                    transient_pos[1] = transient_pos[1] + speed
                    if transient_pos[1] > goal[1]:
                        transient_pos[1] = goal[1]
                elif goal[1] < self.motor_pos[1]:
                    transient_pos[1] = transient_pos[1] - speed
                    if transient_pos[1] < goal[1]:
                        transient_pos[1] = goal[1]

                if goal[2] > self.motor_pos[2]:
                    transient_pos[2] = transient_pos[2] + speed
                    if transient_pos[2] > goal[2]:
                        transient_pos[2] = goal[2]
                elif goal[2] < self.motor_pos[2]:
                    transient_pos[2] = transient_pos[2] - speed
                    if transient_pos[2] < goal[2]:
                        transient_pos[2] = goal[2]

                if goal[3] > self.motor_pos[3]:
                    transient_pos[3] = transient_pos[3] + speed
                    if transient_pos[3] > goal[3]:
                        transient_pos[3] = goal[3]
                elif goal[3] < self.motor_pos[3]:
                    transient_pos[3] = transient_pos[3] - speed
                    if transient_pos[3] < goal[3]:
                        transient_pos[3] = goal[3]

                self.motor_pos = transient_pos

                twist.twist.linear.x = self.motor_pos[0]
                twist.twist.linear.y = self.motor_pos[1]
                twist.twist.linear.z = self.motor_pos[2]
                twist.twist.angular.x = self.motor_pos[3]

                self.sb_motor_publisher.publish(twist)
            time.sleep(0.1)


class DancePerformance:
    global killer, dance_clock

    def __init__(self, sb_motor):
        print("Dance Performance ", threading.current_thread())

        self.danceSteps = []
        self.run = False
        self.stop = False
        self.freeStyle = FreeStyle(None)
        self.sb_motor = sb_motor

    def start_dance(self):
        global dance_clock
        dance_clock = 0
        self.run = True
        # self.execute()
        x = threading.Thread(target=self.execute, args=())
        x.start()

    def stop_dance(self):
        self.run = False

    def addDanceStep(self, danceStep, time):
        self.danceSteps.append([danceStep, time])

    def addFreeStyle(self, freeStyleDanceStep):
        self.freeStyle = freeStyleDanceStep

    def execute(self):
        global dance_clock
        kill = killer.kill_now
        trigger = False

        while not self.run and not killer.kill_now:
            time.sleep(0.1)
            pass
        if killer.kill_now:
            return
        else:
            if kill:
                return
        for actionStep in self.danceSteps:
            dance = actionStep[0]
            dance_release_time = actionStep[1]
            print(
                "----------------------------------- Next step at: ",
                dance_release_time,
                " current time: ",
                dance_clock,
            )
            while dance_clock < dance_release_time and not killer.kill_now:
                time.sleep(0.1)
                pass

            dance.perform(not self.run)
            if not self.run:
                return
        globals.dance_enabled = globals.dance_enabled - 1
        print(
            "============================================ Thank you For Watching !!! ================================"
        )
        print(
            "                                                  Hope you Enjoyed                 "
        )


class DanceStep:
    global killer, dance_clock

    def __init__(self, motor):
        self.motor = motor
        self.steps = []
        self.stop = False

    def addMoves(self, step):
        self.steps.append(step)

    def perform(self, stop):
        count = 0
        for step in self.steps:
            print("..... performing step #", count, step)
            goal = step[0:4]
            speed = step[4]
            self.motor.move(goal, speed)
            count = count + 1
            while self.motor.inProgress() and not killer.kill_now:
                time.sleep(0.1)
                if stop:
                    return
                pass


class FreeStyle(DanceStep):
    def perform(self):
        steps = [
            [10, 0, 0, 20, 10],
            [5, 20, 0, 30, 5],
            [-10, 0, 0, 0, 15],
            [-5, 0, 0, 20, 5],
        ]
        self.addMoves(steps)
        super().perform()

    def confused_random(self, id):
        rand = random.randint(0, 40)
        goal = [0, 0, 0, 0]
        goal[0] = (-1) ** rand * rand
        goal[1] = (-1) ** rand * rand
        goal[2] = (-1) ** rand * rand
        goal[3] = (-1) ** rand * rand

        print("random movement", goal)
        speed = random.randint(3, 10)
        self.motors[id].move(goal, speed)

        goal = self.reverse_goal(goal)
        self.motors[id].move(goal, speed)

    def reverse_goal(self, goal):
        new_goal = [0, 0, 0, 0]
        new_goal[0] = -goal[0]
        new_goal[0] = -goal[1]
        new_goal[0] = -goal[2]
        new_goal[0] = -goal[3]


class DanceBehavior(Observer):
    def __init__(self, sb_motor_0, sb_motor_1, sb_motor_2, sb_motor_3):
        self.sb0_dancePerformance = DancePerformance(sb_motor_0)
        self.sb1_dancePerformance = DancePerformance(sb_motor_1)
        self.sb2_dancePerformance = DancePerformance(sb_motor_2)
        self.sb3_dancePerformance = DancePerformance(sb_motor_3)

        rospy.loginfo("Node started.")
        self.group_name = "survivor_buddy_head"
        self.dancePerformance()

    def update(self, subject: Subject) -> None:
        if subject._state == 1 and globals.dance_enabled == 0:
            globals.dance_enabled = 4
            print(
                "...................................................................................DanceBehavior detected start signal"
            )
            self.sb0_dancePerformance.start_dance()
            self.sb1_dancePerformance.start_dance()
            self.sb2_dancePerformance.start_dance()
            self.sb3_dancePerformance.start_dance()
        if subject._state == 5 and globals.dance_enabled > 0:
            self.sb0_dancePerformance.stop_dance()
            self.sb1_dancePerformance.stop_dance()
            self.sb2_dancePerformance.stop_dance()
            self.sb3_dancePerformance.stop_dance()
            globals.dance_enabled = 0

    def dancePerformance(self):
        self.danceChoreograph0(self.sb0_dancePerformance)
        self.danceChoreograph1(self.sb1_dancePerformance)
        self.danceChoreograph2(self.sb2_dancePerformance)
        self.danceChoreograph3(self.sb3_dancePerformance)

    def danceChoreograph0(self, dancePerformance):
        simpleDanceSteps = []
        global steps1, steps2, steps3, steps4, side_b_1, side_b_reverse, side_t_1, side_t_reverse, side_w_1, side_w_reverse, body_t_1, body_t_reverse
        steps = []

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_reverse)

        steps.append(side_b_reverse)

        steps.append(side_b_reverse)
        # insert solo
        steps.append(steps2)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_1)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps1)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        # 2x
        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_reverse)

        steps.append(side_b_reverse)

        steps.append(side_b_reverse)
        # insert solo
        steps.append(steps3)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_1)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps4)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)
        count = 1
        for step_set in steps:
            danceStep = DanceStep(dancePerformance.sb_motor)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)

        for simpleStep in simpleDanceSteps:
            count = count + 1
            if count == 20:
                dancePerformance.addDanceStep(simpleStep, sync_time)
            dancePerformance.addDanceStep(simpleStep, 1)

    def danceChoreograph1(self, dancePerformance):
        simpleDanceSteps = []

        steps = []

        steps.append(side_w_1)  # 1
        steps.append(side_w_1)  # 2
        steps.append(side_t_reverse)  # 3
        steps.append(side_t_1)  # 4

        steps.append(side_b_1)  # 5
        steps.append(side_b_reverse)  # 6
        # insert solo
        steps.append(steps2)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_1)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps1)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        # 2x
        steps.append(side_w_1)  # 1
        steps.append(side_w_1)  # 2
        steps.append(side_t_reverse)  # 3
        steps.append(side_t_1)  # 4

        steps.append(side_b_1)  # 5
        steps.append(side_b_reverse)  # 6
        # insert solo
        steps.append(steps3)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_1)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps4)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        count = 1
        for step_set in steps:
            danceStep = DanceStep(dancePerformance.sb_motor)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)

        for simpleStep in simpleDanceSteps:
            count = count + 1
            if count == 20:
                dancePerformance.addDanceStep(simpleStep, sync_time)
            dancePerformance.addDanceStep(simpleStep, 1)

    def danceChoreograph2(self, dancePerformance):
        simpleDanceSteps = []

        steps = []

        steps.append(side_w_1)  # 1
        steps.append(side_w_1)  # 2
        steps.append(side_t_reverse)  # 3
        steps.append(side_t_reverse)  # 4

        steps.append(side_b_reverse)  # 5
        steps.append(side_b_reverse)  # 6
        # insert solo
        steps.append(steps2)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_reverse)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps1)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        # 2x
        steps.append(side_w_1)  # 1
        steps.append(side_w_1)  # 2
        steps.append(side_t_reverse)  # 3
        steps.append(side_t_reverse)  # 4

        steps.append(side_b_reverse)  # 5
        steps.append(side_b_reverse)  # 6
        # insert solo
        steps.append(steps3)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_reverse)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps4)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        count = 1
        for step_set in steps:
            danceStep = DanceStep(dancePerformance.sb_motor)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)

        for simpleStep in simpleDanceSteps:
            count = count + 1
            if count == 20:
                dancePerformance.addDanceStep(simpleStep, sync_time)
            dancePerformance.addDanceStep(simpleStep, 1)

    def danceChoreograph3(self, dancePerformance):
        simpleDanceSteps = []

        steps = []

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        steps.append(side_b_reverse)
        # insert solo
        steps.append(steps2)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_reverse)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps1)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        # 2x
        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        steps.append(side_b_reverse)
        # insert solo
        steps.append(steps3)
        # solo end
        steps.append(body_t_1)
        steps.append(body_t_reverse)
        steps.append(neck_w_reverse)
        steps.append(neck_w_reverse)
        # solo 2
        steps.append(steps4)
        steps.append(neck_b_1)
        steps.append(neck_b_reverse)

        steps.append(side_w_1)
        steps.append(side_w_reverse)
        steps.append(side_t_1)
        steps.append(side_t_1)  # 4
        steps.append(side_b_1)

        count = 1
        for step_set in steps:
            danceStep = DanceStep(dancePerformance.sb_motor)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)

        for simpleStep in simpleDanceSteps:
            count = count + 1
            if count == 20:
                dancePerformance.addDanceStep(simpleStep, sync_time)
            dancePerformance.addDanceStep(simpleStep, 1)


def clock_update(event):
    global dance_clock
    dance_clock = dance_clock + time_rate
    # print("                                                                                                 clock updated", dance_clock)


if __name__ == "__main__":
    rospy.init_node("lab_1_node")
    moveit_commander.roscpp_initialize(sys.argv)

    sb_motor_publisher_0 = rospy.Publisher(
        "/sb_0_cmd_state", TwistStamped, queue_size=1
    )
    sb_motor_publisher_1 = rospy.Publisher(
        "/sb_1_cmd_state", TwistStamped, queue_size=1
    )
    sb_motor_publisher_2 = rospy.Publisher(
        "/sb_2_cmd_state", TwistStamped, queue_size=1
    )
    sb_motor_publisher_3 = rospy.Publisher(
        "/sb_3_cmd_state", TwistStamped, queue_size=1
    )

    sb_motor_0 = Motor_Schema(sb_motor_publisher_0)
    sb_motor_1 = Motor_Schema(sb_motor_publisher_1)
    sb_motor_2 = Motor_Schema(sb_motor_publisher_2)
    sb_motor_3 = Motor_Schema(sb_motor_publisher_3)

    gesture_pub = rospy.Publisher("/gesture_publisher", String, queue_size=10)
    displayController = Display_Schema(gesture_pub)

    attentionBehavior = AttentionBehavior(sb_motor_1)
    danceBehavior = DanceBehavior(sb_motor_0, sb_motor_1, sb_motor_2, sb_motor_3)
    clapCartographer = ClapCartographer()
    clapCartographer.addObserver(attentionBehavior)

    handGestureCartographer = HandGestureCartographer()
    youtubeBehavior = YoutubeBehavior()
    handGestureCartographer.addObserver(youtubeBehavior)
    handGestureCartographer.addObserver(danceBehavior)

    cameraPerception = CameraPerception()
    cameraObserver = CameraObserver()
    cameraPerception.addObserver(cameraObserver)
    cameraPerception.addObserver(handGestureCartographer)

    audioPerception = AudioPerception()
    audioPerception.addObserver(clapCartographer)

    camera_sub = rospy.Subscriber(
        "/camera/image/compressed",
        CompressedImage,
        callback=cameraPerception.setChanged,
        queue_size=1,
    )
    audio_sub = rospy.Subscriber(
        "/audio", Float32MultiArray, callback=audioPerception.setChanged, queue_size=1
    )

    mirrorBehavior = MirrorBehavior(sb_motor_1)
    faceCartographer = FaceCartographer()

    cameraPerception.addObserver(faceCartographer)
    faceCartographer.addObserver(mirrorBehavior)

    print("main ", threading.current_thread())

    rospy.Timer(rospy.Duration(time_rate), clock_update)
    rospy.spin()
