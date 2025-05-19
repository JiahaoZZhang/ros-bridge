#!/usr/bin/env python
#
# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Welcome to CARLA ROS manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    B            : toggle manual control

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import datetime
import math
from threading import Thread

import numpy as np
from transforms3d.euler import quat2euler
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_b
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_msgs.msg import CarlaStatus
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.msg import CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleControl
from carla_msgs.msg import CarlaLaneInvasionEvent
from carla_msgs.msg import CarlaCollisionEvent
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool

import sys
if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser


import random
import weakref
import cv2
import os 

# VIEW_WIDTH = 1920 // 2
# VIEW_HEIGHT = 1080 // 2
# #Load the YOLO model, the weights and cfg files
# model_weights = '/home/jiahao.zhang/CTM_CONNECT/carla-ros-bridge/ros2_ws/src/carla_ros/carla_manual_control/yolo/yolov3.weights'
# model_cfg = '/home/jiahao.zhang/CTM_CONNECT/carla-ros-bridge/ros2_ws/src/carla_ros/carla_manual_control/yolo/yolov3.cfg'
# net = cv2.dnn.readNet(modelConfiguration, modelWeight)

# classFiles = '/home/jiahao.zhang/CTM_CONNECT/carla-ros-bridge/ros2_ws/src/carla_ros/carla_manual_control/yolo/coco.names'
# classNames = []
# with open(classFiles,'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')
# np.random.seed(42)
# colors = np.random.randint(0, 255, size=(len(classNames),3), dtype='uint8')

# layer_names = net.getLayerNames()
# outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class ManualControl(CompatibleNode):
    """
    Handle the rendering
    """

    def __init__(self, resolution):
        super(ManualControl, self).__init__("ManualControl")
        self._surface = None
        self.role_name = self.get_param("role_name", "ego_vehicle")
        self.hud = HUD(self.role_name, resolution['width'], resolution['height'], self)
        self.controller = DualControl(self.role_name, self.hud, self)

        self.image_subscriber = self.new_subscription(
            # Image, "/carla/{}/rgb_front/image".format(self.role_name),
            Image, "/carla/{}/rgb_view/image".format(self.role_name),
            self.on_view_image, qos_profile=10)

        self.collision_subscriber = self.new_subscription(
            CarlaCollisionEvent, "/carla/{}/collision".format(self.role_name),
            self.on_collision, qos_profile=10)

        self.lane_invasion_subscriber = self.new_subscription(
            CarlaLaneInvasionEvent, "/carla/{}/lane_invasion".format(self.role_name),
            self.on_lane_invasion, qos_profile=10)

        self.img = None
        
        # Load YOLO
        model_weights = './src/carla_ros/carla_manual_control/yolo/yolov3_darknet.weights'
        model_cfg = './src/carla_ros/carla_manual_control/yolo/yolov3.cfg'                                                                                                                                                                                             
        self._frameWidth = 640
        self._frameHeight = 480
        self._net = cv2.dnn.readNet(model_weights, model_cfg)

        self._classes = []
        with open("./src/carla_ros/carla_manual_control/yolo/coco.names", "r") as f:
            self._classes = [line.strip() for line in f.readlines()] # we put the names in to an array


    def on_collision(self, data):
        """
        Callback on collision event
        """
        intensity = math.sqrt(data.normal_impulse.x**2 +
                              data.normal_impulse.y**2 + data.normal_impulse.z**2)
        self.hud.notification('Collision with {} (impulse {})'.format(
            data.other_actor_id, intensity))

    def on_lane_invasion(self, data):
        """
        Callback on lane invasion event
        """
        text = []
        for marking in data.crossed_lane_markings:
            if marking is CarlaLaneInvasionEvent.LANE_MARKING_OTHER:
                text.append("Other")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                text.append("Broken")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                text.append("Solid")
            else:
                text.append("Unknown ")
        self.hud.notification('Crossed line %s' % ' and '.join(text))


############################################################################################# yolo def ##########################################
    def draw_labels(self, boxes, confs, colors, class_ids, classes, img): 
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            print(range(len(boxes)))
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                print(i)
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
        return img
        
    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.5:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

##########################################################################################################################################


    def on_view_image(self, image):
        """
        Callback when receiving a camera image
        """
        # array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
        # array = np.reshape(array, (image.height, image.width, 4))
        # array = array[:, :, :3]
        # array = array[:, :, ::-1]
        # self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

#########################################################################################################################################
############################################################  yolov3 ####################################################################
        array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        layers_names = self._net.getLayerNames()
        output_layers = [layers_names[i-1] for i in self._net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size = (len(self._classes), 3))

        # Loading image
        i3 = cv2.resize(array, (self._frameWidth, self._frameHeight), None)
        self.img = cv2.cvtColor(i3,cv2.COLOR_RGB2BGR)
        height, width, channels = self.img.shape
        # Detect image
        blob = cv2.dnn.blobFromImage(self.img, 1/255, (self._frameWidth, self._frameHeight), swapRB=True, crop = False)
        self._net.setInput(blob)
        outs = self._net.forward(output_layers)
        # print(outs)

        boxes = []
        confidences = []
        classIDs = []
        h, w = self.img.shape[:2]

        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.0f}%".format(self._classes[classIDs[i]], int(confidences[i] * 100))
                cv2.putText(self.img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        self._surface = pygame.surfarray.make_surface(self.img.swapaxes(0, 1))

        # Showing informations on the screen
        # class_ids = []
        # confidences = []
        # boxes = []
        # for out in outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > 0.8:
        #             #Object detected
        #             center_x = int(detection[0] * width)
        #             center_y = int(detection[1] * height)
        #             w = int(detection[2] * width)
        #             h = int(detection[3] * height)

        #             # Rectangle coordinates
        #             x = int(center_x - w / 2)
        #             y = int(center_y -h / 2)

        #             boxes.append([x, y, w, h])
        #             confidences.append(float(confidence))
        #             # Name of the object
        #             class_ids.append(class_id)
        # #print(center_x, center_y)

        # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # colors = np.random.uniform(0, 255, size=(len(boxes),3))
        # for i in range(len(boxes)):
        #     if i in indexes:
        # #for i in indexes.flatten():
        #         x,y,w,h = boxes[i]
        #         label = str(classes[class_ids[i]])
        #         confidence = str(round(confidences[i], 2))
        #         color = colors[i]
        #         cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
        #         cv2.putText(self.img, label + " " + confidence, (x + 60, y + 60), font, 0.8, (255, 255, 200), 1)
        



#########################################################################################################################################
############################################################  yolov3 ####################################################################


    def render(self, game_clock, display):
        """
        render the current image
        """

        do_quit = self.controller.parse_events(game_clock)
        if do_quit:
            return
        self.hud.tick(game_clock)

        if self._surface is not None:
            display.blit(self._surface, (0, 0))
            # cv2.namedWindow("frame")
            # cv2.imshow('frame',self.img)
            # cv2.waitKey(10)
        self.hud.render(display)

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    """
    Handle input events
    """

    def __init__(self, role_name, hud, node):
        self.role_name = role_name
        self.hud = hud
        self.node = node

        self._autopilot_enabled = False
        self._control = CarlaEgoVehicleControl()
        self._steer_cache = 0.0

        fast_qos = QoSProfile(depth=10)
        fast_latched_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.vehicle_control_manual_override_publisher = self.node.new_publisher(
            Bool,
            "/carla/{}/vehicle_control_manual_override".format(self.role_name),
            qos_profile=fast_latched_qos)

        self.vehicle_control_manual_override = True

        self.auto_pilot_enable_publisher = self.node.new_publisher(
            Bool,
            "/carla/{}/enable_autopilot".format(self.role_name),
            qos_profile=fast_qos)

        self.vehicle_control_publisher = self.node.new_publisher(
            CarlaEgoVehicleControl,
            "/carla/{}/vehicle_control_cmd_manual".format(self.role_name),
            qos_profile=fast_qos)

        self.carla_status_subscriber = self.node.new_subscription(
            CarlaStatus,
            "/carla/status",
            self._on_new_carla_frame,
            qos_profile=10)

        self.set_autopilot(self._autopilot_enabled)

        self.set_vehicle_control_manual_override(
            self.vehicle_control_manual_override)  # disable manual override

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('./src/carla_ros/carla_manual_control/src/carla_manual_control/wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))


    def set_vehicle_control_manual_override(self, enable):
        """
        Set the manual control override
        """
        self.hud.notification('Set vehicle control manual override to: {}'.format(enable))
        self.vehicle_control_manual_override_publisher.publish((Bool(data=enable)))

    def set_autopilot(self, enable):
        """
        enable/disable the autopilot
        """
        self.auto_pilot_enable_publisher.publish(Bool(data=enable))

    # pylint: disable=too-many-branches
    def parse_events(self, clock):
        """
        parse an input event
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    self.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and
                                          pygame.key.get_mods() & KMOD_SHIFT):
                    self.hud.help.toggle()
                elif event.key == K_b:
                    self.vehicle_control_manual_override = not self.vehicle_control_manual_override
                    self.set_vehicle_control_manual_override(self.vehicle_control_manual_override)
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self.hud.notification(
                        '%s Transmission' %
                        ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self.set_autopilot(self._autopilot_enabled)
                    self.hud.notification('Autopilot %s' %
                                          ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._parse_vehicle_wheel()
            self._control.reverse = self._control.gear < 0

    def _on_new_carla_frame(self, data):
        """
        callback on new frame

        As CARLA only processes one vehicle control command per tick,
        send the current from within here (once per frame)
        """
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            try:
                self.vehicle_control_publisher.publish(self._control)
            except Exception as error:
                self.node.logwarn("Could not send vehicle control: {}".format(error))

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        parse key events
        """
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = bool(keys[K_SPACE])


    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0.0:
            throttleCmd = 0.0
        elif throttleCmd > 1.0:
            throttleCmd = 1.0

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0.0:
            brakeCmd = 0.0
        elif brakeCmd > 1.0:
            brakeCmd = 1.0

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """
    Handle the info display
    """

    def __init__(self, role_name, width, height, node):
        self.role_name = role_name
        self.dim = (width, height)
        self.node = node
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self._show_info = True
        self._info_text = []
        self.vehicle_status = CarlaEgoVehicleStatus()

        self.vehicle_status_subscriber = node.new_subscription(
            CarlaEgoVehicleStatus, "/carla/{}/vehicle_status".format(self.role_name),
            self.vehicle_status_updated, qos_profile=10)

        self.vehicle_info = CarlaEgoVehicleInfo()
        self.vehicle_info_subscriber = node.new_subscription(
            CarlaEgoVehicleInfo,
            "/carla/{}/vehicle_info".format(self.role_name),
            self.vehicle_info_updated, 
            qos_profile=QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.x, self.y, self.z = 0, 0, 0
        self.yaw = 0
        self.latitude = 0
        self.longitude = 0
        self.manual_control = False

        self.gnss_subscriber = node.new_subscription(
            NavSatFix,
            "/carla/{}/gnss".format(self.role_name),
            self.gnss_updated,
            qos_profile=10)

        self.odometry_subscriber = node.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(self.role_name),
            self.odometry_updated,
            qos_profile=10
        )

        self.manual_control_subscriber = node.new_subscription(
            Bool,
            "/carla/{}/vehicle_control_manual_override".format(self.role_name),
            self.manual_control_override_updated,
            qos_profile=10)

        self.carla_status = CarlaStatus()
        self.status_subscriber = node.new_subscription(
            CarlaStatus,
            "/carla/status",
            self.carla_status_updated,
            qos_profile=10)

    def tick(self, clock):
        """
        tick method
        """
        self._notifications.tick(clock)

    def carla_status_updated(self, data):
        """
        Callback on carla status
        """
        self.carla_status = data
        self.update_info_text()

    def manual_control_override_updated(self, data):
        """
        Callback on vehicle status updates
        """
        self.manual_control = data.data
        self.update_info_text()

    def vehicle_status_updated(self, vehicle_status):
        """
        Callback on vehicle status updates
        """
        self.vehicle_status = vehicle_status
        self.update_info_text()

    def vehicle_info_updated(self, vehicle_info):
        """
        Callback on vehicle info updates
        """
        self.vehicle_info = vehicle_info
        self.update_info_text()

    def gnss_updated(self, data):
        """
        Callback on gnss position updates
        """
        self.latitude = data.latitude
        self.longitude = data.longitude
        self.update_info_text()

    def odometry_updated(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
        _, _, yaw = quat2euler(
            [data.pose.pose.orientation.w,
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z])
        self.yaw = math.degrees(yaw)
        self.update_info_text()

    def update_info_text(self):
        """
        update the displayed info text
        """
        if not self._show_info:
            return

        x, y, z = self.x, self.y, self.z
        yaw = self.yaw

        heading = 'N' if abs(yaw) < 89.5 else ''
        heading += 'S' if abs(yaw) > 90.5 else ''
        heading += 'E' if 179.5 > yaw > 0.5 else ''
        heading += 'W' if -0.5 > yaw > -179.5 else ''
        fps = 0

        time = str(datetime.timedelta(seconds=self.node.get_time()))[:10]

        if self.carla_status.fixed_delta_seconds:
            fps = 1 / self.carla_status.fixed_delta_seconds
        self._info_text = [
            'Frame: % 22s' % self.carla_status.frame,
            'Simulation time: % 12s' % time,
            'FPS: % 24.1f' % fps, '',
            'Vehicle: % 20s' % ' '.join(self.vehicle_info.type.title().split('.')[1:]),
            'Speed:   % 15.0f km/h' % (3.6 * self.vehicle_status.velocity),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (x, y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (self.latitude, self.longitude)),
            'Height:  % 18.0f m' % z, ''
        ]
        self._info_text += [
            ('Throttle:', self.vehicle_status.control.throttle, 0.0, 1.0),
            ('Steer:', self.vehicle_status.control.steer, -1.0, 1.0),
            ('Brake:', self.vehicle_status.control.brake, 0.0, 1.0),
            ('Reverse:', self.vehicle_status.control.reverse),
            ('Hand brake:', self.vehicle_status.control.hand_brake),
            ('Manual:', self.vehicle_status.control.manual_gear_shift),
            'Gear:        %s' % {
                -1: 'R',
                0: 'N'
            }.get(self.vehicle_status.control.gear, self.vehicle_status.control.gear), ''
        ]
        self._info_text += [('Manual ctrl:', self.manual_control)]
        if self.carla_status.synchronous_mode:
            self._info_text += [('Sync mode running:', self.carla_status.synchronous_mode_running)]
        self._info_text += ['', '', 'Press <H> for help']

    def toggle_info(self):
        """
        show/hide the info text
        """
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """
        display a notification for x seconds
        """
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """
        display an error
        """
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """
        render the display
        """
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)
                                  ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset + 50, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + int(f * (bar_width - 6)), v_offset + 8),
                                               (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8),
                                               (int(f * bar_width), 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """
    Support Class for info display, fade out text
    """

    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """
        set the text
        """
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        """
        tick for fading
        """
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """
        render the fading
        """
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """
    Show the help text
    """

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """
        Show/hide the help
        """
        self._render = not self._render

    def render(self, display):
        """
        render the help
        """
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main(args=None):
    """
    main function
    """
    roscomp.init("manual_control", args=args)

    # resolution should be similar to spawned camera with role-name 'view'
    resolution = {"width": 640, "height": 480}
    # print(os.getcwd())
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("CARLA ROS manual control")

    try:
        display = pygame.display.set_mode((resolution['width'], resolution['height']),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        manual_control_node = ManualControl(resolution)
        clock = pygame.time.Clock()

        executor = roscomp.executors.MultiThreadedExecutor()
        executor.add_node(manual_control_node)

        spin_thread = Thread(target=manual_control_node.spin)
        spin_thread.start()

        while roscomp.ok():
            clock.tick_busy_loop(60)
            if manual_control_node.render(clock, display):
                return
            pygame.display.flip()

    
    except KeyboardInterrupt:
        roscomp.loginfo("User requested shut down.")
    finally:
        roscomp.shutdown()
        spin_thread.join()
        pygame.quit()


if __name__ == '__main__':
    main()
