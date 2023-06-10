#!/usr/env/bin python

# Author: Kent Koehler
# Date: 5/21/2023

import numpy as np

import rospy
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, Bool
from enum import Enum

from pd_controller import PD

# Constants related to robot (ROSBOT 2 for final project)
LINEAR_VELOCITY = 0.1               # [m/s]
ANGULAR_VELOCITY_MAX = np.pi/3      # [rad/s]
FREQUENCY = 10                      # [Hz]
DEPTH_THRESHOLD = 1              # [m]

# for random walk
MIN_SCAN_ANGLE_RAD = -30.0 / 180 * np.pi
MAX_SCAN_ANGLE_RAD = +30.0 / 180 * np.pi
MIN_THRESHOLD_DISTANCE = 0.6    # [m]

# Topics and services
CMD_VEL_TOPIC = 'cmd_vel'
ERROR_TOPIC = 'error'
DEPTH_TOPIC = 'depth'
SCAN_TOPIC = 'scan'

class fsm(Enum):
    EXPLORING = 0
    FOLLOWING = 1
    MARKING = 2

class VisualServo:
    def __init__(self, gain, frequency=FREQUENCY, linear_velocity=LINEAR_VELOCITY, 
                 max_angular_velocity=ANGULAR_VELOCITY_MAX, depth_threshold=DEPTH_THRESHOLD):
        """
        Constructor

        :param gain: proportional and differential gain for the PD controller
        :param linear_velocity: constant linear velocity applied while moving
        :param max_angular_velocity: maximum angular velocity for maneuvers
        """
        # start node in stopped mode
        self._fsm = fsm.EXPLORING
        self.pd = PD(gain[0], gain[1])

        # set parameters
        self.depth_threshold = depth_threshold
        self.linear_velocity = linear_velocity
        self.angular_velocity = 0
        self.direction = 0
        self.max_angular_velocity = max_angular_velocity
        self.rate = rospy.Rate(frequency)

        # for random walk
        self.min_laser_angle = MIN_SCAN_ANGLE_RAD
        self.max_laser_angle = MAX_SCAN_ANGLE_RAD
        self.laser_threshold = MIN_THRESHOLD_DISTANCE
        self.closest_distance = np.inf
        self.close_obstacle = False

        # set up publishers, subscribers, and services
        self._cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self._error_sub = rospy.Subscriber(ERROR_TOPIC, Float64, self._error_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber(DEPTH_TOPIC, Float64, self._depth_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)

        # sleep to register
        rospy.sleep(2)  

    def _error_callback(self, msg):
        """
        Parses the error callback to determine whether to drive or not
        """
        error = msg.data
        if self._fsm == fsm.EXPLORING and not np.isnan(error):
            # start following the object
            self._fsm = fsm.FOLLOWING
        if self._fsm == fsm.FOLLOWING:
            if np.isnan(error):
                # start exploration again
                self.stop()
                self._fsm = fsm.EXPLORING
            else:
                # calculate the error and proceed to the object
                self._calculate_error(error)

    def _calculate_error(self, error):
        """
        Calculates the error and sends to the PD controller to return an angular
        velocity command
        """
        time = rospy.Time.now().to_sec()
        angular_v = self.pd.step(error, time)
        if angular_v >= 0:
            self.direction = 1
            self.angular_velocity = min(angular_v, self.max_angular_velocity)
        else:
            self.direction = -1
            self.angular_velocity = max(angular_v, -self.max_angular_velocity)

    def _depth_callback(self, msg):
        """
        Stop the robot once it gets to the desired depth from the object
        """
        depth = msg.data
        if depth <= self.depth_threshold:
            # stop the robot and start the marking process
            self.stop()
            self._fsm = fsm.MARKING

    def move(self, linear_v, angular_v):
        """
        Move the robot with constant linear velocity and angular velocity as
        determined by the PD controller
        """
        twist_msg = Twist()
        # set values
        twist_msg.linear.x = linear_v
        twist_msg.angular.z = angular_v
        # send
        self._cmd_pub.publish(twist_msg)

    def stop(self):
        # create empty Twist and send to robot if its currently driving
        stop_msg = Twist()
        self._cmd_pub.publish(stop_msg)

    def _laser_callback(self, msg):
        if self._fsm == fsm.EXPLORING and not self.close_obstacle:
            min_index = int(abs(self.min_laser_angle) / msg.angle_increment)
            max_index = int(abs(self.max_laser_angle) / msg.angle_increment)
            ranges_in_view = msg.ranges[:min_index] + msg.ranges[-max_index:]
            # get the minimum distance
            closest_distance = min(ranges_in_view)
            if closest_distance < self.laser_threshold:
                self.close_obstacle = True

    def random_walk(self):
        """
        Random walk exploration when in explore mode
        """
        if not self.close_obstacle:
            self.move(self.linear_velocity, 0)
        else:
            target_yaw = (np.random.random() - 0.5 * 2 * 3 * np.pi/2)
            # set direction to ccw if yaw is positive and cw if negative
            direction = 1 if target_yaw > 0 else -1
            duration = abs(target_yaw) / self.max_angular_velocity   # duration of rotation
            start_time = rospy.get_rostime()                # get start time
            # loop that turns the robot
            while not rospy.is_shutdown():
                self.move(0, direction * self.max_angular_velocity)
                if rospy.get_rostime() - start_time >= rospy.Duration(duration) or self._fsm != fsm.EXPLORING:
                    self.close_obstacle = False
                    self.angular_velocity = 0
                    break
                self.rate.sleep()

    def mark_object(self):
        # Mark the object here
        duration = np.pi / self.max_angular_velocity
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            self.move(0, self.max_angular_velocity)
            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                self._fsm = fsm.EXPLORING
                break
            self.rate.sleep()

    def spin(self):
        while not rospy.is_shutdown():
            self.move(self.linear_velocity, self.angular_velocity)
            # if self._fsm == fsm.FOLLOWING:
            #     self.move(self.linear_velocity, self.angular_velocity)
            # elif self._fsm == fsm.EXPLORING:
            #     self.random_walk()
            # elif self._fsm == fsm.MARKING:
            #     self.mark_object()
            self.rate.sleep()

def main():
    """Main function"""
    # get gains from yaml file
    print("Loading gains...")
    with open('controller_gain.yml', 'r') as f:
        gains = yaml.safe_load(f)
    print("Kp: {} and Kd: {}".format(gains['kp'], gains['kd']))
    # init node
    print("Initializing Node...")
    rospy.init_node('follow_object')
    visual_servo = VisualServo(gain=(gains['kp'], gains['kd']))
    print("Node Initialized")
    # stop robot on shutdown
    rospy.on_shutdown(visual_servo.stop)
    # start node spinning
    try:
        print("Spinning...")
        visual_servo.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted")

if __name__ == "__main__":
    main()