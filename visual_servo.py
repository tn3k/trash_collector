#!/usr/env/bin python

# Author: Kent Koehler
# Date: 5/21/2023

import numpy as np

import rospy
import yaml
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool
from enum import Enum

from pd_controller import PD

# Constants related to robot (ROSBOT 2 for final project)
LINEAR_VELOCITY = 0.1               # [m/s]
ANGULAR_VELOCITY_MAX = np.pi/3      # [rad/s]
FREQUENCY = 10                      # [Hz]
DEPTH_THRESHOLD = 1              # [m]

# Topics and services
CMD_VEL_TOPIC = 'cmd_vel'
ERROR_TOPIC = 'error'
DEPTH_TOPIC = 'depth'
EXPLORE_TOPIC = 'explore'
COLLECT_TOPIC = 'collect'

class fsm(Enum):
    EXPLORING = 0
    FOLLOWING = 1
    COLLECTING = 2

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

        # set up publishers, subscribers, and services
        self._cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self._exp_pub = rospy.Publisher(EXPLORE_TOPIC, Bool, queue_size=1)
        self._col_pub = rospy.Publisher(COLLECT_TOPIC, Bool, queue_size=1)
        self._error_sub = rospy.Subscriber(ERROR_TOPIC, Float64, self._error_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber(DEPTH_TOPIC, Float64, self._depth_callback, queue_size=1)

        # sleep to register
        rospy.sleep(2)
    
    def _error_callback(self, msg):
        """
        Parses the error callback to determine whether to drive or not
        """
        error = msg.data
        if self._fsm == fsm.EXPLORING and not np.isnan(error):
            self._fsm = fsm.FOLLOWING
        if self._fsm == fsm.FOLLOWING:
            if np.isnan(error):
                # send true message to exploration node to start exploring again
                msg = Bool()
                msg.data = True
                self._exp_pub.publish(msg)
                self.stop()
                self._fsm = fsm.EXPLORING
            else:
                # send false message to exploration node to stop exploration
                msg = Bool()
                msg.data = False
                self._exp_pub.publish(msg)
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
            self.stop()
            self._fsm = fsm.COLLECTING
            resp = Bool()
            resp.data = True
            self._col_pub.publish(resp)

    def move(self):
        """
        Move the robot with constant linear velocity and angular velocity as
        determined by the PD controller
        """
        twist_msg = Twist()
        # set values
        twist_msg.linear.x = self.linear_velocity
        twist_msg.angular.z = self.angular_velocity
        # send
        self._cmd_pub.publish(twist_msg)

    def stop(self):
        # create empty Twist and send to robot if its currently driving
        if self._fsm == fsm.FOLLOWING:
            stop_msg = Twist()
            self._cmd_pub.publish(stop_msg)

    def spin(self):
        while not rospy.is_shutdown():
            if self._fsm == fsm.FOLLOWING:
                self.move()
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