#!/usr/env/bin python

# Author: Kent Koehler
# Date: 5/21/2023

import numpy as np

import rospy
import yaml
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, SetBoolResponse
from enum import Enum

from pd_controller import PD

# Constants related to robot (ROSBOT 2 for final project)
LINEAR_VELOCITY = 0.2               # [m/s]
ANGULAR_VELOCITY_MAX = np.pi/3      # [rad/s]
FREQUENCY = 10                      # [Hz]

# Topics and services
CMD_VEL_TOPIC = 'cmd_vel'
ERROR_TOPIC = 'error'
ON_OFF_SERVICE = 'follow_object'

class fsm(Enum):
    STOP = 0
    WAITING_FOR_IMAGE = 1
    DRIVING = 2

class VisualServo:
    def __init__(self, gain, frequency=FREQUENCY, linear_velocity=LINEAR_VELOCITY, 
                 max_angular_velocity=ANGULAR_VELOCITY_MAX):
        """
        Constructor

        :param gain: proportional and differential gain for the PD controller
        :param linear_velocity: constant linear velocity applied while moving
        :param max_angular_velocity: maximum angular velocity for maneuvers
        """
        # start node in stopped mode
        self._fsm = fsm.STOP

        # set up publishers, subscribers, and services
        self._cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self._error_sub = rospy.Subscriber(ERROR_TOPIC, Float64, self._error_callback, queue_size=1)
        self._on_off_srv = rospy.Service(ON_OFF_SERVICE, SetBool, self._on_off)

        # sleep to register
        rospy.sleep(2)

        # initialize PD controller
        self.pd = PD(gain[0], gain[1])

        # set parameters
        self.linear_velocity = linear_velocity
        self.angular_velocity = 0
        self.max_angular_velocity = max_angular_velocity
        self.rate = rospy.Rate(frequency)

    def _on_off(self, req):
        """
        Callback for turning on and off the visual tracking of an object
        """
        # initialize response
        resp = SetBoolResponse()
        # if tracking turned off
        if not req.data:
            self.stop()
            self._fsm = fsm.STOP
            resp.success = True
            resp.message = 'Object Tracking Off'
        else:
            # if tracking off and turned on
            if self._fsm == fsm.STOP:
                self._fsm = fsm.WAITING_FOR_IMAGE
                resp.success = True
                resp.message = 'Object Tracking On'
            # if tracking on and turned on
            else:
                resp.success = False
                resp.message = 'Object Tracking Already On'
        return resp
    
    def _error_callback(self, msg):
        """
        Calculate the error and send to the PD controller to return an angular
        velocity command
        """
        if not self._fsm == fsm.STOP:
            error = msg.data
            time = rospy.Time.now().to_sec()
            angular_v = self.pd.step(error, time)
            if angular_v >= 0:
                self.angular_velocity = min(angular_v, self.max_angular_velocity)
            else:
                self.angular_velocity = max(angular_v, -self.max_angular_velocity)
            # set the fsm to driving
            if self._fsm == fsm.WAITING_FOR_IMAGE:
                self._fsm = fsm.DRIVING


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
        if self._fsm == fsm.DRIVING:
            stop_msg = Twist()
            self._cmd_pub.publish(stop_msg)

    def spin(self):
        while not rospy.is_shutdown():
            if self._fsm == fsm.DRIVING:
                self.move()
            self.rate.sleep()

def main():
    """Main function"""
    # get gains from yaml file
    with open('controller_gain.yml', 'r') as f:
        gains = yaml.safe_load(f)
    # init node
    rospy.init_node('follow_object')
    visual_servo = VisualServo(gain=(gains['kp'], gains['kd']))
    # stop robot on shutdown
    rospy.on_shutdown(visual_servo.stop)
    # start node spinning
    try:
        visual_servo.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted")

if __name__ == "__main__":
    main()