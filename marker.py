#!/usr/env/bin python

# Author: Kent Koehler
# Date: 5/21/2023

import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float64
from enum import Enum

from pd_controller import PD

# Constants related to robot (ROSBOT 2 for final project)
LINEAR_VELOCITY = 0.1               # [m/s]
ANGULAR_VELOCITY_MAX = np.pi/3      # [rad/s]
FREQUENCY = 10                      # [Hz]
DEPTH_THRESHOLD = 1              # [m]

# for PD controller
GAIN = (0.01, 0.0005)

# for random walk
MIN_SCAN_ANGLE_RAD = -30.0 / 180 * np.pi
MAX_SCAN_ANGLE_RAD = +30.0 / 180 * np.pi
MIN_THRESHOLD_DISTANCE = 0.6    # [m]

# Topics and services
CMD_VEL_TOPIC = 'cmd_vel'
SCAN_TOPIC = 'scan'
CAMERA_COLOR_TOPIC = 'camera/color/image_raw' # camera/rgb/image_raw for ROSBOT 2
CAMERA_DEPTH_TOPIC = 'camera/depth/image_raw'

# parameters for object detection
MAX_COLOR_DIFF = 50
MIN_REGION_SIZE = 5

OBJECT_COLOR = (0, 100, 0)

# fsm for determining action taken
class fsm(Enum):
    EXPLORING = 0
    FOLLOWING = 1
    MARKING = 2

class Marker:
    def __init__(self, gain=GAIN, frequency=FREQUENCY, linear_velocity=LINEAR_VELOCITY, 
                 max_angular_velocity=ANGULAR_VELOCITY_MAX, depth_threshold=DEPTH_THRESHOLD, 
                 max_color_diff=MAX_COLOR_DIFF, min_region_size=MIN_REGION_SIZE, object_color=OBJECT_COLOR):
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

        # for object detection
        self.max_color_diff = max_color_diff
        self.min_region_size = min_region_size
        self.object_color = object_color
        self.bridge = CvBridge()
        self.object = None
        self.error = None
        self.depth = None

        # set up publishers, subscribers, and services
        self._cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self._laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)
        self._image_sub = rospy.Subscriber(CAMERA_COLOR_TOPIC, Image, self._image_callback, queue_size=1, buff_size=2*52428800)
        self._depth_sub = rospy.Subscriber(CAMERA_DEPTH_TOPIC, Image, self._depth_callback, queue_size=1)

        # sleep to register
        rospy.sleep(2)  

    def _image_callback(self, msg):
        """Finds the regions of matching color"""
        # Initialize
        visited = set()
        regions = []
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv.pyrDown(cv.pyrDown(raw_image))
        height = len(image)
        width = len(image[0])
		# Looping over all the pixels in the image
        for y in range(height):
            for x in range(width):
                pixelColor = image[y, x]
				# checking if pixel is unvisited and of the correct color
                if (x,y) not in visited and self.isSimilarColor(pixelColor):
                    newRegion = []
                    newRegion.append((x,y))
                    toBeVisited = []
                    toBeVisited.append((x,y))
                    visited.add((x,y))
					# As long as there are pixels to be visited the loop keeps going
                    while (len(toBeVisited) > 0):
                        curr_pixel = toBeVisited.pop(0)
                        newRegion.append(curr_pixel)
						# Checking if neighbors are of the correct color
                        for y2 in range(max(0,curr_pixel[1]-1), min(height-1, curr_pixel[1]+1) + 1):
                            for x2 in range(max(0,curr_pixel[0]-1), min(width-1, curr_pixel[0]+1) + 1):
                                currentPixelColor = image[y2, x2]
                                if ((x2,y2) not in visited and self.isSimilarColor(currentPixelColor)):
                                    visited.add((x2,y2))
                                    toBeVisited.append((x2,y2))
					# If the region is large enough it is added to regions
                    if len(newRegion) >= self.min_region_size:
                        regions.append(newRegion)
        # calculate the largest region    
        largestRegion = self.largestRegion(regions)
        self.object = largestRegion
        # calculate image error
        self.error = (self.calculateHorizontalError(largestRegion, width / 2) 
                      if len(largestRegion) > 0 else np.nan)
        self._error_callback()

    def calculateHorizontalError(self, region, center_x):
        total_x = 0
        for point in region:
            total_x += point[0]
        avg_x = total_x / len(region)

        return center_x - avg_x

    def isSimilarColor(self,color): 
        """Determines if two colors are similar given the threshold"""
        # Calculate the total rgb difference
        total_difference = abs(color[0] - self.object_color[0]) + abs(color[1] - self.object_color[1]) + abs(color[2] - self.object_color[2])
        # If the total difference is greater than the threshold return false
        # otherwise return true
        return not total_difference > self.max_color_diff

    def largestRegion(self, regions):
        """Finds the largest region in a list of regions"""
        # Init largest region variable as empty list
        largest_region = []
        # Loop through the regions and compare lengths to the current largest region
        for region in regions:
            if len(region) > len(largest_region):
                largest_region = region
        return largest_region

    def _depth_callback(self, msg):
        if self._fsm == fsm.FOLLOWING:
            raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            image = cv.pyrDown(cv.pyrDown(raw_image))
            if self.object is not None and len(self.object) != 0:
                total, size = 0, 0
                for point in self.object:
                    if not np.isnan(image[point[1], point[0]]):
                        total += image[point[1], point[0]]
                        size += 1
                if size != 0:
                    self.depth = total / size
                else:
                    self.depth = None
            if self.depth is not None and self.depth <= self.depth_threshold:
                # stop the robot and start the marking process
                self.stop()
                self._fsm = fsm.MARKING

    def _error_callback(self):
        """
        Parses the error callback to determine whether to drive or not
        """
        if self._fsm == fsm.EXPLORING and not np.isnan(self.error):
            # start following the object
            self._fsm = fsm.FOLLOWING
        if self._fsm == fsm.FOLLOWING:
            if np.isnan(self.error):
                # start exploration again
                self.stop()
                self._fsm = fsm.EXPLORING
            else:
                # calculate the error and proceed to the object
                self._calculate_error(self.error)

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
            if self._fsm == fsm.FOLLOWING:
                self.move(self.linear_velocity, self.angular_velocity)
            elif self._fsm == fsm.EXPLORING:
                self.random_walk()
            elif self._fsm == fsm.MARKING:
                self.mark_object()
            self.rate.sleep()

def main():
    """Main function"""
    # get gains from yaml file
    print("Initializing Node...")
    rospy.init_node('follow_object')
    visual_servo = Marker()
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