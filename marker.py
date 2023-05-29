#!/usr/env/bin python

# Author: Kent Koehler and Jack Desmond
# Date: 5/21/2023

import numpy as np
import tf
import cv2 as cv
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from enum import Enum

from pd_controller import PD

# Constants related to robot (ROSBOT 2 for final project)
LINEAR_VELOCITY = 0.1               # [m/s]
ANGULAR_VELOCITY_MAX = np.pi/6      # [rad/s]
FREQUENCY = 10                      # [Hz]
DEPTH_THRESHOLD = 0.6              # [m]

# for PD controller
GAIN = (0.01, 0.0005)

# for random walk
MIN_SCAN_ANGLE_RAD = -30.0 / 180 * np.pi
MAX_SCAN_ANGLE_RAD = +30.0 / 180 * np.pi
MIN_THRESHOLD_DISTANCE = 0.6    # [m]

# to determine if a position is within another radius
MARKER_THRESHOLD = 1

# For returning the object
RETURN_LOCATION = (0, 0)    # in odom frame

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
                 max_color_diff=MAX_COLOR_DIFF, min_region_size=MIN_REGION_SIZE, object_color=OBJECT_COLOR,
                 marker_threshold=MARKER_THRESHOLD):
        """
        Constructor

        :param gain: proportional and differential gain for the PD controller
        :param linear_velocity: constant linear velocity applied while moving
        :param max_angular_velocity: maximum angular velocity for maneuvers
        """
        # start node in stopped mode
        self._fsm = fsm.EXPLORING
        self.pd = PD(gain[0], gain[1])

        # transformer listener
        self.listener = tf.TransformListener()

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

        # for camera callbacks
        self.depth_image = None

        # for object detection
        self.max_color_diff = max_color_diff
        self.min_region_size = min_region_size
        self.object_color = object_color
        self.image_center = 0
        self.bridge = CvBridge()
        self.regions = []
        self.object = None
        self.error = np.nan
        self.depth = np.nan

        # for marking
        self.marked = []
        self.marker_threshold = marker_threshold

        # set up publishers, subscribers, and services
        self._cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self._laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)
        self._image_sub = rospy.Subscriber(CAMERA_COLOR_TOPIC, Image, self._image_callback, queue_size=1, buff_size=2*52428800)
        self._depth_sub = rospy.Subscriber(CAMERA_DEPTH_TOPIC, Image, self._depth_callback, queue_size=1)

        # sleep to register
        rospy.sleep(2)  

################################################################################
# Callbacks
################################################################################

    def _image_callback(self, msg):
        """Finds the regions of matching color"""
        # Initialize
        visited = set()
        regions = []
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv.pyrDown(cv.pyrDown(raw_image))
        height = len(image)
        width = len(image[0])
        self.image_center = width // 2
		# Looping over all the pixels in the image
        for y in range(height):
            for x in range(width):
                pixelColor = image[y, x]
				# checking if pixel is unvisited and of the correct color
                if (x,y) not in visited and self.is_similar_color(pixelColor):
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
                                if ((x2,y2) not in visited and self.is_similar_color(currentPixelColor)):
                                    visited.add((x2,y2))
                                    toBeVisited.append((x2,y2))
					# If the region is large enough it is added to regions
                    if len(newRegion) >= self.min_region_size:
                        regions.append(newRegion)
        # calculate the largest region    
        self.regions = regions

    def _depth_callback(self, msg):
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv.pyrDown(cv.pyrDown(raw_image))
        self.depth_image = image
        if self.object is not None:
            total, size = 0, 0
            for point in self.object:
                if not np.isnan(image[point[1], point[0]]):
                    total += image[point[1], point[0]]
                    size += 1
            if size != 0:
                self.depth = total / size
            else:
                self.depth = np.nan

    def _laser_callback(self, msg):
        if self._fsm == fsm.EXPLORING and not self.close_obstacle:
            min_index = int(abs(self.min_laser_angle) / msg.angle_increment)
            max_index = int(abs(self.max_laser_angle) / msg.angle_increment)
            ranges_in_view = msg.ranges[:min_index] + msg.ranges[-max_index:]
            # get the minimum distance
            closest_distance = min(ranges_in_view)
            if closest_distance < self.laser_threshold:
                self.close_obstacle = True

################################################################################
################################################################################

    def get_object(self):
        """
        Find the biggest region that has not been marked yet
        """
        self.regions.sort(reverse=True)   # sort by length with largest at front
        for region in self.regions:
            # calculate position
            if len(region) > self.min_region_size:
                x = self.calculate_depth(region)
                # follow region if depth is unknown
                if np.isnan(x):
                    return region
                cam_p = np.array([self.calculate_depth(region), 0, 0, 1])
                t, r = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
                o_T_c = tf.transformations.translation_matrix(t)
                o_R_c = tf.transformations.quaternion_matrix(r)
                o_H_c = o_T_c.dot(o_R_c)
                odom_p = o_H_c.dot(np.transpose(cam_p))
                # if position not close to collection site
                difference = np.array([odom_p[1] - RETURN_LOCATION[1], odom_p[0], RETURN_LOCATION[0]])
                # if the region is outside of the collection site
                if np.linalg.norm(difference) > self.marker_threshold:
                    return region
################# FOR MARKING THE OBJECTS ######################################
                # new = True
                # for marker in self.marked:
                #     # find difference from marker
                #     difference = np.array([odom_p[1] - marker[1], odom_p[0] - marker[0]])
                #     # if that difference is less than the threshold then it is not new
                #     if np.linalg.norm(difference) < self.marker_threshold:
                #         new = False
                #         break
                # # if this is a new point then return the region
                # if new:
                #     return region
################################################################################
            # if none of the regions are new 
        return None

    def is_similar_color(self,color): 
        """Determines if two colors are similar given the threshold"""
        # Calculate the total rgb difference
        total_difference = abs(color[0] - self.object_color[0]) + abs(color[1] - self.object_color[1]) + abs(color[2] - self.object_color[2])
        # If the total difference is greater than the threshold return false
        # otherwise return true
        return not total_difference > self.max_color_diff

    def calculate_pixel_error(self):
        """
        Calculate the error of the average object distance from the center of 
        the image
        """
        if self.object is None:
            return np.nan
        total_x = 0
        for point in self.object:
            total_x += point[0]
        avg_x = total_x / len(self.object)

        return self.image_center - avg_x

    def calculate_depth(self, obj):
        """
        Calculate the depth of an object based on its camera point cluster
        """
        if obj is not None and self.depth_image is not None and len(obj) != 0:
            total, size = 0, 0
            for point in obj:
                if not np.isnan(self.depth_image[point[1], point[0]]):
                    total += self.depth_image[point[1], point[0]]
                    size += 1
            if size != 0:
                return total / size
            else:
                return np.nan
        return np.nan

    def calculate_controller_error(self, error):
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

    def random_walk(self):
        """
        Random walk exploration when in explore mode
        """
        if not self.close_obstacle:
            self.move(self.linear_velocity, 0)
        else:
            target_yaw = (np.random.uniform(-np.pi, -np.pi/2.) 
                          if np.random.random() > 0.5 else np.random.uniform(np.pi/2., np.pi))
            # set direction to ccw if yaw is positive and cw if negative
            direction = np.sign(target_yaw)
            duration = abs(target_yaw) / self.max_angular_velocity   # duration of rotation
            start_time = rospy.get_rostime()                # get start time
            # loop that turns the robot
            while not rospy.is_shutdown():
                self.move(0, direction * self.max_angular_velocity)
                self.error = self.calculate_pixel_error()
                if rospy.get_rostime() - start_time >= rospy.Duration(duration) or not np.isnan(self.error):
                    self.close_obstacle = False
                    self.angular_velocity = 0
                    break
                self.rate.sleep()

    def mark_object(self, depth):
        # get the camera point and transform into odom
        # simplification using the depth as the x-coordinate
        cam_p = np.array([depth, 0, 0, 1])
        t, r = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
        o_T_c = tf.transformations.translation_matrix(t)
        o_R_c = tf.transformations.quaternion_matrix(r)
        o_H_c = o_T_c.dot(o_R_c)
        odom_p = o_H_c.dot(np.transpose(cam_p))
        print("MARKED: {}".format(odom_p))
        # append odom point to the marked list
        self.marked.append(odom_p[0:2])

    def update_state(self):
        self.object = self.get_object()
        self.error = self.calculate_pixel_error()
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
                self.depth = self.calculate_depth(self.object)
                if self.depth < self.depth_threshold:
                    self.stop()
                    self._fsm = fsm.MARKING
                    # self.mark_object(self.depth)          # for marking the objects in the map
                else:
                    self.calculate_controller_error(self.error)

    def move_rel(self, distance):
        duration = abs(distance) / self.linear_velocity
        direction = np.sign(distance)
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            self.move(direction * self.linear_velocity, 0)
            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                self.stop()
                break
            self.rate.sleep()

    def rotate_rel(self, angle):
        duration = abs(angle) / self.max_angular_velocity
        print(duration)
        direction = np.sign(angle)
        print(direction)
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            print(rospy.get_rostime())
            self.move(0, direction * self.max_angular_velocity)
            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                self.stop()
                break
            self.rate.sleep()

    def return_object(self):
        # go forward 20 cm then rotate and return to the designated trash area
        self.move_rel(self.depth_threshold)

        # Turn toward goal
        curr_t, curr_r = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
        curr_yaw = tf.transformations.euler_from_quaternion(curr_r)[2]
        print(curr_yaw)
        diff = np.array([RETURN_LOCATION[0] - curr_t[0], RETURN_LOCATION[1] - curr_t[1]])
        print(diff)
        final_yaw = np.arctan2(diff[1], diff[0])
        print(final_yaw)
        angle = final_yaw - curr_yaw
        print(angle)
        self.rotate_rel(np.pi)
        
        # take object back to goal
        distance_return = np.linalg.norm(diff)
        self.move_rel(distance_return)

        # reverse from object
        self.move_rel(self.depth_threshold)
        
        # turn back random angle and explore
        random_angle = (np.random.uniform(-np.pi, -np.pi/2., 1) if np.random.random() > 0.5 
                        else np.random.uniform(np.pi/2., np.pi,))
        self.rotate_rel(random_angle)
        self._fsm = fsm.EXPLORING

    def spin(self):
        while not rospy.is_shutdown():
            self.update_state()
            if self._fsm == fsm.FOLLOWING:
                self.move(self.linear_velocity, self.angular_velocity)
            elif self._fsm == fsm.EXPLORING:
                self.random_walk()
            elif self._fsm == fsm.MARKING:
                self.return_object()
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