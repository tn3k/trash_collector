#!/usr/env/bin python

# Author: Kent Koehler
# Date: 5/21/2023

import numpy as np
import tf
import cv2 as cv
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Bool, Float64

from nav_msgs.msg import OccupancyGrid


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

# to determine if a position is within another radius
MARKER_THRESHOLD = 0.5

# Topics and services
CMD_VEL_TOPIC = 'cmd_vel'
SCAN_TOPIC = 'scan'
CAMERA_COLOR_TOPIC = 'camera/color/image_raw' # camera/rgb/image_raw for ROSBOT 2
CAMERA_DEPTH_TOPIC = 'camera/depth/image_raw'

# parameters for object detection
MAX_COLOR_DIFF = 50
MIN_REGION_SIZE = 5

OBJECT_COLOR = (0, 100, 0)


# exploration constants:
ORIGIN_X = -5
ORIGIN_Y = -5

WIDTH = 300
HEIGHT = 300
RESOLUTION = 0.05 #m/cell


# -pi/4 to pi/4 when laser isn't backwards
# -5pi/4 to -3pi/4 when laser is backwards
MIN_SCAN_ANGLE_RAD = -np.pi*5.0/4.0 
MAX_SCAN_ANGLE_RAD = -np.pi*3.0/4.0

FREQUENCY = 10

LINEAR_VEL = 0.2
ANGULAR_VEL = np.pi/4

EMPTY = 0
OBSTACLE = 100
MAP_FRAME = 'map'

INFINITY = 1000000000

TRASH_CAN_X = 0
TRASH_CAN_Y = 0

# exploration topic names:
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'scan' # 'scan' for robot 'base_scan' for simulation
DEFAULT_MAP_TOPIC = '/map'
ROBOT_FRAME = "/base_link" #"/base_link" for sim,
LASER_FRAME = "laser" #'laser' for gazebo, 'base_link' should work for sim




# fsm for determining action taken
class fsm(Enum):
    EXPLORING = 0
    FOLLOWING = 1
    MARKING = 2




# node class for A* search
class Node:
    def __init__(self, point, dist, h, parent):
        self.point = point
        self.dist = dist
        self.h = h
        self.f = self.dist + h
        self.parent = parent




# occupancy grid for mapping class

class Map:
    def __init__(self, width, height, resolution):
        self.grid = np.ones( height* width ) * (-1)

        self.grid_msg = OccupancyGrid()

        self.grid_msg.header.stamp = rospy.get_rostime()
        self.grid_msg.header.frame_id = '/odom'

        self.grid_msg.data = self.grid
        self.grid_msg.info.resolution = resolution
        self.grid_msg.info.width = width
        self.grid_msg.info.height = height
        self.grid_msg.info.origin.position.x = ORIGIN_X
        self.grid_msg.info.origin.position.y = ORIGIN_Y


    def set_cell(self, x, y, val):
        
        if y < 0 or y > self.grid_msg.info.height or x < 0 or x > self.grid_msg.info.width:
            return
        # print("setting ", x, y, "to", val)
        index = (self.grid_msg.info.width*y) + x
        # print('index', index)
        if self.grid_msg.data[index] == -1 or self.grid_msg.data[index] == EMPTY:
            self.grid_msg.data[index] = val

    # get cell from 2 points in fixed frame (output in cells)
    def get_cell(self, pos_x, pos_y):
        grid_x = int( (pos_x - self.grid_msg.info.origin.position.x) / RESOLUTION )
        grid_y = int( (pos_y - self.grid_msg.info.origin.position.y) / RESOLUTION )
        return grid_x, grid_y
    
    def cell_at(self, x, y):
        index = int( (self.grid_msg.info.width*y) + x )
        return self.grid_msg.data[index]
    
    def get_frontier(self):
        width = self.grid_msg.info.width
        frontier = []
        for i in range(len(self.grid_msg.data)):
            border_indices = [ i - width, i + width, i - 1, i + 1 ]
            if self.grid_msg.data[i] == EMPTY:
                disqualified = False
                in_frontier = False
                for index in border_indices:
                    if index < 0 or index > (len(self.grid_msg.data)):
                        disqualified = True
                        
                    if self.grid_msg.data[index] == -1:
                        in_frontier = True
                    if self.grid_msg.data[index] == 100:
                        disqualified = True
                        
                if in_frontier and not disqualified:
                    frontier.append(i)
        return frontier # list of indices on frontier not bordering obstacles
    















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


        # exploration initialization:
        self.map = Map(WIDTH, HEIGHT, RESOLUTION)
        self._map_pub = rospy.Publisher(DEFAULT_MAP_TOPIC, OccupancyGrid, queue_size=1)
        self._map_pub.publish(self.map.grid_msg)

        self.updating = False
        self.laser_msg = None



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

        # store message for mapping:
        self.laser_msg = msg

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
                print("camera: {}".format(cam_p))
                t, r = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
                o_T_c = tf.transformations.translation_matrix(t)
                o_R_c = tf.transformations.quaternion_matrix(r)
                o_H_c = o_R_c.dot(o_T_c)
                odom_p = o_H_c.dot(np.transpose(cam_p))
                # if position not close to one in marked then return
                new = True
                for marker in self.marked:
                    # find difference from marker
                    difference = np.array([odom_p[1] - marker[1], odom_p[0] - marker[0]])
                    # if that difference is less than the threshold then it is not new
                    if np.linalg.norm(difference) < self.marker_threshold:
                        new = False
                # if this is a new point then return the region
                if new:
                    return region
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
                    
                else:
                    self.calculate_controller_error(self.error)

    def spin(self):
        while not rospy.is_shutdown():
            self.update_state()
            if self._fsm == fsm.FOLLOWING:
                self.move(self.linear_velocity, self.angular_velocity)
            elif self._fsm == fsm.EXPLORING:
                # self.random_walk()
                self.explore()
            elif self._fsm == fsm.MARKING:
                # self.mark_object()
                pos, _ = self.get_Loc()
                path = self.find_path( self.map.get_cell( pos.x, pos.y ), self.map.get_cell( TRASH_CAN_X, TRASH_CAN_Y ) )
                map_path = self.path_to_map(path)
                self.polyline(map_path, False)
                self.translate(-0.3)
                self._fsm = fsm.EXPLORING
            self.rate.sleep()


#######################################################################################
# Exploration Functions
#######################################################################################


# trace from robot to obstacle, filling in free spaces
    def ray_trace(self, robot_x, robot_y, obs_x, obs_y):
        
        x_diff  = obs_x - robot_x
        y_diff = obs_y - robot_y
        if x_diff != 0:
            slope = float(y_diff) / float(x_diff)
        else:
            slope = INFINITY # set

        if abs(x_diff) > abs(y_diff):
            
            if obs_x > robot_x:
                min_x = robot_x
                max_x = obs_x
                y1 = robot_y
            else:
                min_x = obs_x
                max_x = robot_x
                y1 = obs_y

            for x in range(min_x, max_x):
                real_y = y1 + ((x-min_x) * slope)
                y = int(real_y)
                self.map.set_cell(x, y, 0)

        else:
            if obs_y > robot_y:
                min_y = robot_y
                max_y = obs_y
                x1 = robot_x
            else:
                min_y = obs_y
                max_y = robot_y
                x1 = obs_x

            for y in range(min_y, max_y):
                real_x = x1 + ((y-min_y) * (1/slope))
                x = int(real_x)
                self.map.set_cell(x, y, EMPTY)

    # update map being drawn
    def update_map(self):
        self.updating = True

        if self.laser_msg == None:
            self.updating = False
            print('no laser message')
            return

        msg = self.laser_msg

        pos, quat = self.get_Laser_Loc()
        rob_x, rob_y = self.map.get_cell( pos[0], pos[1] )
        euler = tf.transformations.euler_from_quaternion(quat)
        yaw = euler[2]

        index_min = int((MIN_SCAN_ANGLE_RAD - msg.angle_min) / msg.angle_increment)
        index_max = int((MAX_SCAN_ANGLE_RAD - msg.angle_min) / msg.angle_increment)


        for i in range(index_min, index_max):
            if msg.ranges[i] > msg.range_max:
                continue
            angle = (i*msg.angle_increment) + msg.angle_min
            theta = angle + yaw
            
            obs_x = (np.cos(theta) * msg.ranges[i]) + pos[0]
            obs_y = (np.sin(theta) * msg.ranges[i]) + pos[1]
            obs_grid_x, obs_grid_y = self.map.get_cell(obs_x, obs_y)
            self.ray_trace(rob_x, rob_y, obs_grid_x, obs_grid_y)
        self._map_pub.publish(self.map.grid_msg)
        self.updating = False


    # helper function to return the position and rotation of the robot with respect to the odom reference frame
    def get_Loc(self):
        frame = self.map.grid_msg.header.frame_id
        self.listener.waitForTransform(frame, ROBOT_FRAME, rospy.Time(), rospy.Duration(4.0))
        position, quaternion = self.listener.lookupTransform(frame, ROBOT_FRAME, rospy.Time(0))     

        return position, quaternion
    
    # get position and rotation of laser reference frame
    def get_Laser_Loc(self):
        frame = self.map.grid_msg.header.frame_id
        self.listener.waitForTransform(frame, LASER_FRAME, rospy.Time(), rospy.Duration(4.0))
        position, quaternion = self.listener.lookupTransform(frame, LASER_FRAME, rospy.Time(0))     

        return position, quaternion
    

    # input: cells, output: meters
    def m_from_cell(self, x, y):
        map_x = (x * self.map.grid_msg.info.resolution + self.map.grid_msg.info.origin.position.x)
        map_y = (y * self.map.grid_msg.info.resolution + self.map.grid_msg.info.origin.position.y)
        return [ map_x, map_y]


    # move d meters forward
    def translate(self, d):
       
        rate = rospy.Rate(FREQUENCY)

        # Setting velocities.
        twist_msg = Twist()
        twist_msg.linear.x = LINEAR_VEL
        if d < 0:
            twist_msg.linear.x = -LINEAR_VEL

        time = abs( d / LINEAR_VEL )

        duration = rospy.Duration(time)
        start_time = rospy.get_rostime()
        while rospy.get_rostime() - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rate.sleep()

        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)


    # rotate angle radians
    def rotate_rel(self, angle):
        
        rate = rospy.Rate(FREQUENCY)

        # Setting velocities.
        twist_msg = Twist()
        twist_msg.angular.z = ANGULAR_VEL

        if angle < 0:
            twist_msg.angular.z = - ANGULAR_VEL

        time = abs(angle) / ANGULAR_VEL

        duration = rospy.Duration(time)
        start_time = rospy.get_rostime()
        while rospy.get_rostime() - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rate.sleep()

        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)


    # find a path between a start and goal location in the occupancy grid using A* search with memory
    def find_path(self, start, goal):
        frontier = []
        visited = []
        h = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        # h = int(h)
        frontier.append(Node(start, 0, h, None))
        while len(frontier) > 0:
            min_f_index = 0
            for i in range(len(frontier)):
                if frontier[i].f < frontier[min_f_index].f:
                    min_f_index = i
            curr = frontier.pop(min_f_index)
            
            if curr.point not in visited:
                visited.append(curr.point)

                # check if you are at the goal
                if curr.point[0] == goal[0] and curr.point[1] == goal[1]:
                    rev_path = []
                    node = curr
                    while node != None:
                        rev_path.append(node)
                        node = node.parent
                    rev_path.reverse()
                    
                    return rev_path

                # check all neighboring points
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            point = [curr.point[0] + dx, curr.point[1] + dy]
                            if (0 <= point[0] <= self.map.grid_msg.info.width) and (0 <= point[1] <= self.map.grid_msg.info.height):
                                h = np.sqrt((goal[0] - point[0])**2 + (goal[1] - point[1])**2)
                                # h = int(h)
                                step = np.sqrt(dx*dx + dy*dy)
                                new_child = Node(point, curr.dist + step, h, curr)
                                clear = True
                                # for dx2 in range(-5, 6):
                                #     for dy2 in range(-5,6):
                                if self.map.cell_at(point[0], point[1]) != 0:
                                    clear = False
                                if clear:
                                    if new_child.point not in visited:
                                        added = False
                                        for i in range(len(frontier)):
                                            if frontier[i].point == new_child.point:
                                                if frontier[i].f > new_child.f:
                                                    frontier[i] = new_child
                                                    added = True
                                        if not added:
                                            frontier.append(new_child)                    
        print('didnt find')
        return None
    

    def path_to_map(self, path):
        map_path = []
        if path != None:
            for i in range(len(path)):
                map_path.append( self.m_from_cell(path[i].point[0], path[i].point[1]) )
        return map_path
    

    # draw a polygon connecting a series of points
    def polyline(self, points, explore):

        # rotate toward and move to each point in the list
        for i in range(len(points)):
            position, quaternion = self.get_Loc()
            curr_x = position[0]
            curr_y = position[1]
            x_diff = points[i][0] - curr_x
            y_diff = points[i][1] - curr_y

            target_theta = np.arctan2(y_diff, x_diff)

            euler = tf.transformations.euler_from_quaternion(quaternion)
            curr_theta = euler[2]


            theta = target_theta - curr_theta
            
            dist = np.sqrt(x_diff*x_diff + y_diff*y_diff)

            while abs(theta) > np.pi:
                if theta < 0:
                    theta += 2*np.pi
                else:
                    theta -= 2*np.pi
            
            self.rotate_rel(theta)

            self.translate(dist)
            self.update_map()
            self.update_state()
            print(self._fsm)
            if explore and self._fsm != fsm.EXPLORING:
                return
            
            
    # explore map
    def explore(self):
        self.update_map()
        frontier = self.map.get_frontier()
        if len(frontier) == 0:
            rospy.sleep(2)
            print('no frontier')
            return
        random_cell = int( np.random.rand() * len(frontier) )
        target_x = frontier[random_cell] % self.map.grid_msg.info.width
        target_y = (frontier[random_cell] - target_x) / self.map.grid_msg.info.width

        pos, _ = self.get_Loc()
        print(self.map.get_cell(pos[0], pos[1]), [target_x, target_y])
        path = self.find_path(self.map.get_cell(pos[0], pos[1]), [target_x, target_y])
        map_path = self.path_to_map(path)
        print(path)
        print(map_path)
        self.polyline(map_path, True)
        rospy.sleep(2)




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
