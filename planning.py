#!/usr/bin/env python

# Main Contributors: Will Balkan and Paige Harris
# With assistance from Kent Koehler and Jackson Desmond

import numpy as np

import rospy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

import tf

DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'scan' # 'scan' for robot 'base_scan' for simulation
DEFAULT_MAP_TOPIC = '/map'
ROBOT_FRAME = "/base_link" 
LASER_FRAME = "laser" #'laser' for gazebo, 'base_link' should work for stage

FRONT_ANGLE = 0
LEFT_ANGLE = np.pi/4

# -pi/4 to pi/4 when laser isn't backwards (in sim)
# -5pi/4 to -3pi/4 when laser is backwards (in gazebo)
MIN_SCAN_ANGLE_RAD = -np.pi*5.0/4.0 
MAX_SCAN_ANGLE_RAD = -np.pi*3.0/4.0

FREQUENCY = 10

LINEAR_VEL = 0.2
ANGULAR_VEL = np.pi/4

WIDTH = 300
HEIGHT = 300
RESOLUTION = 0.05 #m/cell
ORIGIN_X = -5
ORIGIN_Y = -5

INFINITY = 1000000000 #estimate of infinity

EMPTY = 0
OBSTACLE = 100

MAP_FRAME = 'map'

TRASH_CAN_X = 0
TRASH_CAN_Y = 0


# node class for A* search
class Node:
    def __init__(self, point, dist, h, parent):
        self.point = point
        self.dist = dist
        self.h = h
        self.f = self.dist + h
        self.parent = parent

# map class where an OccupancyGrid is created and filled out
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

    # set a cell at (x, y) in the map to a given value
    def set_cell(self, x, y, val):
        if y < 0 or y > self.grid_msg.info.height or x < 0 or x > self.grid_msg.info.width:
            return
        index = (self.grid_msg.info.width*y) + x
        if self.grid_msg.data[index] == -1 or self.grid_msg.data[index] == EMPTY:
            self.grid_msg.data[index] = val

    # get cell from 2 points in fixed frame (output in cells)
    def get_cell(self, pos_x, pos_y):
        grid_x = int( (pos_x - self.grid_msg.info.origin.position.x) / RESOLUTION )
        grid_y = int( (pos_y - self.grid_msg.info.origin.position.y) / RESOLUTION )
        return grid_x, grid_y
    
    # return value of map cell at (x, y)
    def cell_at(self, x, y):
        index = int( (self.grid_msg.info.width*y) + x )
        return self.grid_msg.data[index]
    
    # return a list of indices on the frontier for exploration
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



# main node for exploration and mapping
class Mapper:
    def __init__(self):
    
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC, Twist, queue_size=1)

        # Setting up laser subscriber
        self._laser_sub = rospy.Subscriber(DEFAULT_SCAN_TOPIC, LaserScan, self.laser_callback, queue_size=1)

        # Setting up tf listener
        self.listener = tf.TransformListener()

        self.map = Map(WIDTH, HEIGHT, RESOLUTION)

        self._map_pub = rospy.Publisher(DEFAULT_MAP_TOPIC, OccupancyGrid, queue_size=1)
        self._map_pub.publish(self.map.grid_msg)

        self.updating = False
        self.laser_msg = None

        self.trash_can_x = TRASH_CAN_X
        self.trash_can_y = TRASH_CAN_Y



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

        self.map.set_cell(obs_x, obs_y, OBSTACLE)
        ## dilate obstacle
        for x in range(-5, 6):
            for y in range(-5, 6):
                self.map.set_cell(obs_x + x, obs_y + y, OBSTACLE)


    # store laser message
    def laser_callback(self, msg):
        self.laser_msg = msg


    # using laser sensor data, update map with empty cells and obstacles
    def update_map(self):
        self.updating = True

        # check for laser mesasge
        if self.laser_msg == None:
            self.updating = False
            print('no laser message')
            return

        msg = self.laser_msg

        # get location of laser sensor
        pos, quat = self.get_Laser_Loc()
        rob_x, rob_y = self.map.get_cell( pos[0], pos[1] )
        euler = tf.transformations.euler_from_quaternion(quat)
        yaw = euler[2]

        # only map in a reduced range so that only areas that the camera can see are considered explored
        index_min = int((MIN_SCAN_ANGLE_RAD - msg.angle_min) / msg.angle_increment)
        index_max = int((MAX_SCAN_ANGLE_RAD - msg.angle_min) / msg.angle_increment)

        # for angles within the range to be checked
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
    
    # get position of laser with respect to the odom reference frame
    def get_Laser_Loc(self):
        frame = self.map.grid_msg.header.frame_id
        self.listener.waitForTransform(frame, LASER_FRAME, rospy.Time(), rospy.Duration(4.0))
        position, quaternion = self.listener.lookupTransform(frame, LASER_FRAME, rospy.Time(0))     

        return position, quaternion

    # get real world location of a map cell given its coordinates
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

        # spin for calculated duration
        duration = rospy.Duration(time)
        start_time = rospy.get_rostime()
        while rospy.get_rostime() - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rate.sleep()

        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)


    # rotate to a rotation of angle radians
    def rotate_abs(self, angle):
        _ , quaternion = self.get_Loc()
        euler = tf.transformations.euler_from_quaternion(quaternion)
        curr_angle = euler[2]

        angle_diff = angle - curr_angle

        self.rotate_rel(angle_diff)

    # stop the robot's movement
    def stop(self):
        stop_msg = Twist()
        self._cmd_pub.publish(stop_msg)


    # find a path between a start and goal location in the occupancy grid using A* search with memory
    def find_path(self, start, goal):
        frontier = []
        visited = []
        h = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
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

                # search all neighboring points
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
    

    # given a path through the occupancy grid, convert to a path in real world coordinates
    def path_to_map(self, path):
        map_path = []
        if path != None:
            for i in range(len(path)):
                map_path.append( self.m_from_cell(path[i].point[0], path[i].point[1]) )
        return map_path
    

    # move along a line connecting a series of points
    def polyline(self, points):
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
            
    
    # main function for exploring the environment, consists of:
    # 1: getting a random location on the frontier
    # 2: finding a path to this location
    # 3: moving the robot along this path to the new location on the frontier
    def explore(self):
        self.update_map()
        frontier = self.map.get_frontier()

        # get a random location on the frontier
        if len(frontier) == 0:
            rospy.sleep(2)
            print('no frontier')
            return
        random_cell = int( np.random.rand() * len(frontier) )
        target_x = frontier[random_cell] % self.map.grid_msg.info.width
        target_y = (frontier[random_cell] - target_x) / self.map.grid_msg.info.width

        # find a path to the frontier location
        pos, _ = self.get_Loc()
        print(self.map.get_cell(pos[0], pos[1]), [target_x, target_y])
        path = self.find_path(self.map.get_cell(pos[0], pos[1]), [target_x, target_y])
        map_path = self.path_to_map(path)
        print(path)
        print(map_path)

        # move to the frontier
        self.polyline(map_path)
        rospy.sleep(2)


def main():
    rospy.init_node("mapper")
    robot = Mapper()
    rospy.sleep(2)

    # stop robot movement on shutdown
    rospy.on_shutdown(robot.stop)

    # robot explores environment
    while not rospy.is_shutdown():
        robot.explore()


    rospy.spin()

    

if __name__ == "__main__":
    
    main()
