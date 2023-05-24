#!/usr/bin/env python

# Author: Jack Desmond
# Date: 05/16/23

# Import of python modules.
import numpy as np
from cv_bridge import CvBridge
import cv2 as cv

# import of relevant libraries.
import rospy # module for ROS APIs
from sensor_msgs.msg import LaserScan # message type for scan
from sensor_msgs.msg import Image # message type for images
from std_msgs.msg import Float64

# Constants.

# Frequency at which the loop operates
FREQUENCY = 10 #Hz.

# Topic names
CAMERA_COLOR_TOPIC = '/camera/color/image_raw'

MAX_COLOR_DIFF = 50
MIN_REGION_SIZE = 10

OBJECT_COLOR = (0,100,0)

class ObjectDetection():
    def __init__(self, maxColorDiff=MAX_COLOR_DIFF, minRegionSize=MIN_REGION_SIZE, objectColor=OBJECT_COLOR):
        """Constructor."""

        # Setting up publishers/subscribers.
        # Setting up subscriber receiving messages
        self._image_sub = rospy.Subscriber(CAMERA_COLOR_TOPIC, Image, self.image_callback, queue_size=1)
        self.error_pub = rospy.Publisher('error', Float64, queue_size=1)
        # Set up parameters for region determination
        self.maxColorDiff = maxColorDiff
        self.minRegionSize = minRegionSize
        self.objectColor = objectColor
        # Initialize cvbridge
        self.bridge = CvBridge()

        self.error = None
    
    def image_callback(self, msg):
        """Image callback function"""
        # Call find regions function
        self.findRegions(msg)

    def findRegions(self, msg):
        """Finds the regions of matching color"""
        # Initialize
        visited = set()
        regions = []
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv.pyrDown(raw_image)
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
                    if len(newRegion) >= self.minRegionSize:
                        regions.append(newRegion)
        # calculate the largest region    
        largestRegion = self.largestRegion(regions)
        # calculate image error
        self.error = (self.calculateHorizontalError(largestRegion, width / 2) 
                      if len(largestRegion) > 0 else 0)

    def isSimilarColor(self,color):
        """Determines if two colors are similar given the threshold"""
        # Calculate the total rgb difference
        total_difference = abs(color[0] - self.objectColor[0]) + abs(color[1] - self.objectColor[1]) + abs(color[2] - self.objectColor[2])
        # If the total difference is greater than the threshold return false
        # otherwise return true
        if total_difference > self.maxColorDiff:
            return False
        else:
            return True

    def largestRegion(self, regions):
        """Finds the largest region in a list of regions"""
        # Init largest region variable as empty list
        largest_region = []
        # Loop through the regions and compare lengths to the current largest region
        for region in regions:
            if len(region) > len(largest_region):
                largest_region = region
        return largest_region

    def calculateHorizontalError(self, region, center_x):
        total_x = 0
        for point in region:
            total_x += point[0]
        avg_x = total_x / len(region)

        return avg_x - center_x
    
    def publish_error(self):
        msg = Float64()
        msg.data = float(self.error)
        self.error_pub.publish(msg)
    
    def spin(self):
        rate = rospy.Rate(FREQUENCY) # loop at 10 Hz.
        while not rospy.is_shutdown():
            # if there is a grid object
            if self.error is not None:
                self.publish_error()
            rate.sleep()

       
def main():
    """Main function."""
    # 1st. initialization of node.
    rospy.init_node("object_detection")
    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)
    # Initialization of the class for the object detection.
    object_detection = ObjectDetection()

    try:
        object_detection.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()