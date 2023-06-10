# Trash Collector Robot

Jack Desmond, Paige Harris, Will Balkan, and Kent Koehler
Dartmouth College, COSC 81, Spring 2023

### Description

#### Primary Code

`marker.py` is the program tested in simulation for which the data in the report is representative of. The program implements color-based object detection and visual servoing with a PD controller as well as object localization based on image data.

#### Secondary Code

`planning.py` implements autonomous frontier-based exploration and mapping as well as an A* pathfinding algorithm.

`cleanup.py` is a theoretical combination of `marker.py` with the real ROSbot 2 that would collect the object in a containment device and return it to the origin.

### Simulation

In the directions `/world` and `/launch` are world files and launch files that were developed for the project and used in simulation testing. The launch files require packages for simulating the ROSbot 2 which can be found [here](https://github.com/husarion/rosbot_ros). Running the launch files requires setting `GAZEBO_RESOURCE_PATH`.