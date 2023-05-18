# Trash Collector Robot

### Launching the simulator

To launch the simulator, first download the required ROSBOT packages from github

```bash
cd {workspace}/src
git clone -b noetic https://github.com/husarion/rosbot_ros.git
```

Install dependencies:

```bash
cd {workspace}
rosdep install --from-paths src --ignore-src -r -y
```

Build workspace:

```bash
cd {workspace}
catkin_make
```

Load system variables:

```bash
source {workspace}/devel/setup.sh
```

Launch the simulation

```bash
cd {workspace}/src/trash_collector/launch
export GAZEBO_RESOURCE_PATH={workspace}/src/trash_collector/world
rosluanch green_boxes.launch
```