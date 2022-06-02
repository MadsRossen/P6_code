# AAU ROB6 Bachelor Project | Bin Picking


## Overview:
[![License](https://img.shields.io/github/license/MadsRossen/P6_project)](https://github.com/MadsRossen/P6_project)
[![Github code size](https://img.shields.io/github/languages/code-size/MadsRossen/P6_project)](https://github.com/MadsRossen/P6_project)
[![Github issue tracker](https://img.shields.io/github/issues/MadsRossen/P6_project)](https://github.com/MadsRossen/P6_project)
[![Commit activity](https://img.shields.io/github/commit-activity/w/MadsRossen/P6_project)](https://github.com/MadsRossen/P6_project)

## Table of content
[Where to find things](#where-to-find-things)<br/>
* [ROS config](#ros)<br/>
* [Launch files](#launch-files)<br/>
* [Image data](#image-data)<br/>

[Issues](#issues)<br/>
[Tasks](#tasks)<br/>
[Installation](#installation)<br/>
[Running the code](#running-the-code)<br/>
[Troubleshooting](#troubleshooting)<br/>

## Where to find things:
### ROS config
[joint_limits](https://github.com/MadsRossen/P6_project/blob/main/src/fmauch_universal_robot/ur_description/config/ur5/joint_limits.yaml)<br/>
[tool_urdf](https://github.com/MadsRossen/P6_project/blob/main/src/fmauch_universal_robot/ur_description/urdf/inc/tool.xacro)<br/>
[ur5_w_tool_urdf](https://github.com/MadsRossen/P6_project/blob/main/src/fmauch_universal_robot/ur_description/urdf/ur5_robot_w_tool.urdf.xacro)<br/>
[meshes](https://github.com/MadsRossen/P6_project/tree/main/src/fmauch_universal_robot/ur_description/meshes/ur5)<br/>
### Launch files
Launch setup assistant
```bash
roslaunch ur5_moveit_config_beumer setup_assistant.launch
```
Connect to real robot 
```bash
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=<ip_of_robot>
```
See real robot in RViz
```bash
roslaunch ur_robot_driver example_rviz.launch 
```

### Image data
[Training](https://github.com/MadsRossen/P6_project/tree/main/images_of_parcels/training)<br/>
[Validation](https://github.com/MadsRossen/P6_project/tree/main/images_of_parcels/validation)<br/>
[Test](https://github.com/MadsRossen/P6_project/tree/main/images_of_parcels/test)<br/>
## Issues

```$ catkin_make``` does not work, use ```$ catkin_make_isolated``` insted

## Tasks 

This is a project in collaboration with BEUMER Group, to suggest a bin_picking algorithm.

- [x] install tenserflow
- [x] intall mask rcnn
- [ ] check transformations from robot base to parcel is correct
- [ ] Define dropoff place

 Tasks are complete :tada:


## Installation
```bash
git clone https://github.com/MadsRossen/P6_project

cd P6_project
source devel/setup.bash

# or 
    nano ~/bashrc
    # On new line:
    cd ~/{Path_To_catkin_WS}/P6_project
    source devel/setup.bash 
catkin_make
catkin_make_isolated
```

## Running the code
Robot running
```bash
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=172.22.22.3
```
Go to the UR5 teachpendent and run the ext2660.urdp file. 
```bash
roslaunch robot_prototype robot_prototype.launch
```
This will launch the planning, Rviz and and main code of the robot

## Troubleshooting

### The robot will not move
- check that it has the corrct IP in teachpendent -> Setup robot -> Network -> IP address 172.22.22.5
- Host IP on teachpendent -> Program robot -> Load Program -> ext2660.urp -> Installation -> External control Host IP: 172.22.22.3 
- Wifi should be disabled 
- Wired connection is enabled 
- Wired settings IPv4 IP is set to: 172.22.22.3
- Check that the firewall is disabled for ubuntu running: 
```bash
sudo ufw status
```
else 
```bash
sudo ufw disable
```


## Training an inference model - using Mask R-CNN and CLAAUDIA



## Geting the Mask R-CNN detector up and running

### Install dependencies / packages

To get going with the deteqctor, we first need the following packages, that must be installed.

Version control is critical for cross-functionality.
```
python3 -m pip install tensorflow==1.15.0
python3 -m pip install keras==2.2.5
python3 -m pip install h5py==2.10.0
```
