#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
import copy
from numpy.core.numeric import Infinity
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

from ...common import utilities as util
from ...common.settings import ENABLE_BACKWARD, EPISODE_TIMEOUT_SECONDS, ENABLE_MOTOR_NOISE


NUM_SCAN_SAMPLES = util.get_scan_count()
LINEAR = 0
ANGULAR = 1
ENABLE_DYNAMIC_GOALS = False

ACTION_LINEAR_MAX   = 0.22  # in m/s
ACTION_ANGULAR_MAX  = 2.0   # in rad/s

# in meters
ROBOT_MAX_LIDAR_VALUE   = 16
MAX_LIDAR_VALUE         = 3.5

MINIMUM_COLLISION_DISTANCE  = 0.13
MINIMUM_GOAL_DISTANCE       = 0.20
OBSTACLE_RADIUS             = 0.16
MAX_NUMBER_OBSTACLES        = 6

ARENA_LENGTH    = 4.2
ARENA_WIDTH     = 4.2
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        
        self.scan_ranges = [MAX_LIDAR_VALUE] * NUM_SCAN_SAMPLES
        self.obstacle_distance = MAX_LIDAR_VALUE
        self.obstacle_distances = [Infinity] * MAX_NUMBER_OBSTACLES


        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_scan =  rospy.Subscriber('scan', LaserScan, self.scan_callback)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        diff_y = self.goal_y - self.position.y
        diff_x = self.goal_x - self.position.x
        goal_distance = math.sqrt(diff_x**2 + diff_y**2)
        goal_angle = math.atan2(diff_y, diff_x)
        heading = goal_angle - yaw

        while heading > math.pi:
            heading -= 2 * math.pi
        while heading < -math.pi:
            heading += 2 * math.pi

        self.heading = round(heading, 2)
        self.distance_to_goal = round(goal_distance, 2)

    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # noramlize laser values
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = np.clip(float(msg.ranges[i]) / MAX_LIDAR_VALUE, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= MAX_LIDAR_VALUE

    def getState(self, action_linear_previous, action_angular_previous):
        scan_range = []
        min_range = 0.13
        done = False

        state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state.append(float(np.clip((self.distance_to_goal / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.heading) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
    
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return state, done

    def setReward(self, done, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [0, -3.14]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * self.goal_distance ) / (self.goal_distance  + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if done:
            rospy.loginfo("Collision!!")
            reward -= 2000
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward += 2500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return float(reward)

    def step(self, action, action_past):
        if ENABLE_MOTOR_NOISE:
            action[LINEAR] += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
            action[ANGULAR] += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)

        # Un-normalize actions
        if ENABLE_BACKWARD:
            action_linear = action[LINEAR] * ACTION_LINEAR_MAX
        else:
            action_linear = (action[LINEAR] + 1) / 2 * ACTION_LINEAR_MAX
        action_angular = action[ANGULAR]*ACTION_ANGULAR_MAX

        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.pub_cmd_vel.publish(twist)

        state, done = self.getState(action_past[LINEAR], action_past[ANGULAR])
        reward = self.setReward(done, action_linear, action_angular, self.distance_to_goal, self.heading, self.obstacle_distance )

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(0.0, 0.0)

        return np.asarray(state)