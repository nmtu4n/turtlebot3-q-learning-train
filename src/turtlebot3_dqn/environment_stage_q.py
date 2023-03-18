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
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.14

ZONE_0_LENGTH = 0.4
ZONE_1_LENGTH = 0.7

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1- 1
HORIZON_WIDTH = 75

K_RO = 2
K_ALPHA = 15
K_BETA = -3
V_CONST = 0.1

CONST_LINEAR_SPEED_FORWARD  = 0.15
CONST_ANGULAR_SPEED_FORWARD = 0.0
CONST_LINEAR_SPEED_TURN     = 0.1
CONST_ANGULAR_SPEED_TURN    = 0.4

class Env():
    def __init__(self, action_size, state_space):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        
        self.state_space = state_space
        self.prev_action = 0
        self.prev_lidar = []

        self.goal_angle = 0
        self.yaw = 0
        self.diff_rotation_to_end_last =0 
        

    def lidar_scan(self, scan_msg):
        scan_range = np.array([])

        for scan_point in scan_msg.ranges:
            distance = scan_msg.range_min
            
            if (scan_point > MAX_LIDAR_DISTANCE or scan_point == float('Inf')):
                distance = MAX_LIDAR_DISTANCE
            elif (scan_point < scan_msg.range_min or np.isnan(scan_point)):
                distance = scan_msg.range_min
            else:
                distance = scan_point

            scan_range = np.append(scan_range, distance)
        
        return np.asarray(scan_range)
                          

    def scan_discretization(self, state_space, lidar):
        x1 = 2
        x2 = 2
        x3 = 3
        x4 = 3

        lidar_left = min(lidar[ANGLE_MIN: ANGLE_MIN + HORIZON_WIDTH])
        if (ZONE_1_LENGTH > lidar_left > ZONE_0_LENGTH):
            x1 = 1
        elif lidar_left <= ZONE_0_LENGTH:
            x1 = 0

        lidar_right = min(lidar[(ANGLE_MAX - HORIZON_WIDTH): ANGLE_MAX])
        if (ZONE_1_LENGTH > lidar_right > ZONE_0_LENGTH):
            x2 = 1
        elif lidar_right <= ZONE_0_LENGTH:
            x2 = 0
        
        if( 
            min(lidar[(ANGLE_MAX - HORIZON_WIDTH // 3): ANGLE_MAX ]) < 1.0 or 
            min(lidar[ANGLE_MIN : (ANGLE_MIN +  HORIZON_WIDTH // 3)]) < 1.0
        ):
            object_front = True
        else:
            object_front = False

        if( min(lidar[ANGLE_MIN : (ANGLE_MIN + 2 * HORIZON_WIDTH // 3)]) < 1.0 ):
            object_left = True
        else:
            object_left = False

        if( min(lidar[(ANGLE_MAX - 2 * HORIZON_WIDTH // 3): ANGLE_MAX]) < 1.0 ):
            object_right = True
        else:
            object_right = False

        if( min(lidar[(ANGLE_MIN + HORIZON_WIDTH // 3) : (ANGLE_MIN +  HORIZON_WIDTH )]) < 1.0):
            object_far_left = True
        else:
            object_far_left = False 

        if( min(lidar[(ANGLE_MAX - HORIZON_WIDTH) : (ANGLE_MAX - HORIZON_WIDTH // 3)]) < 1.0):
            object_far_right = True
        else:
            object_far_right = False

        if ((object_front and object_left ) and (not object_far_left)):
            x3 = 0
        elif ((object_left and object_far_left) and (not object_front)):
            x3 = 1
        elif object_front and object_left and object_far_left:
            x3 = 2

        if ((object_front and object_right) and (not object_far_right)):
            x4 = 0
        elif (object_right and object_far_right and (not object_front)):
            x4 = 1
        elif object_front and object_right and object_far_right:
            x4 = 2

        if self.heading < 0:
            heading = self.heading + 2* pi
        else:
            heading = self.heading
        
        x5 = int(math.modf(heading/ (pi/6))[1])
        ss = np.where(np.all(state_space == np.array([x1, x2, x3, x4, x5]), axis = 1))
        state_index = int(ss[0])

        return (state_index, x1, x2, x3, x4, x5)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = self.goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
        self.yaw = yaw
        self.goal_distance = self.getGoalDistace()

    def getState(self, scan):
        min_range = 0.13
        done = False

        if min_range > min(scan) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return self.scan_discretization(self.state_space, scan), done

    def setReward(self, done, action, prev_action, lidar, prev_lidar):
        lidar_horizon = np.concatenate(
            (lidar[(ANGLE_MIN + HORIZON_WIDTH) : ANGLE_MIN : -1],
            lidar[(ANGLE_MAX) : (ANGLE_MAX-HORIZON_WIDTH) : -1])
        )

        prev_lidar_horizon = np.concatenate(
            (prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],
            prev_lidar[(ANGLE_MAX): (ANGLE_MAX-HORIZON_WIDTH): -1])
        )

        if (action == 3):
            r_action = 0.2
        else:
            r_action = -0.1

        W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))

        if np.sum(W * (lidar_horizon- prev_lidar_horizon)) >= 0:
            r_obstacle = 0.2
        else:
            r_obstacle = -0.2

        if (prev_action == 1 and action == 2) or (prev_action == 2 and action == 1):
            r_change = -0.8
        else:
            r_change = 0.0

        diff_rotation_to_end = self.heading
        rotation_cos_sum = math.cos(diff_rotation_to_end)
        diff_rotations = math.fabs(math.fabs(self.diff_rotation_to_end_last) - math.fabs(diff_rotation_to_end) ) 

        if math.fabs(diff_rotation_to_end) > math.fabs(self.diff_rotation_to_end_last):
            diff_rotations *= -3.0
        else:
            diff_rotations *= 2.0

        reward = r_obstacle + r_action + r_change + (3*rotation_cos_sum) + diff_rotations - 0.1

        self.diff_rotation_to_end_last = self.heading

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def feed_back_control(self):

        _alpha = (self.goal_angle - self.yaw + pi) %  (2 * pi) - pi
        _beta = (pi/2 - self.goal_angle + pi)
        v = K_RO * self.goal_distance
        w = K_ALPHA * _alpha + K_BETA * _beta

        v_scale = v / abs(v) * V_CONST
        w_scale = w / abs(v) * V_CONST

        return v_scale, w_scale

    def step(self, action):
        linear_vel, angular_vel = self.do_action(action)
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        scan_range = self.lidar_scan(data)

        state, done = self.getState(scan_range)
        reward = self.setReward(done, action, self.prev_action, scan_range, self.prev_lidar)

        self.prev_action = action
        self.prev_lidar = scan_range

        return state, reward, done
    
    def do_action(self, action):
        linear_vel, angular_vel = 0 , 0
        if (action == 0):
            linear_vel = CONST_LINEAR_SPEED_FORWARD
            angular_vel = CONST_ANGULAR_SPEED_FORWARD
        elif (action == 1):
            linear_vel = CONST_ANGULAR_SPEED_TURN
            angular_vel = CONST_ANGULAR_SPEED_TURN
        elif (action == 2):
            linear_vel = CONST_LINEAR_SPEED_TURN
            angular_vel = -CONST_ANGULAR_SPEED_TURN
        elif (action == 3):
            linear_vel, angular_vel = self.feed_back_control()
        else:
            linear_vel = 0
            angular_vel = 0

        return linear_vel, angular_vel

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        
        scan_range = self.lidar_scan(data)
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(scan_range)
        self.prev_lidar = scan_range

        return state, done