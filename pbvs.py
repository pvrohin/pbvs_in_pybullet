import pybullet as p
import pybullet_data
import time
import numpy as np
from math import *

from camera import get_image, cam_pose_to_world_pose
from sphere_fitting import sphereFit
from cam_ik import accurateIK


fov = 50
aspect = 1
near = 0.01
far = 10
width = 200
height = 200
pi=3.14159265359
f_x = width/(2*tan(fov*pi/360))
f_y = height/(2*tan(fov*pi/360))

def pix_to_ee_frame(x_pix,y_pix,d):

	#pixel to cam frame
	Zc = d
	Xc = (x_pix - (width/2))*(d/(f_x))
	Yc = (y_pix - (height/2))*(d/(f_y))

	#cam to ee frame
	Xe = Zc
	Ye = -Xc
	Ze = -Yc
	return(Xe,Ye,Ze)


def mask_points(Sbuf,Dbuf,appleId):
    depthImg_buf = np.reshape(Dbuf, [width, height])
    D = far * near / (far - (far - near) * depthImg_buf)
    S = np.reshape(Sbuf,[width,height])
    Xl = []
    Yl = []
    Zl = []
    for i in range(width):
        for j in range(height):
            if S[j][i] == appleId:
                x,y,z = pix_to_ee_frame(i,j,D[j][i])
                Xl.append(x)
                Yl.append(y)
                Zl.append(z)
    return Xl,Yl,Zl

dpos = np.array((0.2,0.0,0.0))
def visual_servo_control(pos,ort):
    lmd = 5
    Vc = -lmd*((dpos-pos)+np.cross(pos,ort))
    Wc = -lmd*(ort)
    V = np.concatenate((Vc,Wc))
    return V

def relative_ee_pose_to_ee_world_pose1(robotId,eeTargetPos,eeTargetOrn):
    ee_link_state = p.getLinkState(robotId,linkIndex=7,computeForwardKinematics=1)
    ee_pos_W=ee_link_state[-2]
    ee_ort_W=ee_link_state[-1]
    return p.multiplyTransforms(ee_pos_W,ee_ort_W,eeTargetPos,eeTargetOrn)


clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

time_step = 0.001
gravity_constant = -9.81
p.resetSimulation()
p.setTimeStep(time_step)
p.setGravity(0.0, 0.0, gravity_constant)

p.loadURDF("plane.urdf", [0, 0, -0.3])

cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

gripper_left = p.addUserDebugParameter('Gripper_left', -0.5, 0.5, 0)
gripper_right = p.addUserDebugParameter('Gripper_right', -0.5, 0.5, 0)

front_left = p.addUserDebugParameter('front_left', 0, 20, 0)
front_right = p.addUserDebugParameter('front_right', 0, 20, 0)
rear_left = p.addUserDebugParameter('rear_left', 0, 20, 0)
rear_right = p.addUserDebugParameter('rear_right', 0, 20, 0)

appleId = p.loadURDF("urdf/apple1/apple.urdf",[0.3,1.0,-0.2],useFixedBase=0)
kukaId = p.loadURDF("ur_description/urdf/arm_with_gripper.urdf",[-0.5,0.5,0.05], cubeStartOrientation)
baseId = p.loadURDF("ur_description/urdf/mobile_base_without_arm.urdf",[-0.6,0.5,-0.1], cubeStartOrientation)

number_of_joints = p.getNumJoints(kukaId)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(kukaId, joint_number)
    print(info[0], ": ", info[1])

number_of_joints = p.getNumJoints(baseId)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(baseId, joint_number)
    print(info[0], ": ", info[1])

roboId = kukaId
required_joints = [0,-1.9,1.9,-1.57,-1.57,0,0]
for i in range(1,7):
    p.resetJointState(bodyUniqueId=roboId,
                            jointIndex=i,
                            targetValue=required_joints[i-1])
p.resetDebugVisualizerCamera( cameraDistance=2.2, cameraYaw=140, cameraPitch=-60, cameraTargetPosition=[0,0,0])


for i in range (10000):
    user_input_left = p.readUserDebugParameter(gripper_left)
    user_input_right = p.readUserDebugParameter(gripper_right)

    wheel_front_left = p.readUserDebugParameter(front_left)
    wheel_front_right = p.readUserDebugParameter(front_right)
    wheel_back_left = p.readUserDebugParameter(rear_left)
    wheel_back_right = p.readUserDebugParameter(rear_right)

    p.setJointMotorControl2(kukaId,14,p.POSITION_CONTROL,targetPosition=user_input_left)
    p.setJointMotorControl2(kukaId,16,p.POSITION_CONTROL,targetPosition=user_input_right)

    #p.setJointMotorControl2(baseId,0,p.VELOCITY_CONTROL,targetVelocity=20,force=500)
    p.setJointMotorControl2(baseId,1,p.VELOCITY_CONTROL,targetVelocity=wheel_front_left,force=500)
    p.setJointMotorControl2(baseId,2,p.VELOCITY_CONTROL,targetVelocity=wheel_front_right,force=500)
    p.setJointMotorControl2(baseId,3,p.VELOCITY_CONTROL,targetVelocity=wheel_back_left,force=500)
    p.setJointMotorControl2(baseId,4,p.VELOCITY_CONTROL,targetVelocity=wheel_back_right,force=500)

    p.stepSimulation()
    time.sleep(1./240.)
    I,Dbuf,Sbuf = get_image(kukaId)
    sx,sy,sz = mask_points(Sbuf,Dbuf,appleId)
    r,cx,cy,cz = sphereFit(sx,sy,sz)
    # cx = np.array(sx).mean()
    # cy = np.array(sy).mean()
    # cz = np.array(sz).mean()
    pos = np.array((cx,cy,cz))[:,0]
    error = np.sqrt(np.mean((pos-dpos)**2))
    if error < 0.00045:
      break

    print(pos,error)
    ort = np.array((0.0,0.0,0.0))
    Ve = visual_servo_control(pos,ort)
    relative_pos_cam = 0.01*Ve[0:3]
    relative_rot_cam = 0.01*Ve[3:6]
    relative_quat_cam = p.getQuaternionFromEuler(relative_rot_cam)
    pos2,ort2 = relative_ee_pose_to_ee_world_pose1(roboId,relative_pos_cam,relative_quat_cam)
    accurateIK(roboId,7,pos2,ort2,useNullSpace=False)

c = input('press enter to end')
p.disconnect()
