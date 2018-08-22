#!/usr/bin/python

# This function is used to experiment the Centroidal Momentum Matrix control with the Quadratic Programming

import sys, os
from klampt import *
from klampt.vis import GLSimulationProgram
import numpy as np
sys.path.insert(0, '/home/shihao/trajOptLib')
from trajOptLib.io import getOnOffArgs
from trajOptLib import trajOptCollocProblem
from trajOptLib.snoptWrapper import directSolve
from trajOptLib.libsnopt import snoptConfig, probFun, solver
import functools
import ipdb, string

Inf = float("inf")
pi = 3.1415926535897932384626
End_Link_Ind = [11, 17, 27, 34]                       # The link index of the end effectors
DOF = 0
eps_dist = 0.015
Terr_No = 0             # Number of Terrains in the current simulation

class MyGLViewer(GLSimulationProgram):
    def __init__(self,world):
        #create a world from the given files
        self.world = world
        GLSimulationProgram.__init__(self,world,"My GL program")
        self.traj = model.trajectory.RobotTrajectory(world.robot(0))
        self.traj.load("/home/shihao/Klampt/data/motions/athlete_flex.path")
    def control_loop(self):
        #Put your control handler here
        sim = self.sim
        traj = self.traj
        starttime = 2.0
        if sim.getTime() > starttime:
            (q,dq) = (traj.eval(self.sim.getTime()-starttime),traj.deriv(self.sim.getTime()-starttime))
            sim.controller(0).setPIDCommand(q,dq)

    def contact_force_login(self):
        world = self.world
        terrain = TerrainModel();  # Now terrain is an instance of the TerrainModel class
        terrainid = terrain.getID()
        print terrainid
        # objectid = world.rigidObject(object_index).getID()
        # linkid = world.robot(robot_index).link(link_index).getID()
        # #equivalent to
        # linkid = world.robotlink(robot_index,link_index).getID()

    def mousefunc(self,button,state,x,y):
        #Put your mouse handler here
        #the current example prints out the list of objects clicked whenever
        #you right click
        print "mouse",button,state,x,y
        if button==2:
            if state==0:
                print [o.getName() for o in self.click_world(x,y)]
            return
        GLSimulationProgram.mousefunc(self,button,state,x,y)

    def motionfunc(self,x,y,dx,dy):
        return GLSimulationProgram.motionfunc(self,x,y,dx,dy)

def Configuration_Loader():
    # This function is only used to load in the initial configuraiton
    # The initial file will be in the .config format
    with open("Config_Init.config",'r') as robot_angle_file:
        robotstate_angle_i = robot_angle_file.readlines()
    config_temp = [x.replace('\t',' ') for x in robotstate_angle_i]
    config_temp = [x.replace('\n','') for x in config_temp]
    config_temp = [float(i) for i in config_temp[0].split()]

    DOF = int(config_temp[0])
    Config_Init = np.array(config_temp[1:])
    return DOF, Config_Init

class Robot_Init_Opt_Prob(probFun):
    # This is the class type used for the initial condition validation optimization problem
    def __init__(self, world, sim_robot, sigma_init, state_init):

        self.grad = False

        self.world = world
        self.sim_robot = sim_robot
        self.sigma_init = sigma_init
        self.state_init = state_init

        nx = len(state_init)
        y_val, y_type = Initial_Constraint_Eqn(sigma_init, state_init, sim_robot, world, state_init)
        nc = len(y_type)
        probFun.__init__(self, nx, nc)

    def __callf__(self, x, y):
        y_val, y_type = Initial_Constraint_Eqn(self.sigma_init, self.state_init, self.sim_robot, self.world, x)
        # ipdb.set_trace()
        for i in range(0,len(y_val)):
            y[i] = y_val[i]

    def __callg__(self, x, y, G, row, col, rec, needg):
        # This function will be used if the analytic gradient is provided
        y[0] = x[0] ** 2 + x[1] ** 2
        y[1] = x[0] + x[1]
        G[:2] = 2 * x
        G[2:] = 1.0
        if rec:
            row[:] = [0, 0, 1, 1]
            col[:] = [0, 1, 0, 1]

def Initial_Constraint_Eqn(sigma_init, state_init, sim_robot, world, x):
    # This function is used to generate the constraint equations for the optimization
    # Due to the specific usage of this function, it can only be used inside the structure where self is defined

    # Obj: The first function is the objective function
    sim_robot.setConfig(x[0:DOF])           # Here the robot configuration and velocities have been updated
    sim_robot.setVelocity(x[DOF:])

    y_val = []
    y_type = []

    # The constraints are:
    # The first step is the objective function which is to measure the deviation from the initial guess

    obj_val = 0
    for i in range(0,len(x)/2):
        obj_val = obj_val + (x[i] - state_init[i]) * (x[i] - state_init[i])
    # ipdb.set_trace()

    y_val.append(obj_val)
    y_type.append(1)

    # Load in the Terrain model
    Terr_Objs = []
    for i in range(0, Terr_No):
        Terr_Obj_i = world.terrain(i).geometry()
        Terr_Objs.append(Terr_Obj_i)

    # The other constraints are the holonomic constraints for the positio nand velocities
    # Three steps: Position first, Velocity second and Orientation last
    Dist = []
    Vel = []
    Ori = []

    for i in range(0,len(End_Link_Ind)):
        # Here the distance between the certain link and the environment should be set to be a relative small value while the orientation should be adjusted to be aligned
        # Only the active constraints need to be set to be zero
        if sigma_init[i]==1:
            # Distance
            End_Link_Ind_i = End_Link_Ind[i]
            robot_link_i = sim_robot.link(End_Link_Ind_i)
            Dist_i = Robot_Link_2_Terr_Dist(robot_link_i, Terr_Objs)
            Dist.append(Dist_i)

            # Velocity`
            Vel_i = robot_link_i.getVelocity()
            for j in range(0, len(Vel_i)):
                Vel.append(Vel_i[j])

            # Orientation: only the pitch angle
            Ori_i = robot_link_i.getWorldDirection([0, 1, 0])
            # ipdb.set_trace()
            Ori.append((Ori_i[2])*(Ori_i[2]))

    # Position
    for i in range(0,len(Dist)):
        y_val.append(Dist[i])
        y_type.append(0)
    # Velocity
    for i in range(0,len(Vel)):
        y_val.append(Vel[i])
        y_type.append(0)
    # Orientation
    for i in range(0,len(Ori)):
        y_val.append(Ori[i])
        y_type.append(0)

    KE_ref = 15
    KE_val = Kinetic_Energy_fn(sim_robot, x)

    y_val.append((KE_val - KE_ref) * (KE_val - KE_ref))
    y_type.append(0)
    return y_val, y_type

def Kinetic_Energy_fn(sim_robot, x):
    D_q = sim_robot.getMassMatrix()
    D_q = np.array(D_q)
    qdot = np.array(x[DOF:])

    KE_val = np.dot(np.transpose(qdot), np.dot(D_q, qdot))
    return KE_val

def Robot_Link_2_Terr_Dist(robot_link, terr_objs):
    # Thios function is used to calculate the relative distance between the rob ot given link  to the environment obstacles
    Dist = []
    for i in range(0,Terr_No):
        Dist_i = terr_objs[i].distance(robot_link.geometry())
        Dist.append(Dist_i)
    return (min(Dist) - eps_dist) * (min(Dist) - eps_dist)

def Robot_Init_Opt_fn(world, sigma_init, state_init):
    # The inputs to this function is the WorldModel, initial contact mode, and the initial state
    # The output is the feasible robotstate
    sim_robot = world.robot(0)
    qmin, qmax = sim_robot.getJointLimits()
    dqmax_val = sim_robot.getVelocityLimits()
    xlb = qmin
    xub = qmax
    for i in range(0,len(dqmax_val)):
        xlb.append(-dqmax_val[i])
        xub.append(dqmax_val[i])

    # Optimization problem setup
    Robot_Init_Opt = Robot_Init_Opt_Prob(world, sim_robot, sigma_init, state_init)
    Robot_Init_Opt.xlb = xlb
    Robot_Init_Opt.xub = xub
    # This self structure is different from the previous self structure defined in the optimization problem
    y_val, y_type = Initial_Constraint_Eqn(sigma_init, state_init, sim_robot, world, state_init)
    lb, ub = Constraint_Bounds(y_type)
    Robot_Init_Opt.lb = lb
    Robot_Init_Opt.ub = ub
    cfg = snoptConfig()
    cfg.printLevel = 1
    cfg.printFile = "result.txt"
    cfg.majorIterLimit = 150
    slv = solver(Robot_Init_Opt, cfg)
    # rst = slv.solveRand()
    rst = slv.solveGuess(state_init)

    file_object  = open("Optimized_Angle.config", 'w')
    # print rst.sol
    file_object.write("36\t")
    for i in range(0,36):
        file_object.write(str(rst.sol[i]))
        file_object.write(' ')
    file_object.close()
    return rst.sol

def Constraint_Bounds(y_type):
    # This function is used generate the bounds for the constraint equations
    lb = []
    ub = []
    High_Bd_Val = 10000000000
    for i in range(0, len(y_type)):
        if(y_type[i]>0):
            lb.append(0)
            ub.append(High_Bd_Val)
        else:
            lb.append(0)
            ub.append(0)
    return lb, ub

def main():
    # This funciton is used for the multi-contact humanoid push recovery
    # The default robot to be loaded is the HRP2 robot in this same folder
    print "This funciton is used for the multi-contact humanoid push recovery"
    if len(sys.argv)<=1:
        print "USAGE: The default robot to be loaded is the HRP2 robot in this same folder"
        exit()
    world = WorldModel() # WorldModel is a pre-defined class
    input_files = sys.argv[1:];  # sys.argv will automatically capture the input files' names

    for fn in input_files:
        result = world.readFile(fn) # Here result is a boolean variable indicating the result of this loading operation
        if not result:
            raise RuntimeError("Unable to load model "+fn)

    global Terr_No
    Terr_No = world.numTerrains()

    # This system call will rewrite the robot_angle_init.config into the robot_angle_init.txt
    # However, the initiali angular velocities can be customerized by the user in robot_angvel_init.txt

    # # Now world has already read the world file

    # The first step is to validate the feasibility of the given initial condition

    Sigma_Init = [1,1,0,0]                      # This is the initial contact status:  1__------> contact constraint is active,
                                                #                                      0--------> contact constraint is inactive
                                                #                                      [left foot, right foot, left hand, right hand]

    DOF_val, Config_Init = Configuration_Loader()        # This is the initial condition: joint angle and joint angular velocities

    # The initial joint angular velocties will be of the same length and value as the config_init
    global DOF
    DOF = DOF_val

    Velocity_Init = np.zeros(len(Config_Init))
    for i in range(0,len(Config_Init)):
        Velocity_Init[i] = Config_Init[i]
    State_Init = np.append(Config_Init, Velocity_Init)

    # Now it is the validation of the feasibility of the given initial condition
    State_Init = Robot_Init_Opt_fn(world, Sigma_Init, State_Init)
    # Here the Initial robot state has been optimized

    # The first simple test is to use a QP-controller to stabilize the robot at the current configuration without any velocity

    

    # Then the problem is to design a QP controller such that the robot's momentum can be dampened out


    # MyGLViewer(world)

    # viewer = MyGLViewer(sys.argv[1:])
    # viewer.run()

if __name__ == "__main__":
    main()
