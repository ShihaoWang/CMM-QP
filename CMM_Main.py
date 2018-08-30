#!/usr/bin/python

# This function is used to experiment the Centroidal Momentum Matrix control with the Quadratic Programming
import osqp
import sys, os
from klampt import *
from klampt import vis
from klampt.vis import GLSimulationPlugin
import numpy as np
sys.path.insert(0, '/home/shihao/trajOptLib')
from trajOptLib.io import getOnOffArgs
from trajOptLib import trajOptCollocProblem
from trajOptLib.snoptWrapper import directSolve
from trajOptLib.libsnopt import snoptConfig, probFun, solver
import functools
import ipdb, string
import scipy as sp
import scipy.sparse as sparse
from klampt.model.trajectory import Trajectory
from random import randint

Inf = float("inf")
pi = 3.1415926535897932384626
End_Link_Ind = [17, 11, 34, 27]                       # The link index of the end effectors
DOF = 0
eps_dist = 0.005
Terr_No = 0             # Number of Terrains in the current simulation
mu = 0.35               # The default friction coefficient

class MyGLViewer(GLSimulationPlugin):
    def __init__(self, world, Init_Config, Init_Velocity, Init_Sigma):
        #create a world from the given files
        self.world = world
        self.sim_robot = world.robot(0)
        self.sigma_init = Init_Sigma
        robot = world.robot(0)
        robot.setConfig(Init_Config)
        robot.setVelocity(Init_Velocity)

        GLSimulationPlugin.__init__(self,world)

        self.Beta_Number = 6
        beta_array = np.zeros(self.Beta_Number)
        Constraint_Force, Constraint_Pyramid = Constraint_Force_Pyramid(beta_array)
        self.Constraint_Pyramid = Constraint_Pyramid
        C_q_qdot = robot.getCoriolisForceMatrix()       # This matrix is only used to initialize the quadratic matrix
        self.State_Number = len(C_q_qdot)
        self.Control_Number = self.State_Number - 6     # The default setting is that all joints are actuated

        self.foot_contacts = 4
        self.hand_contacts = 1
        self.Cont_Force_Number = 2 * (self.foot_contacts + self.hand_contacts)

        # Define the constraint P matrix for the QP problem
        w_qddot = 10
        w_beta = 0
        w_tau = 0

        # Here the Q matrix can be directly formulated since it remains to be a constant value during the whole time
        self.m = self.State_Number + self.Cont_Force_Number * self.Beta_Number + 3 * self.Cont_Force_Number + self.Control_Number

        weight_list = []

        # The first part is the qddot coefficients
        for i in range(0, self.State_Number):
            weight_list.append(w_qddot)

        # The second part is the beta coefficients
        for i in range(0, self.Cont_Force_Number  * self.Beta_Number):
            weight_list.append(w_beta)

        # The third part is the contact force coefficients
        for i in range(0, self.Cont_Force_Number):
            weight_list.append(0)

        # The last part is the torque coefficients
        for i in range(0, self.Control_Number):
            weight_list.append(w_tau)

        # Then the next step is to convert it into a csc_matrix
        Data_List, Row_List, Col_List = Diagnal2DRC(weight_list)

        P = sparse.csc_matrix((Data_List,(Row_List,Col_List)), shape=(self.m,self.m))

        # The next part is the read the torque limit
        TorqueLimits =  self.sim_robot.getTorqueLimits() # From the output, it is clear that the first 6 components are related to the COM
        TorqueLimits = TorqueLimits[6:]
        self.low_torque = -np.array(TorqueLimits)
        self.hgh_torque = np.array(TorqueLimits)

        # The next part is the acceleration limit
        AccLimits =  self.sim_robot.getAccelerationLimits() # From the output, it is clear that the first 6 components are related to the COM
        self.low_acc = -np.array(AccLimits)
        self.hgh_acc = np.array(AccLimits)

        self.P = P

    def control_loop(self):
        # The first simple test is a QP stabilizer to maintain the robot's configuration at the current setting

        qddot_t, beta_t, lamda_t, tau_t = QP_Controller(self)

        #Put your control handler here
        sim = self.sim
        tau_t_list = tau_t.tolist()
        # ipdb.set_trace()

        torque_t = [ 0, 0,  0,  0,  0,  0]
        torque_t = []
        for i in range(0,tau_t.size):
            torque_t.append(tau_t[i])

        dt = 0.02
        # After knowing the sampling rate dt = 0.02

        Pos_t = np.array(self.sim_robot.getConfig())
        Vel_t = np.array(self.sim_robot.getVelocity())
        Acc_t = qddot_t

        Pos_tp1 = Pos_t + dt * Vel_t + 1/2 * dt * dt * Acc_t
        Vel_tp1 = Vel_t + dt * Acc_t


        # sim.controller(0).setTorque(torque_t)

        sim.controller(0).addMilestone(Pos_tp1, Vel_tp1)


        # ipdb.set_trace()


        # if sim.getTime() > starttime:
        #     q=sim.controller(0).getCommandedConfig()
        #     q[7]-=1.0
        #     sim.controller(0).setMilestone(q)
        #     q[7]+=1.5
        #     sim.controller(0).addMilestone(q)

    def contact_force_login(self):
        world = self.world
        terrain = TerrainModel();  # Now terrain is an instance of the TerrainModel class
        terrainid = terrain.getID()

        # objectid = world.rigidObject(object_index).getID()
        # linkid = world.robot(robot_index).link(link_index).getID()
        # #equivalent to
        # linkid = world.robotlink(robot_index,link_index).getID()

    def mousefunc(self,button,state,x,y):
        #Put your mouse handler here
        #the current example prints out the list of objects clicked whenever
        #you right click
        print "mouse",button,state,x,y
        # if button==2:
        #     if state==0:
        #         print [o.getName() for o in self.click_world(x,y)]
        #     return
        GLSimulationPlugin.mousefunc(self,button,state,x,y)

    def motionfunc(self,x,y,dx,dy):
        return GLSimulationPlugin.motionfunc(self,x,y,dx,dy)

def Diagnal2DRC(Diag_List):
    # This function is used to calculate the row, column for the diagnol list
    Data_No = len(Diag_List)
    Index_List = []
    Data_List = []
    for i in range(0, Data_No):
        Data_i = Diag_List[i]
        if Data_i == 0:
            pass
        else:
            Index_List.append(i)
            Data_List.append(Diag_List[i])
    return Data_List, Index_List, Index_List

def Constraint_Force_Pyramid(beta_array):
    # This function is used to generate the pyramid approximation of the contact forces
    # beta_list is the magnitude of the corresponding components along each axis

    n = beta_array.size

    # Then a three-dimensional pyramid is generated for approximation

    # The first point is chosen to be

    Pyramid_Unit = np.zeros((3, n))
    Base_Unit = np.array([mu/np.sqrt(1 + mu * mu), 0, 1/np.sqrt(1 + mu * mu)])
    New_Unit_List = [0, 0, Base_Unit[2]]
    Pyramid_Unit[:,0] = Base_Unit
    Support_Angle = 2*pi/n
    New_Angle = Support_Angle
    for i in range(0, n - 1):
        # The only difference in those support vectors are the x and y components

        New_x = Base_Unit[0] * np.cos(New_Angle)
        New_y = Base_Unit[0] * np.sin(New_Angle)


        New_Unit_List[0] = New_x
        New_Unit_List[1] = New_y

        New_Unit = np.array(New_Unit_List)

        New_Angle = New_Angle + Support_Angle
        Pyramid_Unit[:,i+1] = New_Unit

    # At this step Pyramid_Unit is a 3 by n matrix with each column denoting the supporting vector in that direction
    Contact_Force = np.dot(Pyramid_Unit, beta_array)
    return Contact_Force, Pyramid_Unit

def QP_Controller(self):
    # This function takes in the self structure and computes the QP problem to stabilize the robot

    sim_robot = self.sim_robot

    sigma = self.sigma_init # In this case, the robot sigma is the initial sigma

    # These are four constrains to be added to this problem

    # The first constraint is the dynamics constraint
    D_q = sim_robot.getMassMatrix()
    D_q_sp = sparse.csc_matrix(np.array(D_q))
    Cons1_A = D_q_sp

    Cons1_B = sparse.csc_matrix((self.State_Number, self.Beta_Number * self.Cont_Force_Number))

    C_q_qdot = sim_robot.getCoriolisForces()
    C_q_qdot_sp = np.array(C_q_qdot)

    G_q = sim_robot.getGravityForces([ 0, 0, -9.8])
    G_q_sp = np.array(G_q)

    Tau_list = []
    for i in range(0, self.State_Number-6):
        Tau_list.append(-1)
    Tau_Data_List, Tau_Row_List, Tau_Col_List = Diagnal2DRC(Tau_list)

    B_up_sp = sparse.csc_matrix((6, self.State_Number-6))
    B_down_sp = sparse.csc_matrix((Tau_Data_List,(Tau_Row_List,Tau_Col_List)), shape = (self.State_Number-6,self.State_Number-6))
    Neg_B_sp = sparse.vstack([-B_up_sp, -B_down_sp]).tocsc()

    # So the first layer of the constraint equation is: D*q'' + C(q,q') = J^T * lamda + B * u

    Constraint_Pyramid = self.Constraint_Pyramid

    Jac_q = Contact_Jacobian_Matrix(sim_robot)
    Jac_q_array = np.array(Jac_q)
    Jac_q_T_array = np.transpose(Jac_q_array)
    Jac_q_T_Matrix = np.asmatrix(Jac_q_T_array)
    Neg_Jac_q_T_sp = sparse.csc_matrix(-Jac_q_T_array)

    Cons1_C = Neg_Jac_q_T_sp
    Cons1_D = Neg_B_sp

    Cons1 = sparse.hstack((Cons1_A, Cons1_B, Cons1_C, Cons1_D))

    l1 = - G_q_sp - C_q_qdot_sp
    u1 = - G_q_sp - C_q_qdot_sp

    # The second cobnstraint is the constraint force to beta coefficient

    Cons2_A = sparse.csc_matrix((3 * self.Cont_Force_Number, self.State_Number))

    Constraint_Pyramid = self.Constraint_Pyramid
    Constraint_Pyramid_sp = sparse.csc_matrix(Constraint_Pyramid)

    Cons2_B = Constraint_Pyramid_sp.copy()

    for i in range(0, self.Cont_Force_Number-1):
        Cons2_B = sparse.block_diag((Cons2_B, Constraint_Pyramid_sp), format='csc')

    Cons2_C = sparse.eye(3 * self.Cont_Force_Number)
    Cons2_D = sparse.csc_matrix((3* self.Cont_Force_Number, self.State_Number - 6))

    Cons2 = sparse.hstack((Cons2_A, Cons2_B, -Cons2_C, Cons2_D)).tocsc()
    l2 = np.zeros(3* self.Cont_Force_Number)
    u2 = np.zeros(3* self.Cont_Force_Number)

    # The thrid constraint is the beta feasibility
    Cons3_A = sparse.csc_matrix((self.Beta_Number * self.Cont_Force_Number, self.State_Number))
    Cons3_B = sparse.eye(self.Beta_Number * self.Cont_Force_Number)
    Cons3_C = sparse.csc_matrix((self.Beta_Number * self.Cont_Force_Number, 3 * self.Cont_Force_Number))
    Cons3_D = sparse.csc_matrix((self.Beta_Number * self.Cont_Force_Number, self.State_Number - 6))

    Cons3 = sparse.hstack((Cons3_A, Cons3_B, Cons3_C, Cons3_D)).tocsc()
    l3 = np.zeros(self.Beta_Number * self.Cont_Force_Number)
    u3 = np.inf*np.ones(self.Beta_Number * self.Cont_Force_Number)

    # The fourth constraint is the constraint force active constraint
    Cons4_A = sparse.csc_matrix((3 * self.Cont_Force_Number, self.State_Number))
    Cons4_B = sparse.csc_matrix((3 * self.Cont_Force_Number, self.Beta_Number * self.Cont_Force_Number))
    Cons4_C = Constraint_Force_Selection_Matrix(sigma)
    Cons4_D = sparse.csc_matrix((3 * self.Cont_Force_Number, self.State_Number - 6))
    Cons4 = sparse.hstack((Cons4_A, Cons4_B, Cons4_C, Cons4_D)).tocsc()

    l4 = np.zeros(3 * self.Cont_Force_Number)
    u4 = np.zeros(3 * self.Cont_Force_Number)

    # The five constraint is the torque limit constraint
    Cons5 = sparse.hstack((sparse.csc_matrix(( self.State_Number - 6, self.m - self.State_Number + 6)), sparse.eye(self.State_Number - 6))).tocsc()

    l5 = self.low_torque
    u5 = self.hgh_torque

    # The six constraint is the acceleration limit
    Cons6 = sparse.hstack((sparse.eye(self.State_Number), sparse.csc_matrix((self.State_Number, self.m - self.State_Number)))).tocsc()
    l6 = self.low_acc
    u6 = self.hgh_acc

    # The seventh constraint is the acceleration at the end effector
    Cons7_A_left = Jacobian_Selection_Matrix(sigma).todense()
    Cons7_A_right = np.array(Contact_Jacobian_Matrix(sim_robot))
    Cons7_A = np.dot(Cons7_A_left, Cons7_A_right)
    Cons7_B = sparse.csc_matrix((3 * self.Cont_Force_Number, self.m - self.State_Number))
    Cons7 = sparse.hstack((Cons7_A, Cons7_B)).tocsc()
    l7 = np.zeros(3 * self.Cont_Force_Number)
    u7 = np.zeros(3 * self.Cont_Force_Number)

    # The seventh constraint is at the acceleration which means that we would like the end effectors to have zero acceleration

    Cons = sparse.vstack((Cons1, Cons2, Cons3, Cons4, Cons5, Cons6, Cons7)).tocsc()
    l = np.hstack([l1, l2, l3, l4, l5, l6, l7])
    u = np.hstack([u1, u2, u3, u4, u5, u6, u7])

    K = 5

    # Now it is the test of the second objective function
    M_left = np.eye(6)
    M_right = np.zeros((6, 150))

    M = np.hstack((M_left, M_right))

    P = np.dot(np.transpose(M), M)
    P_sp = sparse.csc_matrix(P)

    # Q_T_sp = sparse.csc_matrix(Q_T)
    # ipdb.set_trace()

    q = np.zeros(self.m)

    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P_sp, q, Cons, l, u)

    # Solve problem
    res = prob.solve()

    # ipdb.set_trace()

    soln = res.x

    qddot = soln[0:self.State_Number]

    soln = soln[self.State_Number:]

    beta = soln[0:self.Beta_Number * self.Cont_Force_Number]
    soln = soln[self.Beta_Number * self.Cont_Force_Number:]

    lamda = soln[0:3 * self.Cont_Force_Number]
    soln = soln[3 * self.Cont_Force_Number:]

    tau = soln[:]

    # Constraint validation
    # 1. Dynamics constraint
    D_q_qddot = np.dot(D_q_sp.todense(), qddot)
    C_q_qdot_G_q = C_q_qdot_sp + G_q_sp
    Jac_T_Lamda = np.dot(Jac_q_T_Matrix, lamda)
    B_Tau = -np.dot(Neg_B_sp.todense(), tau)
    dyn_val = D_q_qddot + C_q_qdot_G_q - Jac_T_Lamda - B_Tau

    # 2. Beta coefficient to lamda
    lamda_beta = []
    for i in range(0, self.Cont_Force_Number):
        beta_i = beta[(i*self.Beta_Number):((i+1)*self.Beta_Number)]
        lamda_beta_i = np.dot(Constraint_Pyramid, beta_i)
        if i == 0:
            lamda_beta = lamda_beta_i.copy()
        else:
            lamda_beta = np.append(lamda_beta, lamda_beta_i)

    # ipdb.set_trace()

    print qddot

    # print np.dot(Cons7_A_right, qddot)

    return qddot, beta, lamda, tau

def Jacobian_Selection_Matrix(sigma):
    # This function is used to select the specific component of the Jacobian matrix
    Status_List = []
    # Left foot force
    if sigma[0] == 1:
        for i in range(0, 12):
            Status_List.append(1)
    else:
        for i in range(0, 12):
            Status_List.append(0)

    # Right foot force
    if sigma[1] == 1:
        for i in range(0, 12):
            Status_List.append(1)
    else:
        for i in range(0, 12):
            Status_List.append(0)

    # Left hand force
    if sigma[2] == 1:
        for i in range(0, 3):
            Status_List.append(1)
    else:
        for i in range(0, 3):
            Status_List.append(0)

    # Right hand force
    if sigma[3] == 1:
        for i in range(0, 3):
            Status_List.append(1)
    else:
        for i in range(0, 3):
            Status_List.append(0)

    Data_List, Row_List, Col_List = Diagnal2DRC(Status_List)

    M = sparse.csc_matrix((Data_List,(Row_List,Col_List)), shape=(len(Status_List), len(Status_List)))

    return M

def Constraint_Force_Selection_Matrix(sigma):
    # This function is used to select the specific component of the constraint force and set them to zero
    Status_List = []
    # Left foot force
    if sigma[0] == 0:
        for i in range(0, 12):
            Status_List.append(1)
    else:
        for i in range(0, 12):
            Status_List.append(0)

    # Right foot force
    if sigma[1] == 0:
        for i in range(0, 12):
            Status_List.append(1)
    else:
        for i in range(0, 12):
            Status_List.append(0)

    # Left hand force
    if sigma[2] == 0:
        for i in range(0, 3):
            Status_List.append(1)
    else:
        for i in range(0, 3):
            Status_List.append(0)

    # Right hand force
    if sigma[3] == 0:
        for i in range(0, 3):
            Status_List.append(1)
    else:
        for i in range(0, 3):
            Status_List.append(0)

    Data_List, Row_List, Col_List = Diagnal2DRC(Status_List)

    M = sparse.csc_matrix((Data_List,(Row_List,Col_List)), shape=(len(Status_List), len(Status_List)))

    return M

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

    sigma_sum = sigma_init[0] + sigma_init[1] + sigma_init[2] + sigma_init[3]

    obj_val = 0
    if sigma_sum == 4:
        obj_val = 0
    else:
        for i in range(0,len(x)/2):
            obj_val = obj_val + (x[i] - state_init[i]) * (x[i] - state_init[i])

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

    KE_ref = 5
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

    #  Here a basic solution separation will be conducted to get the configuraiton and velocity
    Initial_State_List = []
    for i in range(0, rst.sol.size):
        Initial_State_List.append(rst.sol[i])

    Init_Config = Initial_State_List[0:rst.sol.size/2]
    Init_Velocity = Initial_State_List[rst.sol.size/2:]
    return Init_Config, Init_Velocity

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

    # Init_Config, Init_Velocity = Robot_Init_Opt_fn(world, Sigma_Init, State_Init)
    #
    # Init_Config_File = open('Init_Config.txt', 'w')
    # for Config in Init_Config:
    #     print>>Init_Config_File, Config
    #
    # Init_Velocity_File = open('Init_Velocity.txt', 'w')
    # for Velocity in Init_Velocity:
    #     print>>Init_Velocity_File, Velocity

    # However, the easy method is to directly read in the initial configuration and the initial velocity
    with open('./Init_Config.txt') as Init_Config_File:
        Init_Config = Init_Config_File.read().splitlines()
    Init_Config = [float(i) for i in Init_Config]

    with open('./Init_Velocity.txt') as Init_Velocity_File:
        Init_Velocity = Init_Velocity_File.read().splitlines()
    Init_Velocity = [float(i) for i in Init_Velocity]

    # ipdb.set_trace()

    # Here the Initial robot state has been optimized

    # since the optimization of the initial configuration takes longer time, we save the optimized result and directly use it
    # ipdb.set_trace()

    sim_robot = world.robot(0)

    viewer = MyGLViewer(world, Init_Config, Init_Velocity, Sigma_Init)

    # This part of the code  is used to find the extremities in the local coordinate

    # Local_Extremeties = [0.1, 0, -0.1, -0.15 , 0, -0.1, 0.1, 0, -0.1, -0.15 , 0, -0.1, 0, 0, -0.22, 0, 0, -0.205]         # This 6 * 3 vector describes the local coordinate of the contact extremeties in their local coordinate

    # The left foot: 4 points!


    # vis.pushPlugin(viewer)
    #add the world to the visualizer
    # vis.add("world", world)
    # sim_robot.setConfig(Init_Config)
    #
    # vis.show()

    # vis.lock()


    # # left_foot_offset1 = [0.13, 0.055, -0.105]
    # left_foot_offset1 = [-0.015, 0.0, -0.205]
    # # left_foot_offset2 = [0.1, 0, -0.1]
    # # left_foot_offset3 = [0.1, 0, -0.1]
    # # left_foot_offset4 = [0.1, 0, -0.1]
    #
    # low_point = left_foot_offset1
    # up_point = low_point[:]
    # up_point[2] = up_point[2] - 5
    #
    # current_link = sim_robot.link(End_Link_Ind[3])
    # current_link_appear = current_link.appearance()
    #
    # # print current_link.getAxis()
    #
    # link_color = random_colors()
    #
    # # print current_link.getWorldDirection([ 0, 1, 0])
    #
    # current_link_appear.setColor(link_color[0], link_color[1], link_color[2])
    #
    # low_point_coor = current_link.getWorldPosition(low_point)
    # up_point_coor = current_link.getWorldPosition(up_point)
    # print low_point_coor
    #
    # left_ft, rght_ft, left_hd, rght_hd, left_ft_end, rght_ft_end, left_hd_end, rght_hd_end = Contact_Force_vec(robot, Contact_Force_Traj_i, norm)
    # vis.add("foot force", Trajectory([0, 1], [low_point_coor, up_point_coor]))
    # vis.add("right foot force", Trajectory([0, 1], [rght_ft, rght_ft_end]))
    # vis.add("left hand force", Trajectory([0, 1], [left_hd, left_hd_end]))
    # vis.add("right hand force", Trajectory([0, 1], [rght_hd, rght_hd_end]))
    # COMPos_start = robot.getCom()
    # COMPos_end = COMPos_start
    # COMPos_end[2] = COMPos_end[2] + 100
    # vis.add("Center of Mass",  Trajectory([0, 1], [COMPos_start, COMPos_end]))

    # vis.unlock()


    # ipdb.set_trace()

    vis.run(viewer)

    # The first simple test is to use a QP-controller to stabilize the robot at the current configuration without any velocity



    # Then the problem is to design a QP controller such that the robot's momentum can be dampened out


    # MyGLViewer(world)

    # viewer = MyGLViewer(sys.argv[1:])
    # viewer.run()


def Contact_Jacobian_Matrix(sim_robot):
    # This function is used to output the Jacobian matrix given the active contact status

    # A face-to-face contactg is assumed at the foot contact
    # A point-to-face contact is assumed at the hand contact

    Left_Foot_End = [[0.13, 0.075, -0.105], [0.13, -0.055, -0.105], [-0.1, -0.055, -0.105], [-0.1, 0.075, -0.105]]

    Right_Foot_End = [[0.13, 0.055, -0.105], [0.13, -0.075, -0.105], [-0.1, -0.075, -0.105], [-0.1, 0.055, -0.105]]

    Left_Hand_End = [-0.015, 0.0, -0.205]

    Right_Hand_End = [-0.015, 0.0, -0.205]

    Left_Foot_Link = sim_robot.link(End_Link_Ind[0])
    Left_Foot_Link_Jac1 = Left_Foot_Link.getPositionJacobian(Left_Foot_End[0])
    Left_Foot_Link_Jac2 = Left_Foot_Link.getPositionJacobian(Left_Foot_End[1])
    Left_Foot_Link_Jac3 = Left_Foot_Link.getPositionJacobian(Left_Foot_End[2])
    Left_Foot_Link_Jac4 = Left_Foot_Link.getPositionJacobian(Left_Foot_End[3])

    Right_Foot_Link = sim_robot.link(End_Link_Ind[1])
    Right_Foot_Link_Jac1 = Right_Foot_Link.getPositionJacobian(Right_Foot_End[0])
    Right_Foot_Link_Jac2 = Right_Foot_Link.getPositionJacobian(Right_Foot_End[1])
    Right_Foot_Link_Jac3 = Right_Foot_Link.getPositionJacobian(Right_Foot_End[2])
    Right_Foot_Link_Jac4 = Right_Foot_Link.getPositionJacobian(Right_Foot_End[3])

    Left_Hand_Link = sim_robot.link(End_Link_Ind[2])
    Left_Hand_Link_Jac = Left_Hand_Link.getPositionJacobian(Left_Hand_End[:])

    Right_Hand_Link = sim_robot.link(End_Link_Ind[3])
    Right_Hand_Link_Jac = Right_Hand_Link.getPositionJacobian(Right_Hand_End[:])

    # Each of the Jacobian function outputs a list of length 3
    All_Jac = []

    # All four points
    All_Jac.append(Left_Foot_Link_Jac1[0])
    All_Jac.append(Left_Foot_Link_Jac1[1])
    All_Jac.append(Left_Foot_Link_Jac1[2])

    All_Jac.append(Left_Foot_Link_Jac2[0])
    All_Jac.append(Left_Foot_Link_Jac2[1])
    All_Jac.append(Left_Foot_Link_Jac2[2])

    All_Jac.append(Left_Foot_Link_Jac3[0])
    All_Jac.append(Left_Foot_Link_Jac3[1])
    All_Jac.append(Left_Foot_Link_Jac3[2])

    All_Jac.append(Left_Foot_Link_Jac4[0])
    All_Jac.append(Left_Foot_Link_Jac4[1])
    All_Jac.append(Left_Foot_Link_Jac4[2])

    # All four points
    All_Jac.append(Right_Foot_Link_Jac1[0])
    All_Jac.append(Right_Foot_Link_Jac1[1])
    All_Jac.append(Right_Foot_Link_Jac1[2])

    All_Jac.append(Right_Foot_Link_Jac2[0])
    All_Jac.append(Right_Foot_Link_Jac2[1])
    All_Jac.append(Right_Foot_Link_Jac2[2])

    All_Jac.append(Right_Foot_Link_Jac3[0])
    All_Jac.append(Right_Foot_Link_Jac3[1])
    All_Jac.append(Right_Foot_Link_Jac3[2])

    All_Jac.append(Right_Foot_Link_Jac4[0])
    All_Jac.append(Right_Foot_Link_Jac4[1])
    All_Jac.append(Right_Foot_Link_Jac4[2])

    All_Jac.append(Left_Hand_Link_Jac[0])
    All_Jac.append(Left_Hand_Link_Jac[1])
    All_Jac.append(Left_Hand_Link_Jac[2])

    All_Jac.append(Right_Hand_Link_Jac[0])
    All_Jac.append(Right_Hand_Link_Jac[1])
    All_Jac.append(Right_Hand_Link_Jac[2])

    return All_Jac

def random_colors():
    ret = []
    r = randint(0,255)/255
    g = randint(0,255)/255
    b = randint(0,255)/255

    ret.append(r)
    ret.append(g)
    ret.append(b)

    return ret

if __name__ == "__main__":
    main()
