from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import decimal
import copy
solvers.options['show_progress'] = False

class Agent:
    def __init__(self, agent_id, i_state, g_state, vg, wg, p_horizon, u_horizon):
        # agent state
        self.agent_id = agent_id # id of the agent
        self.radius = 1.90/2
        self.i_state = np.array(i_state) # start state
        self.g_state = np.array(g_state) # goal state
        self.c_state1 = i_state # current state for body-1
        self.c_state2 = self.calc_body2() # current state for body-2
        self.c_state3 = self.calc_body3() # current state for body-3
        
        # horizons
        self.p_horizon = p_horizon # planning horizon
        self.u_horizon = u_horizon # update horizon
        # initial guesses
        self.vg = matrix(vg)  # initial velocity
        self.wg = matrix(wg) # initial angular velocity
        # last known values
        self.vl = 0 # last known velocity of the agent
        self.wl = 0 # last known angular velocity of the agent
        # current values
        self.v = self.vl # current velocity
        self.w = self.wl # current angular velocity
        # dt
        self.dt = 0.1
        # lists to store vel and angular vel for debugging
        self.x_traj = []
        self.y_traj = []
        self.v_list = [self.v]
        self.w_list = [self.w]
        self.time_list = [] 
        self.avg_time = 0
        self.max_time = 0
    
    def calc_body2(self):
        x2, y2 = self.calc_pos(2*self.radius, 0, self.c_state1[2])
        x2 = x2 + self.c_state1[0]  
        y2 = y2 + self.c_state1[1]
        return np.array([x2, y2, self.c_state1[2]])
    
    def calc_body3(self):
        x3, y3 = self.calc_pos(4*self.radius, 0, self.c_state1[2])
        x3 = x3 + self.c_state1[0]  
        y3 = y3 + self.c_state1[1]
        return np.array([x3, y3, self.c_state1[2]])
           
    def calc_pos(self, x, y, theta):
        R = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pos = np.array([[x],[y]])
        new_pos = R@pos 
        return new_pos[0,0], new_pos[1,0]

    def get_P_q_x(self):
        
        P_x_ = np.ones((1,2*self.p_horizon))
        
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s
        
        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state1[2]   
        for i in range(self.p_horizon):
            th_[0,i] = th_0 + d_th[0,i]*self.dt
        
        d_x = np.ones((1,self.p_horizon)) # contains a, b, c
        for i in range(self.p_horizon):
            d_x[0,i] = np.cos(th_[0,i])*self.dt
        
        d_w = np.ones((1,self.p_horizon)) # contains d, e, f
        for i in range(self.p_horizon):
            d_w[0,i] = -self.vg[i]*np.sin(th_[0,i])*self.dt**2
            
        s_dw = np.ones((1,self.p_horizon)) # contains d+e+f, e+f, f
        for i in range(self.p_horizon):
            s_dw[0,i] = np.sum(d_w[0,i:self.p_horizon])
        
        P_x_[0,0:self.p_horizon] = d_x
        P_x_[0,self.p_horizon:] = s_dw
        
        P_x = P_x_.T@P_x_
        
        ### Solving for q_x
        q_x = np.ones((2*self.p_horizon))
        x_0 = self.c_state1[0]
        v_sum = 0
        w_sum = 0
        for i in range(self.p_horizon):
#             v_sum = v_sum + self.vg[i]*d_x[0,i]
            w_sum = w_sum + self.wg[i]*s_dw[0,i]
            
        z = x_0 - w_sum - self.g_state[0]
        q_x[0:self.p_horizon] = 2*z*d_x
        q_x[self.p_horizon:] = 2*z*s_dw 
        return P_x, q_x
    
    def get_P_q_y(self):
        
        P_y_ = np.ones((1,2*self.p_horizon))
        
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s
        
        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state1[2]   
        for i in range(self.p_horizon):
            th_[0,i] = th_0 + d_th[0,i]*self.dt
        
        d_y = np.ones((1,self.p_horizon)) # contains a, b, c
        for i in range(self.p_horizon):
            d_y[0,i] = np.sin(th_[0,i])*self.dt
        
        d_w = np.ones((1,self.p_horizon)) # contains d, e, f
        for i in range(self.p_horizon):
             d_w[0,i] = self.vg[i]*np.cos(th_[0,i])*self.dt**2
            
        s_dw = np.ones((1,self.p_horizon)) # contains d+e+f, e+f, f
        for i in range(self.p_horizon):
            s_dw[0,i] = np.sum(d_w[0,i:self.p_horizon])
            
        
        P_y_[0,0:self.p_horizon] = d_y
        P_y_[0,self.p_horizon:] = s_dw
        
        P_y = P_y_.T@P_y_
        
        ### Solving for q_y
        q_y = np.ones((2*self.p_horizon))
        y_0 = self.c_state1[1]
        w_sum = 0
        for i in range(self.p_horizon):
            w_sum = w_sum + self.wg[i]*s_dw[0,i]
            
        z = y_0 - w_sum - self.g_state[1]
        q_y[0:self.p_horizon] = 2*z*d_y
        q_y[self.p_horizon:] = 2*z*s_dw 
        
        return P_y, q_y
        
    def get_P_q_theta(self):
        P_theta = np.zeros((2*self.p_horizon,2*self.p_horizon))
        P_theta[self.p_horizon:,self.p_horizon:]=self.dt**2*np.ones((self.p_horizon,self.p_horizon))
        
        q_theta = np.zeros((2*self.p_horizon))
        theta_0 = self.c_state1[2]
        theta_g = self.g_state[2]
        q_theta[self.p_horizon:]=2*(theta_0 - theta_g)*self.dt * np.ones((self.p_horizon))
    
        return P_theta, q_theta
    
    def get_diff_vel_mat(self):
        P_diff = np.zeros((self.p_horizon,self.p_horizon))
        for i in range(self.p_horizon-1):
            for j in range(i,i+2):
                P_diff[i,j] = 1
        P_diff[-1,-1] = 1
        P_diff = P_diff + P_diff.T - np.eye(self.p_horizon)
        P_diff = np.concatenate( (P_diff, P_diff), axis = 1 )
        P_diff = np.concatenate( (P_diff, P_diff), axis = 0 )
        
        q_diff = np.zeros((2*self.p_horizon))
        return P_diff, q_diff
    
    def get_continuity_mat(self):
        P_cont = np.zeros((self.p_horizon,self.p_horizon))
        P_cont[0,0] = 1
        P_cont = np.concatenate( (P_cont, P_cont), axis = 1 )
        P_cont = np.concatenate( (P_cont, P_cont), axis = 0 )
        
        q_cont = np.zeros((2*self.p_horizon ))
        q_cont[0] = -2*self.vl
        q_cont[self.p_horizon] = -2*self.wl
        
        return P_cont, q_cont
        
    def pred_controls(self):
        # define the cost function here and optimize the controls to minimize it

        w1 = 1
#         w2 = 0
#         w3 = 1
        v_cost = 99999
        w_cost = 99999
        threshold = 1
#         count = 0
        strt_time = time.time()
        while((v_cost > threshold) or (w_cost > threshold)):
#         for i in range(1):
            P_x, q_x = self.get_P_q_x()
            P_y, q_y = self.get_P_q_y()
            P_theta, q_theta = self.get_P_q_theta()
            
            P_cost_1 = w1 * ( P_x + P_y + P_theta ) # P matrix for goal reaching cost
            q_cost_1 = w1 * ( q_x + q_y + q_theta ) # q vector for goal reaching cost
            
            P = 2*matrix( P_cost_1 , tc='d')
            q = matrix( q_cost_1 , tc='d')
            
            ### Constraints
            
            ## Bound constraints
            
            v_bound = 5
            w_bound = 0.1
            amin = -2 # min acceleration
            amax = 1 # max acceleration
            alphamax = 0.1 # max angular acceleration
            alphamin = -0.1 # min angular acceleration
            
            v_ub = 30*np.ones((self.p_horizon,1))
            v_lb = 0*np.ones((self.p_horizon,1))
            w_ub = 0.5*np.ones((self.p_horizon,1))
            w_lb = -0.5*np.ones((self.p_horizon,1))
            a_ubound = amax * self.dt * np.ones((self.p_horizon-1,1))
            a_lbound = - amin * self.dt * np.ones((self.p_horizon-1,1))
            alpha_ubound = alphamax * self.dt * np.ones((self.p_horizon-1,1))
            alpha_lbound = - alphamin * self.dt * np.ones((self.p_horizon-1,1))

            h = matrix( np.concatenate( \
                                      (v_ub, w_ub, -v_lb, -w_lb, self.vg+v_bound, self.wg+w_bound, -(self.vg-v_bound), -(self.wg-w_bound), \
                                       a_ubound, a_lbound, alpha_ubound, alpha_lbound ), \
                                      axis=0), tc='d')  
            
            av_max = np.concatenate( (np.diff(np.eye(self.p_horizon),axis = 0), np.zeros( (self.p_horizon-1,self.p_horizon) ) ), axis=1 )
            av_min = -av_max
            aw_max = np.concatenate( ( np.zeros( (self.p_horizon-1,self.p_horizon) ), np.diff( np.eye(self.p_horizon) ,axis = 0 ) ), axis=1 )
            aw_min = -aw_max
            
            G = matrix(np.concatenate( \
                                      ( np.eye(2*self.p_horizon),-np.eye(2*self.p_horizon), np.eye(2*self.p_horizon), -np.eye(2*self.p_horizon), \
                                      av_max, av_min, aw_max, aw_min ), \
                                      axis=0), tc='d')

            ## Continuity constraints
            
            A = np.zeros((2,2*self.p_horizon))
            A[0,0] = 1
            A[1,self.p_horizon] = 1
            A = matrix(A,tc='d')
            b = matrix([self.vl, self.wl],(2,1),tc='d')
            
            sol=solvers.qp(P, q, G, h, A, b)
            v_cost = np.linalg.norm(self.vg - sol['x'][0:self.p_horizon])
            w_cost = np.linalg.norm(self.wg - sol['x'][self.p_horizon:])
            self.vg = sol['x'][0:self.p_horizon]
            self.wg = sol['x'][self.p_horizon:]
            
        end_time = time.time()
        self.time_list.append(end_time-strt_time)
        return sol
       
    def non_hol_update(self):
        self.c_state1[2] = self.c_state1[2] + self.w*self.dt
        self.c_state1[0] = self.c_state1[0] + self.v*np.cos(self.c_state1[2])*self.dt
        self.c_state1[1] = self.c_state1[1] + self.v*np.sin(self.c_state1[2])*self.dt 
        self.c_state2 = self.calc_body2()
        self.c_state3 = self.calc_body3()
    
    def get_traj(self,k):
        state = copy.deepcopy(self.c_state1)
        for i in range(k,self.p_horizon):
            state[2] = state[2] + self.wg[i]*self.dt
            state[0] = state[0] + self.vg[i]*np.cos(state[2])*self.dt
            state[1] = state[1] + self.vg[i]*np.sin(state[2])*self.dt
            self.x_traj.append(state[0])
            self.y_traj.append(state[1])
                
def draw_circle(x,y,r):
    th = np.arange(0,2*np.pi,0.01)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    return xunit, yunit 

def main():
    p_horizon = 50
    u_horizon = 10
    ### initialize vg and wg
    vg = 20*np.ones((p_horizon,1))
    wg = 0.1*np.ones((p_horizon,1))

    a = Agent(1, [0,0,np.deg2rad(45)],[50,50,np.deg2rad(45)], vg, wg, p_horizon, u_horizon)
    th = 0.5
    timeout = 200
    rec_video = False
    if(rec_video):
        plt_sv_dir = "../2_pipeline/tmp/"
        p = 0
    fig = plt.figure()    
    while( (np.linalg.norm(a.c_state1-a.g_state)>th) and timeout>0):
        a.pred_controls()
        for i in range(u_horizon):
            a.v = a.vg[i]
            a.w = a.wg[i]
            a.v_list.append(a.v)
            a.x_traj = []
            a.y_traj = []
            a.get_traj(i)
            a.non_hol_update()
            # agent_pos = viz_agent(a.c_state1[0], a.c_state1[1], a.c_state1[2]) # coordinates of the three circles
            cx1, cy1 = draw_circle(a.c_state1[0], a.c_state1[1], a.radius)
            cx2, cy2 = draw_circle(a.c_state2[0], a.c_state2[1], a.radius)
            cx3, cy3 = draw_circle(a.c_state3[0], a.c_state3[1], a.radius)

            plt.plot(cx1,cy1,'b')
            plt.plot(cx2,cy2,'b')
            plt.plot(cx3,cy3,'b')

            plt.scatter(a.g_state[0],a.g_state[1],marker='x', color='r')
            plt.scatter(a.x_traj, a.y_traj,marker='.', color='r', s=1)
            plt.plot([a.c_state1[0],a.g_state[0]],[a.c_state1[1],a.g_state[1]], linestyle='dotted', c='k')
            # plt.axis('equal')
            plt.xlim([-10,60])
            plt.ylim([-10,60])
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=100)
                p = p+1
                plt.clf()
            else:
                plt.pause(1e-10)
                plt.clf()
            
            timeout = timeout - a.dt
        a.vl = a.v
        a.wl = a.w
        
    a.avg_time = sum(a.time_list[1:])/26
    print("average time taken for each optimization step: {} secs".format(a.avg_time))
    a.max_time = max(a.time_list[1:])
    print("Maximum time taken for each optimization step: {} secs".format(a.max_time))
    if(timeout <= 0):
        print("Stopped because of timeout.")

if __name__ == "__main__":
    main()