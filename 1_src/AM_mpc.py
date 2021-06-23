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
        self.agent_radius = 2
        self.i_state = np.array(i_state) # start state
        self.g_state = np.array(g_state) # goal state
        self.c_state = i_state # current state
        # horizons
        self.p_horizon = p_horizon # planning horizon
        self.u_horizon = u_horizon # update horizon
        # initial guesses
        self.vg = matrix(vg)  # initial velocity
        self.wg = matrix(wg) # initial angular velocity
#         print(self.wg)
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
        
    def get_P_q_v_x(self):
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s

        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state[2]   
        for i in range(self.p_horizon):
            th_[0,i] = th_0 + d_th[0,i]*self.dt

        d_x = np.ones((1,self.p_horizon)) # contains a, b, c
        for i in range(self.p_horizon):
            d_x[0,i] = np.cos(th_[0,i])*self.dt

        P_v_x = d_x.T@d_x

        ### Solving for q_x
        x_0 = self.c_state[0]
        x_g = self.c_state[2]
        z = x_0 - x_g
        q_v_x = 2*z*d_x
        return P_v_x, q_v_x.T
    
    def get_P_q_v_y(self):
        
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s
        
        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state[2]   
        for i in range(self.p_horizon):
            th_[0,i] = th_0 + d_th[0,i]*self.dt
        
        d_y = np.ones((1,self.p_horizon)) # contains a, b, c
        for i in range(self.p_horizon):
            d_y[0,i] = np.sin(th_[0,i])*self.dt
        
        P_v_y = d_y.T@d_y
        
        ### Solving for q_y
        y_0 = self.c_state[1]
        z = y_0 - self.g_state[1]
        q_v_y = 2*z*d_y
        return P_v_y, q_v_y.T
    
    def get_P_q_w_x(self):
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s
        
        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state[2]   
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
              
        P_w_x = s_dw.T@s_dw
        
        ### Solving for q_x
        x_0 = self.c_state[0]
        v_sum = 0
        w_sum = 0
        for i in range(self.p_horizon):
            v_sum = v_sum + self.vg[i]*d_x[0,i]
            w_sum = w_sum + self.wg[i]*s_dw[0,i]
            
        z = x_0 + v_sum - w_sum - self.g_state[0]
        q_w_x = 2*z*s_dw
        return P_w_x, q_w_x.T
    
    def get_P_q_w_y(self):
        
        P_y_ = np.ones((1,2*self.p_horizon))
        
        d_th = np.ones((1,self.p_horizon)) # wg1, wg1+wg2, wg1+wg2+wg3
        s = 0
        for i in range(self.p_horizon):
            s = s + self.wg[i]
            d_th[0,i] = s
        
        th_ = np.ones((1,self.p_horizon)) # th0+wg1*dt, th0+(wg1+wg2)*dt, th0+(wg1+wg2+wg3)*dt
        th_0 = self.c_state[2]   
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
        
        P_w_y = s_dw.T@s_dw
        
        ### Solving for q_y
        y_0 = self.c_state[1]
        v_sum = 0
        w_sum = 0
        for i in range(self.p_horizon):
            v_sum = v_sum + self.vg[i]*d_y[0,i]
            w_sum = w_sum + self.wg[i]*s_dw[0,i]
        z = y_0 + v_sum - w_sum - self.g_state[1]
        q_w_y = 2*z*s_dw 
#         print(q_w_y.shape)
        return P_w_y, q_w_y.T
        
        
    def get_P_q_w_theta(self):
        P_w_theta = self.dt**2*np.ones((self.p_horizon,self.p_horizon))
        theta_0 = self.c_state[2]
        theta_g = self.g_state[2]        
        q_w_theta = (2 * (theta_0 - theta_g) * self.dt * np.ones((self.p_horizon))).reshape(1,-1)
#         print(q_w_theta.shape)
        return P_w_theta, q_w_theta.T    

        
    def pred_controls(self):
        # define the cost function here and optimize the controls to minimize it
        v_cost = 99999
        w_cost = 99999
        threshold = 0.01
        count = 0
        strt_time = time.time()
#         print(a.wg)
        while((v_cost > threshold) and (w_cost > threshold)):
            v_bound = 5
            w_bound = 0.1
            amin = -2 # min acceleration
            amax = 1 # max acceleration
            alphamax = 0.1 # max angular acceleration
            alphamin = -0.1 # min angular acceleration
            
            if(count % 2 == 0):
                P_v_x, q_v_x = self.get_P_q_v_x()
                P_v_y, q_v_y = self.get_P_q_v_y()
                P_v_cost = P_v_x + P_v_y
                q_v_cost = q_v_x + q_v_y
#                 print(q_v_cost.shape)
                P_v = 2*matrix(P_v_cost, tc = 'd')
                q_v = matrix(q_v_cost, tc = 'd')
                v_ub = 30*np.ones((self.p_horizon,1))
                v_lb = 0*np.ones((self.p_horizon,1))
                
                a_ubound = amax * self.dt * np.ones((self.p_horizon-1,1))
                a_lbound = - amin * self.dt * np.ones((self.p_horizon-1,1))
                
                h_v = matrix( np.concatenate( (v_ub, -v_lb, self.vg+v_bound, -(self.vg-v_bound), a_ubound, a_lbound), axis=0), tc='d')  
            
                av_max = np.diff(np.eye(self.p_horizon),axis = 0)
                av_min = -av_max

                G_v = matrix(np.concatenate( \
                                          ( np.eye(self.p_horizon),-np.eye(self.p_horizon), np.eye(self.p_horizon), -np.eye(self.p_horizon), \
                                          av_max, av_min), \
                                          axis=0), tc='d')

                ## Continuity constraints

                A_v = np.zeros((1,self.p_horizon))
                A_v[0,0] = 1
                A_v = matrix(A_v,tc='d')
                b_v = matrix([self.vl],(1,1),tc='d')

                sol_v =solvers.qp(P_v, q_v, G_v, h_v) #, A_v, b_v)
                v_cost = np.linalg.norm(self.vg - sol_v['x'])
                self.vg = sol_v['x']
                
            else:
                P_w_x, q_w_x = self.get_P_q_w_x()
                P_w_y, q_w_y = self.get_P_q_w_y()
                P_w_theta, q_w_theta = self.get_P_q_w_theta()
                P_w_cost = P_w_x + P_w_y + P_w_theta # P matrix for goal reaching cost
                q_w_cost = q_w_x + q_w_y + q_w_theta # q vector for goal reaching cost
#                 print(q_w_cost.shape)
                P_w = 2*matrix( P_w_cost , tc='d')
                q_w = matrix( q_w_cost , tc='d')
                
                ### Constraints
                w_ub = 0.5*np.ones((self.p_horizon,1))
                w_lb = -0.5*np.ones((self.p_horizon,1))
                alpha_ubound = alphamax * self.dt * np.ones((self.p_horizon-1,1))
                alpha_lbound = - alphamin * self.dt * np.ones((self.p_horizon-1,1))

                h_w = matrix( np.concatenate( (w_ub, -w_lb, self.wg+w_bound, -(self.wg-w_bound), alpha_ubound, alpha_lbound ), axis=0), tc='d')  
                aw_max = np.diff( np.eye(self.p_horizon) ,axis = 0 )
                aw_min = -aw_max

                G_w = matrix(np.concatenate( \
                                          ( np.eye(self.p_horizon),-np.eye(self.p_horizon), np.eye(self.p_horizon), -np.eye(self.p_horizon), \
                                          aw_max, aw_min ), \
                                          axis=0), tc='d')

                ## Continuity constraints
                A_w = np.zeros((1,self.p_horizon))
                A_w[0,0] = 1
                A_w = matrix(A_w,tc='d')
                b_w = matrix([self.wl],(1,1),tc='d')

                sol_w = solvers.qp(P_w, q_w, G_w, h_w) #, A_w, b_w)
                w_cost = np.linalg.norm(self.wg - sol_w['x'])
                self.wg = sol_w['x'][self.p_horizon:]
                
            count = count + 1
            
        end_time = time.time()
        self.time_list.append(end_time-strt_time)
#         return sol
       
    def non_hol_update(self):
        self.c_state[2] = self.c_state[2] + self.w*self.dt
        self.c_state[0] = self.c_state[0] + self.v*np.cos(self.c_state[2])*self.dt
        self.c_state[1] = self.c_state[1] + self.v*np.sin(self.c_state[2])*self.dt
        
    def draw_circle(self):
        th = np.arange(0,2*np.pi,0.01)
        xunit = self.agent_radius * np.cos(th) + self.c_state[0]
        yunit = self.agent_radius * np.sin(th) + self.c_state[1]
        return xunit, yunit  
    
    def get_traj(self,k):
        state = copy.deepcopy(self.c_state)
        for i in range(k,self.p_horizon):
            state[2] = state[2] + self.wg[i]*self.dt
            state[0] = state[0] + self.vg[i]*np.cos(state[2])*self.dt
            state[1] = state[1] + self.vg[i]*np.sin(state[2])*self.dt
            self.x_traj.append(state[0])
            self.y_traj.append(state[1])
                

def main():
    p_horizon = 50
    u_horizon = 10
    ### initialize vg and wg
    vg = 20*np.ones((p_horizon,1))
    wg = 0.1*np.ones((p_horizon,1))
    # vg = np.random.random((p_horizon,1))
    # wg = np.random.random((p_horizon,1))

    a = Agent(1, [0,0,np.deg2rad(45)],[80,50,np.deg2rad(45)], vg, wg, p_horizon, u_horizon)
    th = 0.5
    timeout = 200
    rec_video = False
    if(rec_video):
        plt_sv_dir = "../2_pipeline/tmp/"
        p = 0
        
    while( (np.linalg.norm(a.c_state-a.g_state)>th) and timeout>0):
        a.pred_controls()
        for i in range(u_horizon):
            a.v = a.vg[i]
            a.w = a.wg[i]
            a.v_list.append(a.v)
            a.x_traj = []
            a.y_traj = []
            a.get_traj(i)
            a.non_hol_update()
            # if(not rec_video):
            #     clear_output(wait=True)
            x,y = a.draw_circle()
            plt.plot(x,y,'b')
            plt.scatter(a.g_state[0],a.g_state[1],marker='x', color='r')
            plt.scatter(a.x_traj, a.y_traj,marker='.', color='r', s=1)
            plt.plot([a.c_state[0],a.g_state[0]],[a.c_state[1],a.g_state[1]], linestyle='dotted', c='k')
            
            # plt.show()
            plt.xlim([-10,100])
            plt.ylim([-10,100])
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=100)
                p = p+1
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