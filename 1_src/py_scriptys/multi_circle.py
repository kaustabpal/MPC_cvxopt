import numpy as np 
from matplotlib import pyplot as plt 

def create_circle(x,y):
    length = 4.7 # meters
    width = 1.90 # meters
    radius = width/2
    th = np.arange(0,2*np.pi,0.01)
    xunit = radius * np.cos(th) + x
    yunit = radius * np.sin(th) + y
    return xunit, yunit

class viz_agent:
    def __init__(self,x,y,theta):

        self.length = 4.7 # meters
        self.width = 1.90 # meters
        self.radius = self.width/2
        self.x1 = 0
        self.y1 = 0 
        self.theta = theta

        self.x2 = 0
        self.y2 = 0

        self.x3 = 0
        self.y3 = 0

        def calc_pos(x,y, theta):
            R = np.matrix([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))], [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
            pos = np.array([[x],[y]])
            new_pos = R@pos 
            print(new_pos)
            return new_pos[0,0], new_pos[1,0]

        self.x2, self.y2 = calc_pos(self.x1+2*self.radius, self.y1, self.theta)
        self.x3, self.y3 = calc_pos(self.x1+4*self.radius, self.y1, self.theta)

        self.x1 = self.x1 + x
        self.y1 = self.y1 + y 
        self.x2 = self.x2 + x 
        self.y2 = self.y2 + y
        self.x3 = self.x3 + x 
        self.y3 = self.y3 + y

        

        

if __name__ == "__main__":
    a = viz_agent(3,3,30)
    cx1, cy1 = create_circle(a.x1, a.y1)
    cx2, cy2 = create_circle(a.x2, a.y2)
    cx3, cy3 = create_circle(a.x3, a.y3)

    plt.plot(cx1, cy1, 'r')
    plt.plot(cx2, cy2, 'r')
    plt.plot(cx3, cy3, 'r')
    plt.axis('equal')
    plt.show()


