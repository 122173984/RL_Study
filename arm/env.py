import numpy as np
import tkinter as tk
import time
import threading


class ArmEnv(object):
    
    def __init__(self):
        
        self.height = 400
        self.width = 400
        
        self.dt = .1    # refresh rate
        self.action_bound = [-1, 1]
        self.state_dim = 9
        self.action_dim = 2
    
        self.center_coord = np.array([200,200])
        self.bar_thc = 5
        self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)]) #生成arm信息，包含2列，第一列为长度（l），第二列为角度（r）
        self.arm_info['l'] = 100        # 2 arms length 
        self.arm_info['r'] = np.pi/6    # 2 angles information 
        
        self.goal_info = {'x': 100., 'y': 100., 'l': 40}
        self.on_goal = 0

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound) # 注意action是2个弧度值
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal_info['x'] - a1xy_[0]) / 400, (self.goal_info['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal_info['x'] - finger[0]) / 400, (self.goal_info['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)

        # done and reward
        if self.goal_info['x'] - self.goal_info['l']/2 < finger[0] < self.goal_info['x'] + self.goal_info['l']/2:
            if self.goal_info['y'] - self.goal_info['l']/2 < finger[1] < self.goal_info['y'] + self.goal_info['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        
        return s, r, done

    def reset(self):
        self.goal_info['x'] = np.random.rand() * self.width 
        self.goal_info['y'] = np.random.rand() * self.height
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2) #生成2个随机角度ֵ
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord  # a1 start (x0, y0) arm1的关节坐标
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1) arm2的关节坐标
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2) arm的指尖
        # normalize features
        dist1 = [(self.goal_info['x'] - a1xy_[0])/400, (self.goal_info['y'] - a1xy_[1])/400]
        dist2 = [(self.goal_info['x'] - finger[0])/400, (self.goal_info['y'] - finger[1])/400]
        # state 总共9个值
        # 1. arm2的关节坐标
        # 2. finger的坐标
        # 3. amr2关节到目标的距离值（x，y），指尖到目标的距离值(x,y)
        # 4. 是否on_goal
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.])) 
        
        return s

    def set_goal(self, x, y):
        self.goal_info = {'x': x, 'y': y, 'l': 40}

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(object):
    def __init__(self, env):
        
        self.env = env
        
        self.goal = None
        self.arm1 = None
        self.arm2 = None
        
        self.root = tk.Tk()
        self.root.title('Robot Arm')
        
        self.canvas = tk.Canvas(self.root, bg='white', height= self.env.height, width=self.env.width)
        self.canvas.pack()
        
        self._draw_goal()
        self._draw_arm()
        
        self.root.bind('<Motion>', self.motion)
        
    
    def _draw_goal(self):
        if self.goal is not None:
            self.canvas.delete(self.goal)
        
        goal_info = self.env.goal_info
        
        self.goal = self.canvas.create_polygon( [goal_info['x'] - goal_info['l'] / 2, goal_info['y'] - goal_info['l'] / 2,
                                                 goal_info['x'] - goal_info['l'] / 2, goal_info['y'] + goal_info['l'] / 2,
                                                 goal_info['x'] + goal_info['l'] / 2, goal_info['y'] + goal_info['l'] / 2,
                                                 goal_info['x'] + goal_info['l'] / 2, goal_info['y'] - goal_info['l'] / 2], fill = 'blue')
    
    def _draw_arm(self):
        if self.arm1 is not None:
            self.canvas.delete(self.arm1)
        if self.arm2 is not None:
            self.canvas.delete(self.arm2)
        
        # update arm
        (a1l, a2l) = self.env.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.env.arm_info['r']     # radian, angle
        a1xy = self.env.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.env.arm_info['r'][0], np.pi / 2 - self.env.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.env.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.env.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.env.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.env.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.env.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.env.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.env.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.env.bar_thc
        
        arm1_points = list(np.concatenate((xy01, xy02, xy11, xy12)))
        arm2_points = list(np.concatenate((xy11_, xy12_, xy21, xy22)))
        
        self.arm1 = self.canvas.create_polygon(arm1_points, fill = 'red')
        self.arm2 = self.canvas.create_polygon(arm2_points, fill = 'red')
    
    def show(self):
        self.root.mainloop()
    
    def render(self):
        self._draw_goal()
        self._draw_arm()
        self.root.update()
    
    def motion(self, event):
        self.env.set_goal(event.x, event.y)
#         self._draw_goal()
#         self.root.update()
    

def update():
    while True:
        env.step(env.sample_action())
        view.render()
        time.sleep(0.1)
        


if __name__ == '__main__':
    env = ArmEnv()
    env.reset()
    view = Viewer(env)
    t = threading.Thread(target = update)
    t.setDaemon(True)
    t.start()
    view.show()
