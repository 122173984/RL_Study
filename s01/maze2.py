import numpy as np
import time
import tkinter as tk
from copy import deepcopy

UNIT = 40   # pixels
INDENT = 5
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Maze2')
        
        self.action_names = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_names)
#         self.root.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        
        self.trap_init_loc = [[1,2],[2,1],[1,3],[3,2],[2,4],[3,4],[4,2],[5,2],[6,2],[6,3],[6,4],[6,5],[6,6],
                              [5,6],[4,6],[4,4],[3,6],[1,4],[1,5],[1,6],[1,7],[1,8],[2,8],[3,8],[4,8],[5,8]]
        self.exit_init_loc = [2,2]
        
        self.actor_init_loc = [0,0]
        self.actor_loc = deepcopy(self.actor_init_loc)
        self.actor = None
        
        self._build_maze()
        
    def _draw_grids(self):
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
            
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
    
    def _draw_trap(self):
        for x, y in self.trap_init_loc:
            self.canvas.create_rectangle(x * UNIT + INDENT, y * UNIT + INDENT, 
                                         (x + 1) * UNIT - INDENT, (y + 1) * UNIT - INDENT, 
                                         fill='black')
    
    def _draw_exit(self):
        x, y = self.exit_init_loc
        self.canvas.create_oval(x * UNIT + INDENT, y * UNIT + INDENT, 
                                (x + 1) * UNIT - INDENT, (y + 1) * UNIT - INDENT, 
                                fill='yellow')
    
    def _draw_actor(self):
        if self.actor is not None:
            self.canvas.delete(self.actor)
        x, y = self.actor_loc
        self.actor =  self.canvas.create_rectangle(x * UNIT + INDENT, y * UNIT + INDENT, 
                                                   (x + 1) * UNIT - INDENT, (y + 1) * UNIT - INDENT, 
                                                   fill='red')
    
    def _build_maze(self):
        
        
        self.canvas = tk.Canvas(self.root, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        self._draw_grids()
        self._draw_trap()
        self._draw_exit()
        self._draw_actor()

        # pack all
        self.canvas.pack()
        
        
    def reset(self):
        self.actor_loc = deepcopy(self.actor_init_loc)
        self._draw_actor()
        self.root.update()
        # return observation
        return str(self.actor_loc)

    def step(self, action):
        x, y = self.actor_loc
        if action == 0:   # up
            if y > 0:
                y -= 1
        elif action == 1:   # down
            if y < MAZE_H - 1:
                y += 1
        elif action == 2:   # right
            if x < MAZE_W - 1:
                x += 1
        elif action == 3:   # left
            if x > 0:
                x -= 1
        
        self.actor_loc = [x, y]
        self._draw_actor()
        self.root.update()
        
        # reward function
        if self.actor_loc == self.exit_init_loc:
            reward = 1
            done = True
        elif self.actor_loc in self.trap_init_loc:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return str(self.actor_loc), reward, done

    def show(self):
        self.root.mainloop()

def update():
    for t in range(10):
        s = env.reset()
        while True:
            time.sleep(0.1)
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.show()