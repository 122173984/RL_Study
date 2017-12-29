from s01.maze2 import Maze
from s01.RL_brain import QLearningTable
import threading
import time

def update():
    flag = False
    for episode in range(1000):
        
        status = env.reset()
        count = 0
#         time.sleep(0.01)
        
        while True:
            count += 1
            action = RL.choose_action(status)
            status_, reward, done = env.step(action)
            RL.learn(status, action, reward, status_)
            status = status_
            if flag:
                time.sleep(0.1)
            if done:
                if reward == 1:
                    print ('SUCCESS -', episode, count)
                    if count < 40:
                        flag = True
                else:
                    print ('FAIL    -', episode, count)
#                 print (RL.q_table)
                break
    
    
    print ('game over!')
    

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))    
    t = threading.Thread(target = update)
    t.setDaemon(True)
    t.start()
    env.show()
    
