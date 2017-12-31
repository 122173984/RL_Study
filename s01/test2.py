from s01.maze2 import Maze
from s01.RL_brain2 import DeepQNetwork
import time
import threading

def run_maze():
    step = 0
    
    for episode in range(3000):
        observation = env.reset()
        count = 0
        while True:
#             time.sleep(0.1)
            count += 1
            action = RL.choose_action(observation)
            
            observation_, reward, done = env.step(action)
            
            RL.store_transition(observation, action, reward, observation_)
            
            if step > 200 and step % 5 == 0:
                RL.learn()
            
            observation = observation_
            
            if done:
                if reward == 1:
                    print ('SUCCESS -', episode, count)
                else:
                    print ('FAIL    -', episode, count)
#                 print (RL.q_table)
                break
            
            step += 1
    
    print('game over!')
    


if __name__ == '__main__':
    env = Maze()
    
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate = 0.01,
                      reward_decay = 0.9,
                      e_greedy = 0.9,
                      replace_target_iter = 200,
                      memory_size = 2000)
    
    t = threading.Thread(target = run_maze)
    t.setDaemon(True)
    t.start()
    env.show()

