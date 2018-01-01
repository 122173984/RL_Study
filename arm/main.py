"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from arm.env import ArmEnv, Viewer
from arm.rl import DDPG
import threading
import time

MAX_EPISODES = 900
MAX_EP_STEPS = 200


def train(env, rl):
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
#     rl.save()


def eval(env,rl, view):
    s = env.reset()
    while True:
        view.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)
        time.sleep(0.05)

def start_eval():
    # set env
    env = ArmEnv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    
    # set RL method (continuous)
    rl = DDPG(a_dim, s_dim, a_bound)
    rl.restore()
    
    view = Viewer(env)
    t = threading.Thread(target = eval, args=(env,rl, view))
    t.setDaemon(True)
    t.start()
    view.show()

def start_train():
    # set env
    env = ArmEnv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    
    # set RL method (continuous)
    rl = DDPG(a_dim, s_dim, a_bound)
    train(env, rl)
    rl.save()

if __name__ == '__main__':
    start_eval()
    
