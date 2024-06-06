import gym
import surrol.gym
import cv2
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--env', default="NeedleReach-v0")
args = parser.parse_args()
env = gym.make(args.env)

obs = env.reset()
rgb = env.render()

print(obs)

done = False
while not done:
    action = env.get_oracle_action(obs)
    obs, reward, done, info = env.step(action)
    # print(obs, reward, done, info)
    img = env.render(mode="rgb_array")
    print(action, done, info)
    while True:
        
        img = cv2.resize(img, 
                            (1000, 1000),
                            interpolation=cv2.INTER_NEAREST)
        cv2.imshow('preview', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(0)
        cv2.setWindowTitle(
            'preview', 'press n to continue, q to quit')
        if k & 0xFF == ord('q'):    # Esc key to stop
            done = True
            break
        elif k & 0xFF == ord('n'):
            break
        else:
            continue
    