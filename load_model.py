from stable_baselines3 import PPO, A2C 
from snakeenv import SnekEnv
import gym

model_dir = "models/1663165320"

env = SnekEnv()
env.reset()

model_path = f"{model_dir}/20000.zip"
model = PPO.load(model_path, env=env)

episodes = 500
for episode in range(episodes):
    obs = env.reset()
    done = False 
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)
