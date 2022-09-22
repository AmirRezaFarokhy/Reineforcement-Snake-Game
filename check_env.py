from snakeenv import SnekEnv

env = SnekEnv()
episodes = 50
for episode in range(episodes):
    done = False
    env.reset()
    while True:
        random_action = env.action_space.sample()
        print(f"action {random_action}")
        obs, reward, done, info = env.step(random_action)
        print(f"reward {reward}")

