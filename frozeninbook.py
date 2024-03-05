import gym
env = gym.make('FrozenLake-v1',render_mode="human")
numepisode = 10
timestep = 20
for i in range(numepisode):
  state = env.reset()
  print('time step 0 "')
  env.render()

  for i in range(timestep):
    
    randomAction=env.action_space.sample()

    obs, reward, terminated, truncated , info  = env.step(randomAction)
   
    env.render()
  
    if truncated:
     break
