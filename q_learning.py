# Import the necessary libraries

from jinja2 import environment
import numpy as np
import random
from IPython.display import clear_output
import gym
from pyparsing import alphas

# Then we need to create an environment
environment = gym.make("Taxi-v3").env
environment.render()

print("Number of states: {}".format(environment.observation_space.n)) # Here we have 500 states
print("Number of actions: {}".format(environment.action_space.n)) # Here we have 6 actions

# Now we can proceed with the training of the agent. 
# Let's first initialize necessary variables
alpha = 0.1
gamma = 0.6
epsilon = 0.1
q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

# Then we run the training
num_of_episodes = 100000

for episode in range(0, num_of_episodes):
    # Reset the environment
    state = environment.reset()

    # Initilize variables
    reward = 0
    terminated = False

    while not terminated:
        # Take learned path pr explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, info = environment.step(action)

        # Recalculate
        q_value = q_table[state, action]
        max_value = np.max(q_table[next_state])
        new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

        # Update Q-table
        q_table[state, action] = new_q_value
        state = next_state
    if (episode + 1 ) % 100 == 0:
        clear_output(wait=True)
        print("Episode : {}".format(episode + 1))
        environment.render()
print("*********************************")
print("Training is done! \n")
print("********************************")


# Finally, we can evaluate the model we trained:
total_epochs = 0
total_penalties = 0
num_of_episodes = 100

for _ in range(num_of_episodes):
    state = environment.reset()
    epochs = 0
    penalties = 0
    reward = 0
    terminated = False

    while not terminated:
        action = np.argmax(q_table[state])
        state, reward, terminated, info = environment.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("********************")
print("The Results")
print("********************")
print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
print("Penalties per episode: {}".format(total_penalties / num_of_episodes))