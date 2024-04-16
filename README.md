# RL_Minesweeper
RL Agent on a Minesweeper game. The agent is trained using Q-learning and Deep Q-learning algorithms.
## ```minesweeper.py```
A simple implementation of Minesweeper with a GUI. 
You can initialize a game by calling ```Minesweeper()``` and start it with ```.reset()```. 
A step is done by calling ```.step(action)```. ```action``` is a number for a field which you want to reveal. 
The game ends when you reveal a field with a bomb or the last field without a bomb.
```.step()``` returns the new_state, reward and a flag if the game is done.
Also you can call ```.render()``` to show the current state of the game in a GUI window or ```.plot_inline()``` to show the game in the console.
### Rewards
| Action                                         | Reward |
|------------------------------------------------|--------|
| Progress (reveal a non revealed field)         | 0.1    |
| Lose (hit a bomb)                              | - 0.1  |
| Win                                            |  0.1   |
| Guess (reveal a field with no revealed around) | - 0.1  |
| No progress (already revealed)                 | - 0.1  |
## ```model.py```
The model is a Convolutional Neural Network (CNN) with **4 convolutional layers** and **2 fully connected layers.**<br>
It takes as input the current state of the game and outputs the Q-values for each possible action. We choose a kernel size of 5x5 for the convolutional layers and a padding of 2 to keep the input size the same and also include values from the border fields. 
We use ReLU as the activation function for the convolutional layers and the fully connected layers. Also we use two dropout layers with ```p = 0.2```to prevent overfitting.
## ``agent.py``
The agent is the **interface between the game and the model**. It handles the returns from the game logic and trains the model.
### Training
For a new game the first field is revealed to avoid a random guess. 
An action is chosen using an epsilon-greedy policy. The agent will choose a random action with probability epsilon and the action with the highest Q-value with probability 1-epsilon. Epsilon is decreased over time to make the agent more greedy.
The action is executed and the new state, reward and game status are stored in the replay memory.
The replay memory is used to sample random batches of experiences to train the model. Training is done using the Q-learning algorithm ([Explanation of Q-Learning](https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187)).
<br>We calculate the Q-value as follows: ```new_q = reward + gamma * max_future_q```, where ```max_future_q``` is the maximum Q-value for the next state.
A simple MSE is calculated between the predicted Q-values and the target Q-values.
## Usage
To train the model you can run ````agent.py````. A model will be saved if the model reaches a new high score of the win rate.
In ```agent.py``` the funtion ```train()``` is called to train the model. You call train with the following parameters:
- ````nrows````: Number of rows of the game board
- ````ncols````: Number of columns of the game board
- ````nmines````: Number of bombs on the game board
Game and model are created with the given parameters.
## Dependencies
- **Pygame**: The GUI is created with Pygame
- **Numpy**: States are stored as numpy arrays
- **Pytorch**: The model is implemented with Pytorch
- **random**: Epsilon-greedy policy and sampling of experiences
- **deque**: Replay memory
- **pandas**: Inline representation of the game


