import numpy as np
import random
import PathMaking
import GraphicDemonstration as gp


def analyzeEnvironment(penalty):
    '''
    this function will read the path.txt and analyse all information of the environment from path.txt

    :param: penalty: the living-penalty for every step so that the agent will find the fastest route

    :return:
        reward_matrix: assgin rewards base on the penalty and the environment
        terminate_list: list of point where the agent step on the game is terminated
        (row_num, col_num): tuple of the size of the environment
    '''
    # read file as list of line
    with open('path.txt', mode='r') as f:
        lineList = [line.rstrip('\n') for line in f]
    # convert string list in to the machine-readable list
    maze = []
    row_num = len(lineList)
    for row in range(row_num):
        maze.append([int(x) * 100 for x in lineList[row].split()])
    col_num = len(maze[0])

    # create reward_matrix and terminate_list
    reward_matrix = []
    terminate_list = []
    for row in range(row_num):
        for col in range(col_num):
            if maze[row][col] == 0:
                reward_matrix.append(penalty)
            else:
                reward_matrix.append(maze[row][col])
                terminate_list.append(row * col_num + col)

    return reward_matrix, terminate_list, (row_num, col_num)


def getPosibleAction():
    '''
    this function will determine the all of the next posibles action for all states
    :return: posible_action: list of numbers indicating the directions
        0: moving up
        1: moving left
        2: moving down
        3: moving right
    '''
    posible_action = []
    # check if the agent can move up
    if current_state // size[1] != 0:
        posible_action.append(0)
    # check if the agent can move left
    if current_state % size[1] != 0:
        posible_action.append(1)
    # check if the agent can move down
    if current_state // size[1] != size[0] - 1:
        posible_action.append(2)
    # check if the agent can move right
    if (current_state + 1) % size[1] != 0:
        posible_action.append(3)
    return posible_action


def chooseAction(posible_action):
    '''
    this function will the choose a specific action from the list of posible_action
    this is an implementation of the 'epsiplon greedy algorithm'
    :param posible_action: list of posible action of a state
    :return: the action for the agent
    '''
    """
        the exploration_rate indicates the trade of between exploration and exploitation
        when as in the beginning the agent will always discovery 
        for time passed, the agent will exploit more
    """
    exploration_rate = minimum_discovery_rate + (1 - minimum_discovery_rate) * np.exp(-discovery_decay_rate * episode)
    while True:
        epsilon = random.random()
        if epsilon < exploration_rate:
            # exploration
            return random.choice(posible_action)
        else:
            # exploitation
            pos = np.argmax(q_table[current_state, :])
            # check whether the best action is in the posible action
            if pos not in posible_action:
                continue
            else:
                return np.argmax(q_table[current_state, :])


def getNextState():
    '''
    this function will determine the next state for the agent to take
    :return:
        action: the action by which the agent uses to reach the next state
        next_state: the indice of next state
        reward_matrix[next_state]: the reward of the environment of the next state
    '''
    posible_action = getPosibleAction()
    action = chooseAction(posible_action)
    next_state = current_state
    if action == 0:
        next_state -= size[1]
    elif action == 1:
        next_state -= 1
    elif action == 2:
        next_state += size[1]
    else:
        next_state += 1
    return action, next_state, reward_matrix[next_state]


def printPolicy():
    '''
    this function will print out the best policy of all states
    :return: none
    '''
    direction = dict(zip([0, 1, 2, 3], ['^', '<', 'v', '>']))
    for i in range(len(q_table)):
        if reward_matrix[i] == 100:
            print('A', end=' ')
        elif reward_matrix[i] == -100:
            print('X', end=' ')
        else:
            print(direction[np.argmax(q_table[i, :])], end=' ')
        if (i + 1) % size[1] == 0:
            print()


def printAverageReward():
    '''
    this function will print the Average reward per thousand episodes
    :return:
    '''
    rewards_per_thousand_episodes = np.split(np.array(episode_reward), maximum_episode / 1000)
    count = 1000
    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000


if __name__ == '__main__':
    row_num = 10
    col_num = 10
    PathMaking.path_making(row_num, col_num, 1, 10)
    penalty = -1
    reward_matrix, terminate_list, size = analyzeEnvironment(penalty)
    q_table = np.zeros((size[0] * size[1], 4))
    maximum_episode = 1000
    maximum_step = 10
    learning_rate = 0.1
    discount_rate = 0.9
    minimum_discovery_rate = 0.001
    discovery_decay_rate = 0.1
    episode_reward = []
    # graphic initalize
    time = 0.00001
    canvass = gp.Demonstration(time)
    canvass.drawStructure()
    canvass.initiateGridOutline(row_num, col_num, reward_matrix)
    # Q Learning
    for episode in range(maximum_episode):
        current_episode_reward = 0
        step = 0
        # if you want the agent to find the best policy from a fixed state
        # current_state = 0
        # if you want the agent to find the best policy in all state
        current_state = random.randint(0, size[0] * size[1] - 1)
        while step < maximum_step and current_state not in terminate_list:
            action, next_state, reward = getNextState()
            canvass.drawCanvas(current_state, action)
            q_table[current_state, action] = q_table[current_state, action] * (1 - learning_rate) + learning_rate * (
                    reward + discount_rate * np.max(q_table[next_state, :]))
            current_state = next_state
            current_episode_reward += reward_matrix[next_state]
            step += 1
            try:
                canvass.updateInfoValue(episode, sum(episode_reward) / len(episode_reward))
            except ZeroDivisionError:
                canvass.updateInfoValue(episode, 0)
        episode_reward.append(current_episode_reward)

    printPolicy()
    printAverageReward()
    canvass.close()
