import random
import math

environment_matrix = [0, -100, 0, 0, 0, 0, 0, 0, 100, 0]
q_table = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]


def getNextPosibleAction(current_positon):
    posible_action = []
    if current_position - 1 in range(10):
        posible_action.append(0)
    if current_position + 1 in range(10):
        posible_action.append(1)
    return posible_action


def isGameOver(current_position):
    return current_position in [1, 8]


def getNextPosition(choice, current_position):
    return current_position - 1 if choice == 0 else current_position + 1


def out_path(q_table):
    for i in range(len(q_table)):
        if q_table[i][0] > q_table[i][1]:
            print('<- ', end='')
        elif q_table[i][0] < q_table[i][1]:
            print('-> ', end='')
        else:
            print('x ', end='')


def getAction(episode, posible_action, current_position):
    e = 1 - (episode / 1000)
    properbility = random.random()
    if properbility > e:
        return random.choice(posible_action)
    else:
        max = 0
        max_action = 0
        for action in posible_action:
            if max <= q_table[current_position][action]:
                max = q_table[current_position][action]
                max_action = action
        return max_action


learning_rate = 0.1
esiplon = 0.9
for episode in range(1000):
    current_position = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    while not isGameOver(current_position):
        posible_action = getNextPosibleAction(current_position)
        action = getAction(episode, posible_action, current_position)
        next_position = getNextPosition(action, current_position)
        q_table[current_position][action] += learning_rate * (environment_matrix[next_position] +
                                                              esiplon * max(q_table[next_position]) -
                                                              q_table[current_position][action])
        current_position = next_position
        print(q_table)
out_path(q_table)
