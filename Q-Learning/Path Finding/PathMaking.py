import random


def path_making(row_num=3, column_num=3, reward=1, hole=1):
    path = []
    for row in range(row_num):
        path.append([0 for _ in range(column_num)])
    i =0
    while i < reward:
        x = random.randint(0, row_num - 1)
        y = random.randint(0, column_num - 1)
        if x == 0 and y == 0:
            continue
        if path[x][y] == 0:
            path[x][y] = 1
            i += 1

    i = 0
    while i < hole:
        x = random.randint(0, row_num-1)
        y = random.randint(0, column_num-1)
        if x == 0 and y == 0:
            continue
        if path[x][y] == 0:
            path[x][y] = -1
            i += 1
    file = open('path.txt', mode='w')
    for row in range(row_num):
        for col in range(column_num-1):
            file.write('{} '.format(path[row][col]))
        file.write('{}\n'.format(path[row][column_num-1]))
    file.close()

if __name__ == '__main':
    path_making()
