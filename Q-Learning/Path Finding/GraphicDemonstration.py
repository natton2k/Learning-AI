from graphics import *
import time


class Button:
    def __init__(self, win, shape=Rectangle(Point(0, 0), Point(0, 0)), symbol=''):
        self.shape = shape
        self.symbol = symbol
        self.win = win
        self.text = Text(self.shape.getCenter(), self.symbol)
        self.text.setSize(36)

    def drawButton(self, color):
        self.shape.setFill(color)
        self.shape.draw(self.win)
        self.text.draw(self.win)

    def undrawButton(self):
        self.shape.undraw()
        self.text.undraw()


class InfoValue:
    def __init__(self, point=Point(0, 0)):
        self.point = point
        self.text = Text(point, '')
        self.text.setSize(25)

    def drawText(self, win, text):
        self.text = Text(self.point, text)
        self.text.draw(win)

    def undrawText(self):
        self.text.undraw()


class Demonstration:
    def __init__(self,time):
        self.win = GraphWin('Qlearning', 1000, 500)
        self.episode = 0
        self.average_reward = 0.0
        self.grid_outline = []
        self.button1 = Button(self.win, Rectangle(Point(112, 270), Point(168, 326)), '^')
        self.button2 = Button(self.win, Rectangle(Point(50, 332), Point(106, 388)), '<')
        self.button3 = Button(self.win, Rectangle(Point(112, 394), Point(168, 450)), 'v')
        self.button4 = Button(self.win, Rectangle(Point(174, 332), Point(230, 388)), '>')
        self.reward = InfoValue()
        self.episode = InfoValue()
        self.time = time

    def drawOutline(self):
        info_outline = Rectangle(Point(30, 30), Point(250, 470))
        info_outline.setFill('white')
        info_outline.draw(self.win)

        environment_outline = Rectangle(Point(280, 30), Point(970, 470))
        environment_outline.setFill('white')
        environment_outline.draw(self.win)

        rec3 = Rectangle(Point(295, 40), Point(955, 460))
        rec3.setFill('blue')
        # rec3.draw(win)

        button_space = Rectangle(Point(45, 265), Point(235, 455))
        button_space.setFill('red')
        button_space.draw(self.win)

    def drawButton(self, color_list=['green' for _ in range(4)]):
        self.button1.drawButton(color_list[0])
        self.button2.drawButton(color_list[1])
        self.button3.drawButton(color_list[2])
        self.button4.drawButton(color_list[3])

    def undrawButton(self):
        self.button1.undrawButton()
        self.button2.undrawButton()
        self.button3.undrawButton()
        self.button4.undrawButton()

    def drawInfoTable(self):
        episode_title_space = Rectangle(Point(45, 40), Point(235, 70))
        episode_title = Text(episode_title_space.getCenter(), 'Episode:')
        episode_title.setSize(15)
        episode_title.draw(self.win)

        episode_value_space = Rectangle(Point(45, 75), Point(235, 115))
        episode_value_space.setFill('green')
        episode_value_space.draw(self.win)
        self.episode.point = Rectangle(Point(45, 75), Point(235, 115)).getCenter()
        self.episode.drawText(self.win, '0')

        reward_title_space = Rectangle(Point(45, 135), Point(235, 165))
        reward_title = Text(reward_title_space.getCenter(), 'Average reward:')
        reward_title.setSize(15)
        reward_title.draw(self.win)

        reward_value_space = Rectangle(Point(45, 170), Point(235, 210))
        reward_value_space.setFill('green')
        reward_value_space.draw(self.win)
        self.reward.point = reward_value_space.getCenter()
        self.reward.drawText(self.win, '0.0')

    def updateInfoValue(self, episode=0, average_reward=0.0):
        self.reward.undrawText()
        self.episode.undrawText()

        self.episode.drawText(self.win, str(episode))
        self.reward.drawText(self.win, str(average_reward))

    def initiateGridOutline(self, row_num, col_num, reward_matrix):
        grid_outline = []
        y = 40 + 30 * ((14 - row_num) // 2)
        for i in range(0, col_num):
            x = 295 + 30 * ((23 - col_num) // 2)
            for j in range(0, row_num):
                state = Rectangle(Point(x, y), Point(x + 30, y + 30))
                grid_outline.append(state)
                x += 30
            y += 30
        for count, state in enumerate(reward_matrix):
            if state == -100:
                grid_outline[count].setFill('grey')
            elif state == 100:
                grid_outline[count].setFill('yellow')
            else:
                grid_outline[count].setFill('blue')
        self.grid_outline = grid_outline

    def drawStructure(self):
        self.drawOutline()
        self.drawButton()
        self.drawInfoTable()

    def drawDemonstration(self, row_num, col_num, reward_matrix=[]):
        self.drawStructure()
        self.grid_outline = self.initiateGridOutline(row_num, col_num, reward_matrix)

    def close(self):
        self.win.getMouse()
        self.win.close()

    def drawCanvas(self, current_state, action):
        for state in self.grid_outline:
            state.undraw()
        self.undrawButton()

        self.grid_outline[current_state].setFill('pink')
        for state in self.grid_outline:
            state.draw(self.win)
        button_color_list = ['green' for _ in range(4)]
        button_color_list[action] = 'yellow'
        self.drawButton(button_color_list)
        time.sleep(self.time)
        self.grid_outline[current_state].setFill('blue')


if __name__ == "__main__":
    pass