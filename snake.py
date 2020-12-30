import math
import numpy as np
import pygame
import random
from colors import *
from food import Food

class Snake:
    def __init__(self, x, y, size = 1, width=10,height=10,speed=1,board_width=800,board_height=600,brain=None,game_x=0,game=None, epsilon = 0.01, random_action = 0.005, map_output = [], colision_body=True, colision_walls=True, reward_eat=1000):
        self.x, self.y, self.size = x, y, size

        self.pos = [(self.x, self.y)]
        self.width, self.height=width, height
        self.speed, self.speed_x, self.speed_y=speed, 0, 0
        self.fitness = 0
        self.board_width, self.board_height = board_width, board_height
        self.sensors_size = self.width * 15
        self.sensors_angles = [0, math.radians(90), math.radians(180), math.radians(270), math.radians(45), math.radians(135), math.radians(225), math.radians(315)]
        self.sensors_detect = []
        self.sensors_color = (red,green,yellow)
        self.loses = False
        self.brain = brain
        self.timeToDeath = 250
        self.game_x = game_x
        self.epsilon = epsilon
        self.liveOn = 0
        self.game = game
        self.map_output = map_output
        self.random_action = random_action
        self.colision_body = colision_body
        self.colision_walls = colision_walls
        self.reward_eat = reward_eat
        self.max_distance = math.sqrt( self.board_width * self.board_width + self.board_height * self.board_height ) + 1 + epsilon
        self.food = Food(round(random.randrange(0, self.board_width - self.width) / 10.0) * 10.0,round(random.randrange(0, self.board_height - self.height) / 10.0) * 10.0, width=self.width, height=self.height, game=self.game, game_x=self.game_x)

    def copy(self):
        return Snake(self.board_width / 2, self.board_height / 2, brain=self.brain)

    def mutate(self):
        self.brain = self.brain.mutate()
        return self

    def crossover(self, other):
        brain = self.brain.crossover(other.brain)
        return Snake(self.board_width / 2, self.board_height / 2, brain=brain)

    def get_line_equation(self):
        res = []
        for angle in self.sensors_angles:
            head_x = self.x + self.width / 2
            head_y = self.y + self.height / 2
            x,y = 0,0
            if (angle == 0):
                x = self.board_width
                y = head_y
            elif (angle == math.radians(90)):
                x = head_x
                y = self.board_height
            elif (angle == math.radians(180)):
                y = head_y
            elif (angle == math.radians(270)):
                x = head_x
            elif (angle == math.radians(45)):
                x = self.board_width
                y = self.board_height
            elif (angle == math.radians(135)):
                y = self.board_height
            elif (angle == math.radians(315)):
                x = self.board_width
            m = (y - head_y) / ((x - head_x) if (x != head_x) else 1)
            n = head_y - (m * head_x)
            res.append((x, y, m, n))
        return res

    def sensor_colision(self, line_equations, x,y,w,h):
        res = []
        for i,a in enumerate(self.sensors_angles):
            head_x = self.x + self.width / 2
            head_y = self.y + self.height / 2
            flag = 0
            calc_y = (x + (w / 2)) * line_equations[i][2] + line_equations[i][3]
            if (a == math.radians(270) and (y - h / 2 <= head_y + self.height / 2 + self.epsilon)
            and (head_x - self.width / 2 - self.epsilon <= x + w / 2 <= head_x + self.width / 2 + self.epsilon)):
                flag = 1
            elif (a == math.radians(90) and (head_y - self.height / 2 - self.epsilon <= y + h / 2)
            and (head_x - self.width / 2 - self.epsilon <= x + w / 2 <= head_x + self.width / 2 + self.epsilon)):
                flag = 1
            elif (a == math.radians(180) and (head_y - self.height / 2 - self.epsilon <= y + h / 2 <= head_y + self.height / 2 + self.epsilon)
            and (x + w / 2 <= head_x + self.width / 2 + self.epsilon)):
                flag = 1
            elif (a == 0 and (head_y - self.height / 2 - self.epsilon <= y + h / 2 <= head_y + self.height / 2 + self.epsilon)
            and (head_x - self.width / 2 - self.epsilon <= x + w / 2)):
                flag = 1
            elif ((calc_y - h / 2 - self.epsilon <= y + h / 2 <= calc_y + h / 2 + self.epsilon)
            and (
                (a in [math.radians(45), math.radians(315)] and (head_x - self.width / 2 - self.epsilon <= x + w / 2))
                or (a in [math.radians(135), math.radians(225)] and (x + w / 2 <= head_x + self.width / 2 + self.epsilon))
            )):
                flag = 1
            res.append(flag)
        return res



    def draw(self, drawSensors=False):
        if drawSensors:
            line_equations = self.get_line_equation()
            colisions = self.sensor_colision(line_equations, self.food.x, self.food.y, self.food.width, self.food.height)
            for i in range(len(self.sensors_angles)):
                head_x = self.game_x + self.x + self.width / 2
                head_y = self.y + self.height / 2
                pygame.draw.line(self.game, self.sensors_color[colisions[i]], (head_x,head_y), (self.game_x + line_equations[i][0],line_equations[i][1]), 1)

        for p in self.pos:
            pygame.draw.rect(self.game, white, [self.game_x + p[0], p[1], self.width, self.height])

        self.food.draw()
        

    def reward(self, reward, eat=False):
        self.fitness += reward
        if eat:
            self.size += 1
            self.timeToDeath = 200

    def think(self):
        predict = self.brain.predict(self.get_sensors())
        if (np.random.uniform(0,1,1)[0] <= self.random_action):
            self._move(round(np.random.uniform(0,len(self.map_output) - 1,1)[0]))
        else:
            self._move(self.map_output[predict])

    def _move(self, moviment):
        if moviment == pygame.K_LEFT and self.speed_x == 0:
            self.speed_x = -self.width * self.speed
            self.speed_y = 0
        elif moviment == pygame.K_RIGHT and self.speed_x == 0:
            self.speed_x = self.width * self.speed
            self.speed_y = 0
        elif moviment == pygame.K_UP and self.speed_y == 0:
            self.speed_y = -self.height * self.speed
            self.speed_x = 0
        elif moviment == pygame.K_DOWN and self.speed_y == 0:
            self.speed_y = self.height * self.speed
            self.speed_x = 0
            
        self.x += self.speed_x
        self.y += self.speed_y
        self.liveOn += 1
        self.pos.append((self.x,self.y))
        self.timeToDeath -= 1
        if len(self.pos) > self.size:
            del self.pos[0]

        self.handle_colision()

        if self.timeToDeath <= 0:
            self.loses = True

    def handle_colision(self):
        if not self.isCrashed():
            self.reward(1)

            if self.check_colision_walls():
                if self.x < 0 or self.x > self.board_width:
                    self.x = 0 if self.x > self.board_width else self.board_width
                if self.y < 0 or self.y > self.board_height:
                    self.y = 0 if self.y > self.board_height else self.board_height
        else:
            self.loses = True

        if self.x == self.food.x and self.y == self.food.y:
            self.reward(self.reward_eat, True) 

        if self.food.time == 0 or (self.x == self.food.x and self.y == self.food.y):
            self.food = Food(round(random.randrange(0, self.board_width - self.width) / 10.0) * 10.0,round(random.randrange(0, self.board_height - self.height) / 10.0) * 10.0, width=self.width, height=self.height, game=self.game, game_x=self.game_x)


    def move(self, events):
        moviment = None
        for event in events:
            if event.key == pygame.K_p:
                self.speed_y = 0
                self.speed_x = 0
            else:
                moviment = event.key

        self._move(moviment)

    def get_sensors(self):
        """ def isBetween(x1, y1, x2, y2, x, y):
            ab = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
            ap = math.sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))
            pb = math.sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y))
            if(ab == ap + pb):
                return True """

        self.sensors_detect = []
        x1 = self.x + self.width / 2
        y1 = self.y + self.height / 2
        x2 = self.food.x + self.food.width / 2
        y2 = self.food.y + self.food.height / 2
        line_equations = self.get_line_equation()

        # Food detection
        food_colisions = self.sensor_colision(line_equations, self.food.x, self.food.y, self.food.width, self.food.height)
        for i, c in enumerate(food_colisions):
            if c:
                self.sensors_detect.append(math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)) / self.max_distance)
            else:
                self.sensors_detect.append(1)

        # Tail detection
        if (self.size > 1):
            tx = self.pos[0][0] + self.width / 2
            ty = self.pos[0][1] + self.height / 2
            tail_colisions = self.sensor_colision(line_equations, self.pos[0][0], self.pos[0][1], self.width, self.height)
            for z, c in enumerate(tail_colisions):
                if c:
                    self.sensors_detect.append(math.sqrt(math.pow(tx - x1, 2) + math.pow(ty - y1, 2)) / self.max_distance)
                else:
                    self.sensors_detect.append(1)
        else:
            for z in range(len(self.sensors_angles)):
                self.sensors_detect.append(1)

        # Walls detection
        for z, (x,y, _, _) in enumerate(line_equations):
            i = len(self.sensors_angles) * 2 + z
            self.sensors_detect.append(math.sqrt(math.pow(x - x1, 2) + math.pow(y - y1, 2)) / self.max_distance)
        
        return self.sensors_detect

    def isCrashed(self):
        crash = False
        if self.colision_body:
            crash |= self.pos[-1] in self.pos[:-1]
        if self.colision_walls:
            crash |= self.check_colision_walls()
        return crash

    def check_colision_walls(self):
        return self.x >= self.board_width or self.x < 0 or self.y >= self.board_height or self.y < 0