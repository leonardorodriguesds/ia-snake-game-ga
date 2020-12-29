import pygame
import time
import random
import math
import tensorflow as tf
import numpy as np
 
pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
lightblue = (51, 255, 255)
pink = (255, 51, 153)
orange = (255, 178, 102)
 
dis_width = 800
dis_height = 600
epsilon = 0.001
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Edureka')
 
clock = pygame.time.Clock()
 
snake_block = 10
clock_speed = 15
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
sensors_font = pygame.font.SysFont("comicsansms", 20)
 
 
def draw_score(score, gen, sensors = []):
    gen = score_font.render(f'Gen: {gen}', True, yellow)
    value = score_font.render("Your Score: " + str(score), True, yellow)
    for i,s in enumerate(sensors):
        sensors_text = sensors_font.render(f'Sensor {i}: {s}', True, yellow)
        dis.blit(sensors_text, [0, i * 20 + 70])

    dis.blit(gen, [0, 0])
    dis.blit(value, [0, 30])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])


map_predict = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
mutate_ratio = 0.2
random_action = 0.05
epsilon = 0.01

class NeuralNetwork:
    def __init__(self, model = None, map_output=[], input_shape = None, hidden_shape = None, output_shape = None, dropout_percentage=0.02):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.dropout_percentage = dropout_percentage
        self.map_output = map_output
        self.activations = ['relu', 'sigmoid', 'softmax']
        if model == None:
            model = self.create_model()
        self.model = model

    def copy(self):
        model = self.create_model()
        return NeuralNetwork(
            model=model,
            map_output=self.map_output,
            input_shape=self.input_shape,
            hidden_shape=self.hidden_shape,
            output_shape=self.output_shape,
            dropout_percentage=self.dropout_percentage)

    def mutate(self):
        for z,l in enumerate(self.model):
            for i,r in enumerate(l):
                for j in range(len(r)):
                    if (np.random.uniform(0,1,1)[0] < mutate_ratio):
                        self.model[z][i][j] = np.random.uniform(-1,1,1)[0]
        return self

    def crossover(self, other):
        i = round(np.random.uniform(0,2,1)[0])

        self.model[i] = other.model[i]
        self.activations[i] = other.activations[i]

        return self


    def predict(self, sensors):
        L1, L2, L3 = self.model
        A1, A2, A3 = self.activations

        X = np.array(sensors)
        Z1 = np.matmul(L1, X)
        O1 = getattr(self, A1)(Z1)
        Z2 = np.matmul(L2, O1)
        O2 = getattr(self, A2)(Z2)
        Z3 = np.matmul(L3, O2)
        Y = getattr(self, A3)(Z3)
        return self.map_output[np.argmax(Y)]

    def relu(self, values):
        values[values < 0] = 0
        return values

    def sigmoid(self, values):
        return np.array(list(map(
            lambda x: 1 / (1 + math.exp(-x)),
            values
        )))

    def softmax(self, values):
        e_x = np.exp(values - np.max(values))
        return e_x / e_x.sum()

    def create_model(self):
        L1 = np.random.uniform(-1, 1, size=(self.input_shape[1], self.input_shape[0]))
        L2 = np.random.uniform(-1, 1, size=(self.hidden_shape[1], self.hidden_shape[0]))
        L3 = np.random.uniform(-1, 1, size=(self.output_shape[1], self.output_shape[0]))
        model = [L1,L2,L3]

        return model

class Snake:
    def __init__(self, x, y, size = 1, width=snake_block,height=snake_block,speed=1,board_width=dis_width,board_height=dis_height,brain=None):
        self.x, self.y, self.size = x, y, size

        self.pos = [(self.x, self.y)]
        self.width, self.height=width, height
        self.speed, self.speed_x, self.speed_y=speed, 0, 0
        self.fitness = 0
        self.board_width, self.board_height = board_width, board_height
        self.sensors_size = self.width * 15
        self.sensors_angles = [0, math.radians(90), math.radians(180), math.radians(270), math.radians(45), math.radians(135), math.radians(225), math.radians(315)]
        self.sensors_detect = [0,0,0,0,0,0,0,0]
        self.sensors_color = (red,green,yellow)
        self.loses = False
        self.brain = brain
        self.liveOn = 0
        self.food = Food(round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0,round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
        if self.brain == None:
            self.brain = NeuralNetwork(input_shape=(len(self.sensors_angles) + 6, 8), hidden_shape=(8, 32), output_shape=(32, 4), map_output=map_predict)

    def copy(self):
        return Snake(dis_width / 2, dis_height / 2, brain=self.brain)

    def mutate(self):
        self.brain = self.brain.mutate()
        return self

    def crossover(self, other):
        brain = self.brain.crossover(other.brain)
        return Snake(dis_width / 2, dis_height / 2, brain=brain)

    def draw(self, drawSensors=False):
        if drawSensors:
            for i,angle in enumerate(self.sensors_angles):
                head_x = self.x + self.width / 2
                head_y = self.y + self.height / 2
                x = head_x + math.cos(angle) * self.sensors_size
                y = head_y + math.sin(angle) * self.sensors_size
                pygame.draw.line(dis, self.sensors_color[self.sensors_detect[i]], (head_x,head_y), (x,y), 1)

        for p in self.pos:
            pygame.draw.rect(dis, black, [p[0], p[1], self.width, self.height])

        self.food.draw()
        

    def reward(self, reward):
        self.fitness += reward
        self.size += 1

    def think(self):
        predict = self.brain.predict(self.get_sensors())
        if (np.random.uniform(0,1,1)[0] <= random_action):
            self._move(round(np.random.uniform(0,len(map_predict) - 1,1)[0]))
        else:
            self._move(predict)

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
        if len(self.pos) > self.size:
            del self.pos[0]

        if self.x == self.food.x and self.y == self.food.y:
            self.reward(1) 

        if self.food.time == 0 or (self.x == self.food.x and self.y == self.food.y):
            self.food = Food(round(random.randrange(0, dis_width - self.food.width) / 10.0) * 10.0,round(random.randrange(0, dis_height - self.food.height) / 10.0) * 10.0)

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
        def isBetween(x1, y1, x2, y2, x, y):
            ab = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
            ap = math.sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))
            pb = math.sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y))
            if(ab == ap + pb):
                return True

        """ def isBetween(x1, y1, x2, y2, x3, y3):
            crossproduct = (y3 - y1) * (x2 - x1) - (x3 - x1) * (y2 - y1)

            # compare versus epsilon for floating point values, or != 0 if using integers
            if abs(crossproduct) > epsilon:
                return False

            dotproduct = (x3 - x1) * (x2 - x1) + (y3 - y1)*(y2 - y1)
            if dotproduct < 0:
                return False

            squaredlengthba = (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)
            if dotproduct > squaredlengthba:
                return False

            return True """

        """ def isBetween(x1, y1, x2, y2, x3, y3):
            if (x2 != x1):
                slope = (y2 - y1) / (x2 - x1)
                pt3_on = (y3 - y1) == slope * (x3 - x1)
                pt3_between = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
                return pt3_on and pt3_between
            else:
                return (x3 == x2) and (min(y1, y2) <= y3 <= max(y1, y2)) """

        for i,angle in enumerate(self.sensors_angles):
            x1 = self.x + self.width / 2
            y1 = self.y + self.height / 2
            x2 = x1 + math.cos(angle) * self.sensors_size
            y2 = y1 + math.sin(angle) * self.sensors_size

            self.sensors_detect[i] = 2 if (x2 < 0 or x2 > self.board_width or y2 < 0 or y2 > self.board_height) else (1 if isBetween(x1, y1, x2, y2, self.food.x + self.food.width / 2, self.food.y + self.food.height / 2) else 0)

        return [
            self.x / (self.board_width), self.y / (self.board_height), 
            (self.board_width-self.x-self.width/2)  / (self.board_width), 
            (self.board_height - self.y - self.height / 2) / (self.board_height), 
            *list(map(
                lambda x: (x + 1) / (3),
                self.sensors_detect)), 
            self.pos[0][0]  / (self.board_width), 
            self.pos[0][1]  / (self.board_height)]

    def isCrashed(self):
        return self.x >= self.board_width or self.x < 0 or self.y >= self.board_height or self.y < 0 or self.pos[-1] in self.pos[:-1]


class Food:
    def __init__(self, x, y, width=snake_block,height=snake_block):
        self.x, self.y = x, y
        self.width, self.height=width, height
        self.time = 600

    def draw(self):
        self.time -= 1
        pygame.draw.rect(dis, green, [self.x, self.y, self.width, self.height])

def cal_pop_fitness(snakes):
    fitness_list = np.array(list(map(
        lambda s: s.fitness,
        snakes
    )))
    liveOn_list = np.array(list(map(
        lambda s: s.liveOn,
        snakes
    )))
    fitness = list(map(
        lambda s: (s.fitness - np.min(fitness_list) / (np.max(fitness_list) - np.min(fitness_list)) if np.max(fitness_list) - np.min(fitness_list) > 0 else 0 ) * 0.95 + ((((s.liveOn - np.min(liveOn_list)) / (np.max(liveOn_list) - np.min(liveOn_list))) if np.max(liveOn_list) - np.min(liveOn_list) > 0 else 0) * 0.05),
        snakes
    ))
    return np.array(fitness)


def select_mating_pool(pop, fitness, num_parents):

    parents = []

    for _ in range(num_parents):

        max_fitness_idx = np.where(fitness == np.max(fitness))

        max_fitness_idx = max_fitness_idx[0][0]

        parents.append(pop[max_fitness_idx])

        fitness[max_fitness_idx] = -99999999999

    return parents

def gameLoop(nSnakes = 16, max_gen = 200):
    game_over = False
    game_close = False
    exit_game = False
    only_plays = False

    num_parents = round(nSnakes / 2)

    population = []
    for i in range(nSnakes):
        snake = Snake(dis_width/2, dis_height/2)
        population.append(snake)
 
    for i in range(max_gen):
        game_over = False

        if exit_game:
            break

        while not game_over and not exit_game:
            dis.fill(blue)

            events = list(filter(
                lambda e: e.type == pygame.KEYDOWN,
                pygame.event.get()
            ))
            
            game_close = len(list(filter(lambda x: x.loses == False, population))) == 0

            if game_close:
                dis.fill(blue)
                message(f'Gen {i} gameover', red)
                pygame.display.update()
                time.sleep(1)
                game_over = True                
                break
    
            for event in events:
                if event.key == pygame.K_ESCAPE:
                    exit_game = True
                    break

            for snake in list(filter(
                lambda x: x.loses == False,
                population)):
                snake.move(events)
                if snake.isCrashed():
                    snake.loses = True
                    continue
            
                #snake.draw(True)
                snake.think()
                snake.move(events)
            
            best = population[0]
            best.draw(True)
            # sensors = best.get_sensors()
            draw_score(best.fitness, i, [])

            pygame.display.update()

            clock.tick(clock_speed)

        fitness = cal_pop_fitness(population)
        parents = select_mating_pool(population, fitness, num_parents)
        new_population = []
        childs = []
        for j in range(nSnakes - num_parents):
            idx1 = j % (num_parents)
            idx2 = (j + 1) % (num_parents)
            child = population[idx1].crossover(population[idx2])
            childs.append(child)

        mutated_childs = []
        clone_parents = []
        for p in parents:
            clone_parents.append(p.copy())

        for c in childs:
            mutated_childs.append(c.mutate())

        new_population = [*clone_parents, *mutated_childs]

        for i,snake in enumerate(new_population):
            snake = new_population[i]
            snake.loses = False
            snake.x = dis_width / 2
            snake.y = dis_height / 2
            snake.liveOn = 0
            new_population[i] = snake

        population = new_population

 
    pygame.quit()
    quit()
 
gameLoop()