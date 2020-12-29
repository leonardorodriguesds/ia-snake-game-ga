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
    def __init__(self, model = None, map_output=[], input_nodes = None, hidden_nodes = None, output_Nodes = None, dropout_percentage=0.02):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_Nodes = output_Nodes
        self.dropout_percentage = dropout_percentage
        self.map_output = map_output
        if model == None:
            model = self.create_model()
        self.model = model
        # self.model.summary()

    def copy(self):
        model = self.create_model()
        weights = self.model.get_weights()
        model.set_weights(weights)
        return NeuralNetwork(
            model=model,
            map_output=self.map_output,
            input_nodes=self.input_nodes,
            hidden_nodes=self.hidden_nodes,
            output_Nodes=self.output_Nodes,
            dropout_percentage=self.dropout_percentage)

    def mutate(self):
        weights = self.model.get_weights()
        for i,l in enumerate(weights):
            for j,w in enumerate(l):
                if random.uniform(0, 1) <= mutate_ratio:
                    weights[i][j] += np.random.normal(0, 0.1, 1)[0]
        self.model.set_weights(weights)
        return self

    def crossover(self, other):
        weights_1 = self.model.get_weights()
        weights_2 = other.model.get_weights()
        i, j = round(random.randrange(0, len(weights_1) - 1)), round(random.randrange(0, len(weights_1[0]) - 1))
        weights_1[i][j], weights_2[i][j] = weights_2[i][j], weights_1[i][j]
        model_1 = self.copy()
        model_1.model.set_weights(weights_1)
        return model_1


    def predict(self, sensors):
        x = np.asarray(sensors)
        x = x.reshape(-1, self.input_nodes[0])
        return self.map_output[np.argmax(self.model.predict(x))]

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_nodes, activation='relu', input_shape=self.input_nodes),
            tf.keras.layers.Dropout(self.dropout_percentage),
            tf.keras.layers.Dense(self.output_Nodes, activation='softmax')
        ])    
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
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
            self.brain = NeuralNetwork(input_nodes=(len(self.sensors_angles) + 6, ), hidden_nodes=32, output_Nodes=4, map_output=map_predict)

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
            self._move(np.random.uniform(0,len(map_predict),1)[0])
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
        lambda s: (s.fitness - np.min(fitness_list) / (np.max(fitness_list) - np.min(fitness_list)) if np.max(fitness_list) - np.min(fitness_list) > 0 else 0 ) * 0.95 + (((s.liveOn - np.min(liveOn_list)) / (np.max(liveOn_list) - np.min(liveOn_list))) * 0.05),
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

def gameLoop(nSnakes = 8, max_gen = 200):
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