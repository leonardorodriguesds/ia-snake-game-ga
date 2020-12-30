import pygame
import time
import math
import numpy as np

from ga import cal_pop_fitness, select_mating_pool
from snake import Snake
from food import Food
from nn import NeuralNetwork
from colors import *
 
pygame.init()
 
window_width = 1200
window_height = 600
game_width = 800
game_height = window_height
game_x = window_width - game_width
epsilon = 0.001
 
dis = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Snake Game')
 
clock = pygame.time.Clock()
 
snake_block = 10
clock_speed = 15
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
sensors_font = pygame.font.SysFont("comicsansms", 20)
 
 
def draw_score(score, gen, sensors = []):
    reset_score()
    gen = score_font.render(f'Gen: {gen}', True, white)
    value = score_font.render("Your Score: " + str(score), True, white)
    for i,s in enumerate(sensors):
        sensors_text = sensors_font.render(f'Sensor {i}: {s}', True, white)
        dis.blit(sensors_text, [0, i * 20 + 70])

    dis.blit(gen, [0, 0])
    dis.blit(value, [0, 30])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [game_width / 6, game_height / 3])


map_predict = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
epsilon = 0.01
new_snake = 0.1
reward_eat = 1000
colision_body = True
colision_walls = True
mutate_ratio = 0.01

def gameLoop(nSnakes = 400, max_gen = 1000):
    game_over = False
    game_close = False
    exit_game = False
    only_plays = False

    num_parents = round(nSnakes / 2)

    population = []
    for i in range(nSnakes):
        brain = NeuralNetwork(input_shape=(24, 32), hidden_shape=(32, 32), output_shape=(32, 4), mutate_ratio=mutate_ratio)
        snake = Snake(game_width/2, game_height/2, reward_eat=reward_eat, colision_body=colision_body, colision_walls=colision_walls, epsilon=epsilon, brain=brain, map_output=map_predict, board_height=game_height, board_width=game_width, width=snake_block, height=snake_block, game_x=game_x, game=dis)
        population.append(snake)
 
    for i in range(max_gen):
        game_over = False

        if exit_game:
            break

        while not game_over and not exit_game:
            reset_game()
            events = list(filter(
                lambda e: e.type == pygame.KEYDOWN,
                pygame.event.get()
            ))
            
            game_close = len(list(filter(lambda x: x.loses == False, population))) == 0

            if game_close:
                if only_plays:
                    gameLoop()
                
                reset_game()
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
                if only_plays:
                    snake.move(events)
                else:
                    snake.think()
                if not snake.loses:
                    snake.draw(True)
            
            best = population[0]
            # best.draw(True)
            sensors = best.get_sensors()
            draw_score(best.fitness, i, sensors)

            pygame.display.update()

            clock.tick(clock_speed if not only_plays else clock_speed / 1.35)

        if only_plays:
            break

        fitness = cal_pop_fitness(population)
        parents = select_mating_pool(population, fitness, num_parents)
        new_population = []
        childs = []

        print(f'top 10 gen {i}:\n')
        for j,s in enumerate(parents[:10]):
            print(f'{j + 1}º: {s.fitness}')

        for j in range(nSnakes - num_parents):
            idx1 = j % (num_parents)
            idx2 = (j + 1) % (num_parents)
            child = population[idx1].crossover(population[idx2])
            childs.append(child)

        new_population = [*parents, *childs]
        mutated_population = []

        for c in new_population:
            mutated_population.append(c.mutate())


        new_snakes = []
        for _ in range(150):
            if (np.random.uniform(0,1,1)[0] < new_snake):
                brain = NeuralNetwork(input_shape=(24, 32), hidden_shape=(32, 32), output_shape=(32, 4), mutate_ratio=mutate_ratio)        
                new_snakes.append(Snake(game_width/2, game_height/2, reward_eat=reward_eat, colision_body=colision_body, colision_walls=colision_walls, brain=brain, epsilon=epsilon, map_output=map_predict, board_height=game_height, board_width=game_width, width=snake_block, height=snake_block, game_x=game_x, game=dis))

        mutated_population[nSnakes - len(new_snakes):nSnakes] = new_snakes

        for i,snake in enumerate(mutated_population):
            snake = mutated_population[i]
            snake.loses = False
            snake.x = game_width / 2
            snake.y = game_height / 2
            snake.liveOn = 0
            snake.timeToDeath = 200
            mutated_population[i] = snake

        population = mutated_population

 
    pygame.quit()
    quit()


def reset_window():
    reset_score()
    reset_game()

def reset_game():
    pygame.draw.rect(dis, black, [game_x, 0, window_width, window_height])

def reset_score():
    pygame.draw.rect(dis, red, [0, 0, game_x - 1, window_height])

reset_window()
gameLoop()