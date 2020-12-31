import pygame
import time
import math
import numpy as np
from ga import GeneticAlgorithm
from snake import Snake
from food import Food
from nn import NeuralNetwork
from colors import *
 
pygame.init()
 
window_width = 1300
window_height = 600
game_width = 800
game_height = window_height
game_x = window_width - game_width
epsilon = 0.001
input_shape = (24,16)
hidden_shape = []
output_shape = (16,3)
 
dis = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Snake Game')
 
clock = pygame.time.Clock()
 
snake_block = 10
clock_speed = 15
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
sensors_font = pygame.font.SysFont("comicsansms", 20)
 
 
def draw_score(best, gen, sensors = []):
    reset_score()
    gen = score_font.render(f'Gen: {gen}', True, white)
    value = score_font.render("Your Score: " + str(best.size), True, white)
    """ for i,s in enumerate(sensors):
        sensors_text = sensors_font.render(f'Sensor {i}: {s}', True, white)
        dis.blit(sensors_text, [0, i * 20 + 70]) """

    score_width = window_width - game_width - 40
    neural_activations = best.neural_activations

    # , *[x[0] for x in hidden_shape], output_shape(0)
    layers = [input_shape[0], *[x[0] for x in hidden_shape], output_shape[0], output_shape[1]]

    circles_layer = []
    width_per_layer = (score_width/len(layers))
    for z, n in enumerate(layers):
        circles = []
        circle_radius = (window_height - 50) / (n * 3)
        if circle_radius > 15:
            circle_radius = 15

        circle_margin = circle_radius / 1.5

        circle_count = (n * (circle_radius * 2 + circle_margin)) / 2
        base_y = 50 + (game_height / 2 - circle_radius) - circle_count
        for i in range(n):
            circles.append((20 + (z * width_per_layer + (width_per_layer / 2)), base_y + ((circle_radius * 2 + circle_margin) * i), circle_radius))
        circles_layer.append(circles)

    circles_activeds = []
    for (w,v) in reversed(neural_activations):
        activations = np.full((len(v),), 0)
        activations[np.argmax(v)] = 1
        circles_activeds.append(activations)

    activations = np.full((len(sensors),), 0)
    activations[np.argmax(sensors)] = 1
    circles_activeds.append(activations)
    circles_activeds.reverse()
    
    for i in range(len(circles_layer) - 1):
        for j,(x1,y1,_) in enumerate(circles_layer[i]):
            for z,(x2,y2,_) in enumerate(circles_layer[i + 1]):
                actived_line = circles_activeds[i][j] > 0 and circles_activeds[i + 1][z] > 0 
                pygame.draw.line(dis, green if actived_line else black, (x1,y1), (x2,y2), 1)

    for i, circles in enumerate(circles_layer):
        for j, (x,y,r) in enumerate(circles):
            pygame.draw.circle(dis,(white if i == 0 or circles_activeds[i][j] == 0 else green),(x,y),r)

    dis.blit(gen, [0, 0])
    dis.blit(value, [0, 30])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [game_x + game_width / 6, game_height / 3])


map_predict = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
epsilon = 0.01
reward_eat = 1000
colision_body = True
colision_walls = True

def gameLoop(population_size = 10, num_generations = 100):
    game_over = False
    exit_game = False
    only_plays = False
    infinite_plays = True
    draw_all = True

    ga = GeneticAlgorithm(Snake, population_size=population_size, num_generations=num_generations)
    population = ga.generate_population(
        x=game_width/2, 
        y=game_height/2, 
        reward_eat=reward_eat, 
        colision_body=colision_body, 
        colision_walls=colision_walls, 
        epsilon=epsilon, 
        map_output=map_predict, 
        board_height=game_height, 
        board_width=game_width, 
        width=snake_block, 
        height=snake_block, 
        game_x=game_x, 
        game=dis,
        random_action=0.0,
        brain_kwargs={"input_shape": (24, 16), "hidden_shapes": [], "output_shape": (16, 3), "continue_mutate_probability": 0.0, "continue_crossover_probability": 0.0})
 
    while not ga.is_last_generation():
        game_over = False
        
        while not game_over:
            reset_game()
            if only_plays:
                events = list(filter(
                    lambda e: e.type == pygame.KEYDOWN,
                    pygame.event.get()
                ))

                for event in events:
                    if event.key == pygame.K_ESCAPE:
                        exit_game = True
                        break
                
            game_over = len(list(filter(lambda x: x.loses == False, population))) == 0

            if game_over:
                population = ga.next_generation()          
                break

            best_idx = ga.cal_pop_fitness()[0][1]
            for indv in list(filter(
                lambda x: x.loses == False,
                population)):

                if only_plays:
                    indv.move(events)
                else:
                    indv.think()
                if not indv.loses and draw_all:
                    indv.draw(True)
            
            best = population[best_idx]
            if not draw_all:
                best.draw(True)
            sensors = best.get_sensors()
            draw_score(best, ga.current_gen, sensors)

            pygame.display.update()

            clock.tick(clock_speed if not only_plays else clock_speed / 1.35)

        if only_plays or exit_game:
            break
    
    best_idx = ga.cal_pop_fitness()[0]
    if not only_plays:
        message(f'Melhor indíviduo: {best_idx[1]}{best_idx[0]}')
 
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