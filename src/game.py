import time
import pygame
import numpy as np
import pickle
import os
import re
import argparse
from ga import GeneticAlgorithm
from snake import Snake
from colors import *
 
pygame.init()

window_width = 1300
window_height = 600
game_width = 800
game_height = 600
input_shape = (24,16)
hidden_shape = [(16,8)]
output_shape = (8,4)
game_x = window_width - game_width
    
game = None
clock = None

 
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

    if neural_activations:
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
        for (_,v) in reversed(neural_activations):
            activations = np.full((len(v),), 0)
            activations[np.argmax(v)] = 1
            circles_activeds.append(activations)

        activations = np.full((len(sensors),), 0)
        activations[np.argmin(sensors)] = 1
        circles_activeds.append(activations)
        circles_activeds.reverse()
        circles_activeds[0][:] = 0
        circles_activeds[0][np.argmin(sensors)] = 1
        
        for i in range(len(circles_layer) - 1):
            for j,(x1,y1,_) in enumerate(circles_layer[i]):
                for z,(x2,y2,_) in enumerate(circles_layer[i + 1]):
                    actived_line = circles_activeds[i][j] > 0 and circles_activeds[i + 1][z] > 0 
                    pygame.draw.line(game, green if actived_line else black, (x1,y1), (x2,y2), 1)

        for i, circles in enumerate(circles_layer):
            for j, (x,y,r) in enumerate(circles):
                pygame.draw.circle(game,(white if circles_activeds[i][j] == 0 else green),(x,y),r)

    game.blit(gen, [0, 0])
    game.blit(value, [0, 30])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    game.blit(mesg, [game_x + game_width / 6, game_height / 3])


map_predict = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]

def init(
    population_size = 300,
    num_generations = 500,
    play = False,
    sb = True,
    s = 10,
    epsilon = 0.01,
    colision_body = True,
    colision_walls = True,
    width = 1300,
    height = 600,
    g_width = 800,
    g_height = 600,
    nn_input_shape = (24,16),
    nn_hidden_shape = [(16,8)],
    nn_output_shape = (8,4),
    dir_name = 'saved_models',
    load_nn = True,
    continue_mutate_probability = 0.05,
    continue_crossover_probability = 0.01
):
    print(sb)
    global window_width, window_height, game_width, game_height, input_shape, hidden_shape, output_shape, game_x, game, clock
    window_width = width
    window_height = height
    game_width = g_width
    game_height = g_height 
    input_shape = nn_input_shape 
    hidden_shape = nn_hidden_shape
    output_shape = nn_output_shape
    game_x = window_width - game_width
    
    game = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Snake Game')
    
    clock = pygame.time.Clock()

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    nns = [[],[]]
    if load_nn:
        files = os.listdir(dir_name)
        regex = '(\d+)_nn_(\d+).pkl'
        result = [re.search(regex, file) for file in files]
        file_nns = sorted(list(map(
            lambda g: (g.group(0), int(g.group(1)), g.group(2)),
            result
        )), key=lambda x: x[1], reverse=True)

        for (nn,fitness,_) in file_nns[:s]:
            with open(os.path.join(dir_name, nn), "rb") as input_file:
                nns[0].append(fitness)
                nns[1].append(pickle.load(input_file))

    population_kwargs = {
        'x':game_width/2, 
        'y':game_height/2, 
        'colision_body':colision_body, 
        'colision_walls':colision_walls, 
        'epsilon':epsilon, 
        'map_output':map_predict, 
        'board_height':game_height, 
        'board_width':game_width, 
        'width':snake_block, 
        'height':snake_block, 
        'game_x':game_x, 
        'game':game,
        'random_action':0.015,
        'limit_moves':(not play),
        'brain_kwargs':{
            "input_shape": input_shape, 
            "hidden_shapes": hidden_shape, 
            "output_shape": output_shape, 
            "continue_mutate_probability": continue_mutate_probability, 
            "continue_crossover_probability": continue_crossover_probability
        }
    }
    ga = GeneticAlgorithm(Snake, population_size=population_size, num_generations=num_generations, population_kwargs=population_kwargs, sample=nns)
    gameloop_kwargs = {
        'ga': ga,
        'game_over': False,
        'exit_game': False,
        'play': play,
        'draw_all': True,
        'clock': clock
    }

    reset_window()
    population = gameLoop(**gameloop_kwargs)

    """ if not play:
        print(f'Melhor indíviduo: {best_idx[1]}{best_idx[0]}') """

    if sb:
        bests = ga.cal_pop_fitness()
        for (fitness, idx) in bests[:s]:
            best = population[idx]
            filename = f'{round(fitness)}_nn_{round(time.time() * 1000 + idx)}.pkl'

            obj = {
                "model" : best.brain.model,
                "activations" : best.brain.activations,
                "bias" : best.brain.bias,
                "random_weights_intensity" : best.brain.random_weights_intensity,
                "random_bias_intensity" : best.brain.random_bias_intensity,
                "dropout_intensity" : best.brain.dropout_intensity,
                "mutation_intensity" : best.brain.mutation_intensity,
                "input_shape" : best.brain.input_shape,
                "hidden_shapes" : best.brain.hidden_shapes,
                "output_shape" : best.brain.output_shape,
                "dropout_percentage" : best.brain.dropout_percentage,
                "crossover_probability" : best.brain.crossover_probability,
                "activations" : best.brain.activations,
                "continue_mutate_probability" : best.brain.continue_mutate_probability,
                "continue_crossover_probability" : best.brain.continue_crossover_probability 
            }
                
            with open(os.path.join(dir_name, filename), 'wb+') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
 
    pygame.quit()
    quit()

def gameLoop(
    ga = None, 
    play = False, 
    game_over = False, 
    exit_game = False,
    draw_all = True,
    clock = None
):    
    population = []
    while not ga.is_last_generation():
        population = ga.next_generation()
        game_over = False
        
        while not game_over:
            reset_game()
            events = []
            if play:
                events = list(filter(
                    lambda e: e.type == pygame.KEYDOWN,
                    pygame.event.get()
                ))

                for event in events:
                    if event.key == pygame.K_ESCAPE:
                        exit_game = True
                        break
                
            game_over = len(list(filter(lambda x: x.loses == False, population))) == 0

            _, best_idx = ga.cal_pop_fitness()[0]
            for indv in list(filter(
                lambda x: x.loses == False,
                population)):

                if play:
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

            clock.tick(clock_speed if not play else clock_speed / 1.35)

        if play or exit_game:
            break

    return population

def reset_window():
    reset_score()
    reset_game()

def reset_game():
    pygame.draw.rect(game, black, [game_x, 0, window_width, window_height])

def reset_score():
    pygame.draw.rect(game, red, [0, 0, game_x - 1, window_height])

def main():
    parser = argparse.ArgumentParser(description='IA Snake Game with GA and NN algorithm')
    parser.add_argument('-play', action='store_true', help='Desativar GA e NN para jogar.')
    parser.add_argument('-population_size', default=300, type=int, help='Tamanho da população.')
    parser.add_argument('-num_generations', default=50, type=int, help='Número de gerações.')
    parser.add_argument('-sb', default=True, action='store_true', help='Salvar os melhores indivíduo.')
    parser.add_argument('--s', default=20, type=int, help='Quantidade de invidíduos para salvar.')
    parser.add_argument('-epsilon', default=0.01, type=float, help='Margem de erro a ser considerada.')
    parser.add_argument('-colision_body', default=True, action='store_false', help='Desligar colisão com o corpo.')
    parser.add_argument('-colision_walls', default=True, action='store_false', help='Desligar colisão com as paredes.')
    parser.add_argument('-width', default=1300, type=int, help='Largura da janela')
    parser.add_argument('-height', default=600, type=int, help='Altura da janela')
    parser.add_argument('-g_width', default=800, type=int, help='Largura do jogo')
    parser.add_argument('-g_height', default=600, type=int, help='Altura do jogo')
    parser.add_argument('-dir_name', default='saved_models', type=str, help='Diretório de saída')
    parser.add_argument('-load_nn', default=True, action='store_false', help='Desativar o load de nns salvas.')
    parser.add_argument('-continue_mutate_probability', default=0.05, type=float, help='Probabilidade de continuar a mutação.')
    parser.add_argument('-continue_crossover_probability', default=0.01, type=float, help='Probabilidade de continuar o crossover.')
    args = parser.parse_args()
    init(**vars(args))

if __name__ == "__main__": 
    main() 