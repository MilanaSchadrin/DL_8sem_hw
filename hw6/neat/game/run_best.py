import pickle
import neat
import game.wrapped_flappy_bird as game
import time
import pygame

def run_best(config_path, genome_path='best_genome.pkl'):
    # Загрузка сохранённого лучшего генома
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    score = 0
    while True: 
        game_state = game.GameState()
        score = 0
        pipe_idx = 0

        while True:
            next_pipe = None
            for i in range(len(game_state.upperPipes)):
                if game_state.upperPipes[i]['x'] + 52 > game_state.playerx:
                    next_pipe = game_state.upperPipes[i], game_state.lowerPipes[i]
                    break

            if not next_pipe:
                break

            inputs = [
                game_state.playery / 512,
                (next_pipe[0]['y'] + 320) / 512,
                next_pipe[0]['x'] / 288
            ]

            output = net.activate(inputs)
            action = [1, 0]
            if output[0] > 0.5:
                action = [0, 1]

            _, _, done = game_state.frame_step(action)

            pipe_x = next_pipe[0]['x']
            if pipe_x + 52 < game_state.playerx <= pipe_x + 56:  # пролетел трубу
                score += 1


            if done:
                break

        return score

def show_score(score):
    font = pygame.font.SysFont("Roboto Condensed", 32)
    text_surface = font.render(f"Score: {score}", True, (255, 255, 255))

    bg_surface = pygame.Surface((150, 40))
    bg_surface.set_alpha(120)
    bg_surface.fill((0, 0, 0))


    game.SCREEN.blit(bg_surface, (10, 10))
    game.SCREEN.blit(text_surface, (20, 15))
    pygame.display.update()