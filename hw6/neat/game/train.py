import neat
import pickle
from game.wrapped_flappy_bird import PIPE_HEIGHT,PIPEGAPSIZE,SCREENHEIGHT
import game.wrapped_flappy_bird as game
import matplotlib.pyplot as plt
import os
TARGET_SCORE = 500

def eval_genomes(genomes, config):
    global best_fitness
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game_state = game.GameState(render=False)

        fitness = 0.0
        genome.fitness = 0.0

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

            _, reward, done = game_state.frame_step(action)

            fitness += reward
            if done:
                break

        genome.fitness = fitness
        if fitness > best_fitness:
            best_fitness = fitness

def train(config_path):
    global best_fitness
    best_fitness = 0
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)

    generation = 0
    while True:
        generation += 1
        winner = population.run(eval_genomes, 50)

        print(f"[INFO] Поколение {generation}, лучший fitness: {best_fitness:.2f}")
        if best_fitness > TARGET_SCORE:
            print(f"Cчёт достигнут! Остановка обучения.")
            break

    #Лучшая птичка
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    gen = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]
    avg = stats.get_fitness_mean()
    stdev = stats.get_fitness_stdev()

    plt.figure(figsize=(10, 6))
    plt.plot(gen, best, label="Лучший fitness")
    plt.plot(gen, avg, label="Средний fitness")
    plt.fill_between(gen,
                     [a - s for a, s in zip(avg, stdev)],
                     [a + s for a, s in zip(avg, stdev)],
                     alpha=0.2,
                     label="Стандартное отклонение")
    plt.title("Динамика обучения NEAT")
    plt.xlabel("Поколение")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/training_progress.png")
    plt.show()
