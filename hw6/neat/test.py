import pytest
import os
import pickle
from game.wrapped_flappy_bird import GameState
from game.train import eval_genomes
from game.run_best import run_best
import neat

@pytest.fixture
def game_state():
    return GameState(render=False)

@pytest.fixture
def sample_genome():
    config_path = os.path.join(os.path.dirname(__file__), '../neat-config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    return genome

def test_game_initialization(game_state):
    assert game_state.score == 0
    assert len(game_state.upperPipes) == 2
    assert game_state.playerx == int(288 * 0.2)

def test_genome_evaluation(sample_genome):
    config_path = os.path.join(os.path.dirname(__file__), '../neat-config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    genomes = [(1, sample_genome)]
    eval_genomes(genomes, config)
    assert sample_genome.fitness >= 0

def test_run_best(sample_genome):
    config_path = os.path.join(os.path.dirname(__file__), '../neat-config.txt')
    with open('test_genome.pkl', 'wb') as f:
        pickle.dump(sample_genome, f)
    
    score = run_best(config_path, 'test_genome.pkl')
    assert isinstance(score, int)
    os.remove('test_genome.pkl')