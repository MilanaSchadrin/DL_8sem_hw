import pytest
from game.wrapped_flappy_bird import GameState
from agent import *
import torch

@pytest.fixture
def game_state():
    return GameState()

def test_initial_state(game_state):
    assert game_state.score == 0
    assert len(game_state.upperPipes) == 2
    assert len(game_state.lowerPipes) == 2
    assert game_state.playerx == int(288 * 0.2)
    assert 0 <= game_state.playery < 512

def test_frame_step(game_state):
    image_data, reward, terminal = game_state.frame_step([1, 0])
    assert image_data.shape == (288, 512, 3)
    assert isinstance(reward, float)
    assert terminal is False

    image_data, reward, terminal = game_state.frame_step([0, 1])
    assert reward >= 0.1 

def test_pipe_movement(game_state):
    initial_x = game_state.upperPipes[0]['x']
    game_state.frame_step([1, 0])
    assert game_state.upperPipes[0]['x'] < initial_x 

def test_preprocessing():
    test_frame = np.random.randint(0, 255, size=(288, 512, 3), dtype=np.uint8)
    
    processed = preprocess(test_frame)
    
    assert processed.shape == (80, 80)
    assert processed.min() >= 0
    assert processed.max() <= 1.0
    assert processed.dtype == np.float64

