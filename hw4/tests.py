import pytest
import numpy as np
import torch
from data import extract_sample, read_alphabets,testx, testy
from test import test
from train import train
from model import load_protonet_conv, ProtoNet, euclidean_dist
import torch.optim as optim

def test_extract_sample_shape():
    n_way = 3
    n_support = 2
    n_query = 2
    img_shape = (3, 28, 28)
    
    x = np.random.rand(30, *img_shape)
    y = np.repeat(np.arange(10), 3)[:30]

    sample = extract_sample(n_way, n_support, n_query, x, y)

    assert 'xs' in sample and 'xq' in sample and 'ys' in sample and 'yq' in sample
    assert sample['xs'].shape == (n_way * n_support, *img_shape)
    assert sample['xq'].shape == (n_way * n_query, *img_shape)
    assert sample['ys'].shape == (n_way * n_support,)
    assert sample['yq'].shape == (n_way * n_query,)

def test_model_forward():
    dummy_sample = {
        'images': torch.rand(5, 6, 3, 28, 28).cuda(),  # 5 classes, 6 images each
        'n_way': 5,
        'n_support': 3,
        'n_query': 3,
    }

    model = load_protonet_conv(x_dim=(3, 28, 28), hid_dim=32, z_dim=32)
    loss, output = model.set_forward_loss(dummy_sample)

    assert isinstance(loss, torch.Tensor)
    assert 'acc' in output
    assert 'loss' in output
    assert 'y_hat' in output
    assert output['y_hat'].shape == (5, 3)

def test_model_forward_loss():
    model = load_protonet_conv((3, 28, 28), 64, 64)
    n_way, n_support, n_query = 2, 2, 2

    sample = {
        'xs': np.random.rand(n_way * n_support, 3, 28, 28),
        'xq': np.random.rand(n_way * n_query, 3, 28, 28),
        'ys': np.repeat(np.arange(n_way), n_support),
        'yq': np.repeat(np.arange(n_way), n_query)
    }

    loss, output = model.set_forward_loss(sample)

    assert isinstance(loss, torch.Tensor)
    assert 'loss' in output and 'acc' in output and 'y_hat' in output
    assert isinstance(output['loss'], float)
    assert isinstance(output['acc'], float)

def test_model_output_dim():
    model = load_protonet_conv((3, 28, 28), 64, 64)
    dummy_input = torch.randn(4, 3, 28, 28)
    out = model.encoder(dummy_input)
    assert out.shape[-1] == 64, "Размеры не совпали"

def test_euclidean_dist_shape():
    x = torch.randn(5, 64)
    y = torch.randn(10, 64)
    dist = euclidean_dist(x, y)
    assert dist.shape == (5, 10), "Евклидова матрица должна быть размера (5, 10)"

def test_test_function_runs():
    model = load_protonet_conv()
    model.eval().cuda()
    test(model, testx, testy, n_way=2, n_support=1, n_query=1, test_episode=2)

def test_train_step():
    model = load_protonet_conv((3, 28, 28), 64, 64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x = np.random.rand(100, 3, 28, 28)
    y = np.repeat(np.arange(20), 5)

    train(model, optimizer, x, y,
          n_way=5,
          n_support=5,
          n_query=5,
          max_epoch=1,
          epoch_size=5)

    for param in model.parameters():
        assert param.grad is None or not torch.all(param.grad == 0), "Веса не изменились"