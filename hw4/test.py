from data import extract_sample, testx, testy, display_sample
from model import load_protonet_conv
from tqdm import trange
from train import model
import matplotlib.pyplot as plt
import numpy as np
from visualis import visualize_prediction, visualize_n_predictions
import yaml
import torch

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in trange(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']
        if episode == 0:
            visualize_prediction(sample, output['y_hat'])
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    n_way = params["n_way"]
    n_support =params["n_support"]
    n_query = params["n_query"]
    test_x = testx
    test_y = testy
    model = load_protonet_conv(x_dim=(3,28,28),
    hid_dim=64,
    z_dim=64)
    model.load_state_dict(torch.load("protonet_model.pt"))
    model.eval()
    model.cuda()
    test_episode = params["test_episode"]
    test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
    visualize_n_predictions(model, test_x, test_y, n_way, n_support, n_query, n_images=30)