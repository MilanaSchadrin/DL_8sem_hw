import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from model import EncoderDecoder, NoamOpt
from train import fit
from stuff import DEVICE
from data import iterr, load_dataset, load_word_field

def main():
    word_field = load_word_field('./saved_data/wordfield')

    train_dataset = load_dataset('./saved_data/train')
    test_dataset = load_dataset('./saved_data/test')

    train_iter, test_iter = iterr(train_dataset,test_dataset)
    
    model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab)).to(DEVICE)
    pad_idx = word_field.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1).to(DEVICE)
    optimizer = NoamOpt(model.d_model, model)
    fit(model, criterion, optimizer, train_iter, start_epoch=0, epochs_count=1, val_iter=test_iter)
    torch.save(model.state_dict(), "results/model.pt")


if __name__ == '__main__':
    main()