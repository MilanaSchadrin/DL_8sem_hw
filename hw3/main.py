import torch
import argparse
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from model import EncoderDecoder, NoamOpt
from train import fit
from stuff import DEVICE
from data import iterr, load_dataset, load_word_field
import yaml
from torchtext.vocab import GloVe
from label import LabelSmoothing

def main(pretrained_embeddings, use_label_smoothing):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    word_field = load_word_field('./saved_data/wordfield')

    train_dataset = load_dataset('./saved_data/train')
    test_dataset = load_dataset('./saved_data/test')

    train_iter, test_iter = iterr(train_dataset,test_dataset)
    
    model = EncoderDecoder(
        source_vocab_size=len(word_field.vocab),
        target_vocab_size=len(word_field.vocab),
        d_model=params['train']['d_model'],
        pretrained_embeddings=pretrained_embeddings,
        word_field=word_field
    ).to(DEVICE)
    pad_idx = word_field.vocab.stoi['<pad>']
    if use_label_smoothing:
        criterion = LabelSmoothing(
            size=len(word_field.vocab),
            padding_idx=pad_idx,
            smoothing=0.1
        ).to(DEVICE)
    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=0.1
        ).to(DEVICE)
    optimizer = NoamOpt(model.d_model, model)
    fit(model, criterion, optimizer, train_iter, start_epoch=0, epochs_count=params['train']['epochs'], val_iter=test_iter)
    if pretrained_embeddings and use_label_smoothing:
        save_path = "results/shared_emb_label_sm.pt"
    elif pretrained_embeddings and not use_label_smoothing:
        save_path = "results/shared_emb_not_label_smooth.pt"
    elif not pretrained_embeddings and use_label_smoothing:
        save_path = "results/model_label_smooth.pt"
    else:  # no pretrained embeddings, no label smoothing
        save_path = "results/model_not_label_sm.pt"
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with params')
    parser.add_argument('--pretrained_embeddings', type=lambda x: x.lower() == 'true', default=True,
                        help='Use pretrained embeddings (true/false)')
    parser.add_argument('--use_label_smoothing', type=lambda x: x.lower() == 'true', default=False,
                        help='Use LabelSmoothing loss (true/false)')

    args = parser.parse_args()

    main(args.pretrained_embeddings, args.use_label_smoothing)