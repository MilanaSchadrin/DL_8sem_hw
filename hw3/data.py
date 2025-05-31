import pandas as pd
from tqdm.auto import tqdm
from stuff import BOS_TOKEN, EOS_TOKEN, DEVICE
from torchtext.data import Field, Example, Dataset, BucketIterator
from pathlib import Path
import dill
import torch

word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
fields = [('source', word_field), ('target', word_field)]

def ensure_path(path):
    return Path(path) if not isinstance(path, Path) else path

def save_pickle(obj, filepath):
    torch.save(obj, filepath, pickle_module=dill)

def load_pickle(filepath):
    return torch.load(filepath, pickle_module=dill)

def save_dataset(dataset, path):
    path = ensure_path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(dataset.examples, path / "examples.pkl")
    save_pickle(dataset.fields, path / "fields.pkl")

def load_dataset(path):
    path = ensure_path(path)
    examples = load_pickle(path / "examples.pkl")
    fields = load_pickle(path / "fields.pkl")
    return Dataset(examples, fields)

def save_word_field(word_field, path):
    path = ensure_path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(word_field, path / "word_field.pkl")

def load_word_field(path):
    path = ensure_path(path)
    return load_pickle(path / "word_field.pkl")

def iterr(train_dataset,test_dataset):
    train_iter, test_iter = BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False)
    return train_iter, test_iter

def main():
    data = pd.read_csv('datasets/news.csv', delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))
    
    dataset = Dataset(examples, fields)
    train_dataset, test_dataset = dataset.split(split_ratio=0.85)
    word_field.build_vocab(train_dataset, min_freq=7)
    print('Vocab size =', len(word_field.vocab))

    save_dataset(train_dataset, './saved_data/train')
    save_dataset(test_dataset, './saved_data/test')
    
    save_word_field(word_field, './saved_data/wordfield')
                       
    print('Train size =', len(train_dataset))
    print('Test size =', len(test_dataset))

    

    

if __name__ == '__main__':
    main()