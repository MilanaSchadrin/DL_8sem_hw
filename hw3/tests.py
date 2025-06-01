import pytest
from torchtext.data import Field, Example, Dataset
from stuff import BOS_TOKEN, EOS_TOKEN, DEVICE
import torch
from data import save_dataset, load_dataset, save_word_field, load_word_field, iterr
from stuff import words_to_tokens, tokens_to_words, subsequent_mask
from model import EncoderDecoder
from types import SimpleNamespace
from sum import get_data
import tempfile
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from evale import eval
from sum import summarize

@pytest.fixture
def word_field():
    field = Field(tokenize=str.split, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    return field


@pytest.fixture
def sample_dataset(word_field):
    fields = [('source', word_field), ('target', word_field)]
    examples = [
        Example.fromlist(["the cat sat".split(), "cat sat".split()], fields),
        Example.fromlist(["the dog barked".split(), "dog barked".split()], fields),
    ]
    dataset = Dataset(examples, fields)
    word_field.build_vocab(dataset)
    return dataset


def test_dataset_save_load(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_dataset(sample_dataset, tmp_path)
        loaded = load_dataset(tmp_path)

        assert len(loaded.examples) == len(sample_dataset.examples)
        assert loaded.fields['source'].preprocess("hello") == ['hello']


def test_word_field_save_load(word_field):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_word_field(word_field, tmp_path)
        loaded = load_word_field(tmp_path)

        assert loaded.init_token == word_field.init_token
        assert loaded.tokenize("hello world") == ["hello", "world"]


def test_iterators(sample_dataset):
    train_iter, test_iter = iterr(sample_dataset, sample_dataset)
    batch = next(iter(train_iter))
    assert hasattr(batch, 'source')
    assert hasattr(batch, 'target')
    assert isinstance(batch.source, torch.Tensor)


@pytest.fixture
def mock_field():
    class MockVocab:
        stoi = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 5, 'hello': 3, 'world': 4}
        itos = ['<pad>', '<s>', '</s>', 'hello', 'world', '<unk>']

    field = SimpleNamespace()
    field.vocab = MockVocab()
    field.preprocess = lambda x: x.lower().split()
    field.tokenize = lambda x: x.lower().split()
    field.init_token = '<s>'
    field.eos_token = '</s>'
    field.unk_token = '<unk>'
    field.pad_token = '<pad>'
    return field


def test_get_data(tmp_path):
    file = tmp_path / "test.txt"
    content = "First line\n\nSecond line\n"
    file.write_text(content)

    data = get_data(str(file))
    assert data == ["First line", "Second line"]


def test_words_to_tokens(mock_field):
    result = words_to_tokens(mock_field, "hello world")
    expected = torch.tensor([1, 3, 4, 2])
    assert torch.equal(result, expected)


def test_tokens_to_words(mock_field):
    result = tokens_to_words(mock_field, [3, 4])
    assert result == ["hello", "world"]


def test_subsequent_mask():
    mask = subsequent_mask(4)
    assert mask.shape == (1, 4, 4)
    m = mask[0]
    assert torch.all(m == torch.tril(torch.ones_like(m, dtype=m.dtype)))
    assert torch.all(m.triu(1) == 0)


def test_summarize_mock(monkeypatch, mock_field):
    dummy_output = torch.tensor([[[0.0, 0.0, 0.0, 10.0, 0.0]]])

    class DummyDecoder(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return dummy_output

    class DummyEncoder(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros(1, 1, 512)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DummyEncoder()
            self.decoder = DummyDecoder()

        def eval(self):
            pass

    model = DummyModel()

    input_text = "hello world"  # вместо тензора передаем строку
    summary = summarize(model, input_text, mock_field, max_summary_len=5, beam_size=2)

    assert isinstance(summary, str)
    assert summary != ""


@pytest.fixture
def dummy_vocab():
    class Vocab:
        stoi = {BOS_TOKEN: 0, EOS_TOKEN: 1, '<pad>': 2, 'hello': 3, 'world': 4}
        itos = ['<s>', '</s>', '<pad>', 'hello', 'world']
    return Vocab()


@pytest.fixture
def mock_field_with_vocab(dummy_vocab):
    field = SimpleNamespace()
    field.vocab = dummy_vocab
    field.preprocess = lambda x: x.lower().split()
    field.tokenize = lambda x: x.lower().split()
    field.init_token = BOS_TOKEN
    field.eos_token = EOS_TOKEN
    field.pad_token = '<pad>'
    return field


class DummyBatch:
    def __init__(self):
        self.source = torch.tensor([[3, 4, 1, 2]])  # 'hello world </s> <pad>'
        self.target = torch.tensor([[3, 4, 1, 2]])


@pytest.fixture
def dummy_iter():
    batch = DummyBatch()
    batch.source = batch.source.to(DEVICE)
    batch.target = batch.target.to(DEVICE)
    return [batch]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, trg, src_mask, trg_mask):
        output = torch.zeros(trg.shape[0], trg.shape[1], 5, device=trg.device)
        output[:, -1, 1] = 10.0
        return output


def test_eval_creates_metrics_file(monkeypatch, mock_field_with_vocab, dummy_iter):
    with TemporaryDirectory() as tmpdir:
        original_open = open  # сохраняем оригинал

        monkeypatch.setattr("evale.evaluate.load", lambda name: DummyRouge())
        monkeypatch.setattr("evale.json.dump", lambda obj, f, indent=None: f.write(json.dumps(obj)))
        monkeypatch.setattr("builtins.open", lambda path, mode="r": original_open(os.path.join(tmpdir, "rouge_metrics.json"), mode))

        model = DummyModel().to(DEVICE)
        os.makedirs(os.path.join(tmpdir, "metrics"), exist_ok=True)

        eval(model, mock_field_with_vocab, dummy_iter, max_len=5)

        with open(os.path.join(tmpdir, "rouge_metrics.json")) as f:
            data = json.load(f)

        assert "rouge1" in data
        assert "rouge2" in data
        assert "rougeL" in data
        assert all(isinstance(val, float) for val in data.values())


class DummyRouge:
    def compute(self, predictions, references, rouge_types, use_stemmer):
        DummyScore = SimpleNamespace(mid=SimpleNamespace(fmeasure=0.5))
        return {
            'rouge1': DummyScore,
            'rouge2': DummyScore,
            'rougeL': DummyScore,
        }