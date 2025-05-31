import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from identify_face import identify_face
import torch
import torch.nn as nn
import os
import pytest
from visualize_identify import evaluate_predictions, visualize_comparison
import tempfile
import shutil
from PIL import Image
from torchvision import transforms
from prepare_data import TripletDataset, filter_classes
from torch.utils.data import DataLoader, Dataset
from train import train 
from get_data import download_and_save_lfw  

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = torch.randn(1, 128)

@pytest.fixture
def mock_transform():
    return MagicMock()

@pytest.fixture
def dummy_image_paths(tmp_path):
    return [str(tmp_path / f"img{i}.jpg") for i in range(15)]

@patch("identify_face.get_image_paths")
@patch("identify_face.compute_embeddings")
@patch("identify_face.visualize_comparison")
def test_identify_face_basic(mock_visualize, mock_embeddings, mock_get_paths, mock_model, mock_transform, tmp_path):

    ref_path = str(tmp_path / "ref.jpg")
    dummy_ref_embedding = np.ones(128)
    dummy_test_embeddings = np.vstack([np.ones(128)*0.9 if i < 5 else np.ones(128)*0.2 for i in range(15)])
    
    mock_get_paths.return_value = [str(tmp_path / f"img{i}.jpg") for i in range(15)]
    mock_embeddings.side_effect = lambda model, paths, transform, device: \
        [dummy_ref_embedding] if len(paths) == 1 else dummy_test_embeddings

    matches, non_matches, threshold, ref_name = identify_face(
        reference_path=ref_path,
        test_folder=str(tmp_path),
        model=mock_model,
        transform=mock_transform,
        device="cpu",
        ref_im=None,
        threshold=None
    )

    assert isinstance(matches, list)
    assert isinstance(non_matches, list)
    assert isinstance(threshold, float)
    assert isinstance(ref_name, str)
    assert len(matches) + len(non_matches) == 15
    assert mock_visualize.called

@pytest.fixture
def dummy_gt_file():
    content = """filename;Alice;Bob
img1.jpg;1;0
img2.jpg;1;0
img3.jpg;0;1
img4.jpg;0;1
img5.jpg;1;0
"""
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix=".csv") as f:
        f.write(content)
        return f.name

@pytest.fixture
def dummy_matches():
    return [("img1.jpg", 0.3), ("img2.jpg", 0.25), ("img5.jpg", 0.2)]

@patch("visualize_identify.wandb.log")
def test_evaluate_predictions(mock_wandb_log, dummy_gt_file, dummy_matches):
    results, metrics = evaluate_predictions(
        gt_file=dummy_gt_file,
        reference_person="Alice",
        matches=dummy_matches,
        threshold=0.4,
        output_file="dummy_output.txt"
    )

    assert isinstance(results, list)
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1', 'threshold'])
    assert os.path.exists("dummy_output.txt")

    os.remove("dummy_output.txt")
    os.remove(dummy_gt_file)

@patch("visualize_identify.Image.open")
@patch("visualize_identify.plt")
def test_visualize_comparison(mock_plt, mock_image):
    dummy_ref_path = "ref.jpg"
    dummy_matches = [("img1.jpg", 0.2), ("img2.jpg", 0.3)]
    dummy_non_matches = [("img3.jpg", 0.5)]
    
    dummy_img = MagicMock(spec=Image.Image)
    mock_image.return_value = dummy_img

    visualize_comparison(
        reference_path=dummy_ref_path,
        matches=dummy_matches,
        non_matches=dummy_non_matches,
        threshold=0.4,
        ref_im="Alice",
        save_to_file=False
    )

    assert mock_image.called
    assert mock_plt.imshow.called
    assert mock_plt.show.called


@pytest.fixture
def dummy_dataset_root():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "class1"), exist_ok=True)
    os.makedirs(os.path.join(root, "class2"), exist_ok=True)
    os.makedirs(os.path.join(root, "class3"), exist_ok=True)

    def create_image(path):
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(path)

    for i in range(3):
        create_image(os.path.join(root, "class1", f"img{i}.jpg"))
    for i in range(2):
        create_image(os.path.join(root, "class2", f"img{i}.jpg"))
    for i in range(1): 
        create_image(os.path.join(root, "class3", f"img{i}.jpg"))

    yield root
    shutil.rmtree(root)

def test_filter_classes(dummy_dataset_root):
    valid_classes = filter_classes(dummy_dataset_root, min_samples=2)
    assert "class1" in valid_classes
    assert "class2" in valid_classes
    assert "class3" not in valid_classes
    assert len(valid_classes) == 2

def test_triplet_dataset_structure(dummy_dataset_root):
    valid_classes = filter_classes(dummy_dataset_root, min_samples=2)
    transform = transforms.ToTensor()
    dataset = TripletDataset(dummy_dataset_root, valid_classes, transform=transform)

    a, p, n = dataset[0]
    assert isinstance(a, torch.Tensor)
    assert isinstance(p, torch.Tensor)
    assert isinstance(n, torch.Tensor)
    assert a.shape == p.shape == n.shape

def test_triplet_different_classes(dummy_dataset_root):
    valid_classes = filter_classes(dummy_dataset_root, min_samples=2)
    dataset = TripletDataset(dummy_dataset_root, valid_classes)

    a, p, n = dataset[0]

    assert isinstance(a, Image.Image)
    assert isinstance(p, Image.Image)
    assert isinstance(n, Image.Image)
    assert a.size == p.size == n.size

@pytest.fixture
def temp_output_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)

class DummyImageTensor:
    def __init__(self, array):
        self._array = array
    def numpy(self):
        return self._array

class DummyLabel:
    def __init__(self, value):
        self._value = value
    def data(self):
        return {'value': self._value}

class DummyDataset:
    def __init__(self):
        self.data = [
            {'images': DummyImageTensor(np.zeros((64,64,3), dtype=np.uint8)), 'labels': DummyLabel('Person_A')},
            {'images': DummyImageTensor(np.ones((64,64,3), dtype=np.uint8)*255), 'labels': DummyLabel('Person_B')}
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

@patch("get_data.deeplake.load")
def test_download_and_save_lfw(mock_deeplake_load, temp_output_dir):
    mock_deeplake_load.return_value = DummyDataset()

    # Импортируем функцию внутри теста, если нужно
    from get_data import download_and_save_lfw

    download_and_save_lfw(output_dir=temp_output_dir)

    person_a_dir = os.path.join(temp_output_dir, 'Person_A')
    person_b_dir = os.path.join(temp_output_dir, 'Person_B')

    assert os.path.isdir(person_a_dir)
    assert os.path.isdir(person_b_dir)

    person_a_files = os.listdir(person_a_dir)
    person_b_files = os.listdir(person_b_dir)

    assert len(person_a_files) == 1
    assert len(person_b_files) == 1

    img_a = Image.open(os.path.join(person_a_dir, person_a_files[0]))
    img_b = Image.open(os.path.join(person_b_dir, person_b_files[0]))

    assert img_a.size == (64, 64)
    assert img_b.size == (64, 64)