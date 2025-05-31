import urllib.request
import zipfile
import os

background_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
evaluation_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"

def download_and_extract(url, extract_to):
    zip_filename = os.path.basename(url)
    urllib.request.urlretrieve(url, zip_filename)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(zip_filename)
    print(f"Данные сохранены в {extract_to}")

download_and_extract(background_url, 'images_background')
download_and_extract(evaluation_url, 'images_evaluation')