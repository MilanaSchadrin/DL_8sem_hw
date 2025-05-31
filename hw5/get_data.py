import deeplake
import os
from PIL import Image
from tqdm import tqdm

def download_and_save_lfw(output_dir="lfw-funneled"):

    ds = deeplake.load('hub://activeloop/lfw-funneled')
    os.makedirs(output_dir, exist_ok=True)


    for i in tqdm(range(len(ds))):
        img_tensor = ds[i]['images'].numpy()        
        label = ds[i]['labels'].data()['value']   

        person_dir = os.path.join(output_dir, label)
        os.makedirs(person_dir, exist_ok=True)
  
        filename = os.path.join(person_dir, f"{i:05d}.jpg")

        img = Image.fromarray(img_tensor)
        img.save(filename)

    print(f"Загружено и сохранено {len(ds)} изображений в '{output_dir}'")