import os
import kagglehub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

_cache = {}

def download_data():
    # Download latest version
    path = kagglehub.dataset_download("shxrlenee/aigc-detection-dataset")
    print("Path to dataset files:", path)

    return path


def _load_data(path, folder):
    global _cache
    cache_key = (path, folder)
    if cache_key in _cache:
        return _cache[cache_key]

    # Data is found inside transformed_data/transformed_data in this specific dataset version
    data_dir = os.path.join(path, "transformed_data", "transformed_data", folder)
    if not os.path.exists(data_dir):
        # Fallback in case path directly contains 'train'/'test'
        if os.path.exists(os.path.join(path, folder)):
            data_dir = os.path.join(path, folder)
        else:
            raise FileNotFoundError(f"Could not find '{folder}' directory in {path}")
            
    X = []
    y = []
    
    classes = {'real': 0, 'fake': 1}
    
    for cls_name, label in classes.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        for file in os.listdir(cls_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(cls_dir, file)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((128, 128)) ######
                    img_array = np.array(img).flatten()
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
    X = np.array(X)
    y = np.array(y)
    
    _cache[cache_key] = (X, y)
    return X, y


def X_train(path):
    X, _ = _load_data(path, "train")
    return X


def y_train(path):
    _, y = _load_data(path, "train")
    return y


def X_test(path):
    X, _ = _load_data(path, "test")
    return X


def y_test(path):
    _, y = _load_data(path, "test")
    return y

def _test():
    from random import randint
    path = download_data()
    X_train_test = X_train(path)
    y_train_test = y_train(path)
    X_test_test = X_test(path)
    y_test_test = y_test(path)

    for i in range(2):
        rand_index = randint(0, len(y_train_test) - 1)
        X_sample = X_train_test[rand_index].reshape(128,128,3)
        y_label = y_train_test[rand_index]


    for i in range(2):
        rand_index = randint(0, len(y_test_test) - 1)
        X_sample = X_test_test[rand_index].reshape(128,128,3)
        y_label = y_test_test[rand_index]

    # display image
    plt.imshow(X_sample)
    plt.title(y_label)
    plt.show()

if __name__ == "__main__":
    _test()
