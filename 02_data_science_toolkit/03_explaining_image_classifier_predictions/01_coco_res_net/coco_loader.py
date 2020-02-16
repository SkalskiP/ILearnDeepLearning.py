from skimage.io import imread
from skimage.transform import resize
import numpy as np

class CocoLoader:

    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_CHANNEL = 3

    def preprocess_image(self, path: str):
        x = imread(path)
        x = resize(x, (CocoLoader.IMG_WIDTH, CocoLoader.IMG_HEIGHT)) * 255
        return x

    def load_sample(self, size: int):
        with open("./dataset/5k.txt", "r") as f:
            images_paths = f.read().split('\n')
            dataset = np.ndarray(shape=(size, CocoLoader.IMG_WIDTH, CocoLoader.IMG_HEIGHT, CocoLoader.IMG_CHANNEL), dtype=np.float32)
            for index, images_path in enumerate(images_paths[:size]):
                dataset[index] = self.preprocess_image(images_path)
        return dataset