import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input



class DataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.dataframe))
        self.num_classes = len(self.dataframe['label'].unique())
        self.on_epoch_end()

        if self.augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]

        images = []
        labels = []

        for _, row in batch_data.iterrows():
            img = load_img(row['path'], target_size=(224, 224))
            img_array = img_to_array(img)

            if self.augment:
                img_array = self.augmentor.random_transform(img_array)

            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(row['label'])

        images = np.array(images)
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
