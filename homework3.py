import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Data Preprocessing

img_size = (128, 128)
batch_size = 32
data_dir = 'Fish'

print('Loading dataset...')

# Train test split

full_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

train_bch = tf.data.experimental.cardinality(full_train_ds)
val_size = train_bch // 6

val_ds = full_train_ds.take(val_size)
train_ds = full_train_ds.skip(val_size)

class_names = full_train_ds.class_names
num_classes = len(class_names)
print(f'Classes detected: {class_names}')

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
])

normalization_layer = layers.Rescaling(1./255)

autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)
test_ds = test_ds.cache().prefetch(buffer_size=autotune)

print('Dataset loaded successfully!')
