import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import keras_tuner  as kt
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

#  CNN Architecture

print('Building training model..')

EPOCHS = 15

baseline_model = models.Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    data_augmentation,
    normalization_layer,

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes, activation='softmax'),
])

baseline_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline = baseline_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

baseline_model.save_weights('baseline_model_weights.weights.h5')
print('Baseline model trained successfully')

# Hyperparameter Optimization

print('Starting Hyperparameter Optimization..')

def build_tuned_model(hp):
    model = models.Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),
        data_augmentation,
        normalization_layer,

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 4, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),

        layers.Dropout(hp.Choice('dropout', values=[0.3, 0.5])),

        layers.Dense(num_classes, activation='softmax')
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_tuned_model,
    objective='val_loss',
    max_trials=4,
    directory='tuning_dir',
    project_name='fish_classification'
)

for batch_size in [32, 64]:
    print(f"\nTuning with batch size: {batch_size}")
    # Re-batch datasets for the tuner
    temp_train_ds = full_train_ds.skip(val_size).unbatch().batch(batch_size).cache().prefetch(autotune)
    temp_val_ds = full_train_ds.take(val_size).unbatch().batch(batch_size).cache().prefetch(autotune)

    # Run the search
    tuner.search(temp_train_ds, epochs=10, validation_data=temp_val_ds)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nOptimization Complete!")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")
print(f"Best Dropout Rate: {best_hps.get('dropout')}")

# Build and train the final optimized model
print("Training optimized model..")
optimized_model = tuner.hypermodel.build(best_hps)
history_optimized = optimized_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

optimized_model.save_weights('optimized_model_weights.weights.h5')
print("\nOptimized model trained and weights saved successfully.")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation
print("\nEvaluating Models on Held-out Test Dataset...")

# Extract true labels from the test dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Generate predictions
print("Generating predictions for baseline model...")
baseline_preds = np.argmax(baseline_model.predict(test_ds), axis=-1)

print("Generating predictions for optimized model...")
opt_preds = np.argmax(optimized_model.predict(test_ds), axis=-1)

# Classification reports
print("\n--- Baseline Model Classification Report ---")
print(classification_report(y_true, baseline_preds, target_names=class_names))

print("\n--- Optimized Model Classification Report ---")
print(classification_report(y_true, opt_preds, target_names=class_names))

# Side-by-side visualizations
print("Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Baseline Loss/Accuracy Curves
axes[0, 0].plot(history_baseline.history['accuracy'], label='Train Acc')
axes[0, 0].plot(history_baseline.history['val_accuracy'], label='Val Acc')
axes[0, 0].plot(history_baseline.history['loss'], label='Train Loss')
axes[0, 0].plot(history_baseline.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Baseline Model Training Curves')
axes[0, 0].legend()

# Optimized Loss/Accuracy Curves
axes[0, 1].plot(history_optimized.history['accuracy'], label='Train Acc')
axes[0, 1].plot(history_optimized.history['val_accuracy'], label='Val Acc')
axes[0, 1].plot(history_optimized.history['loss'], label='Train Loss')
axes[0, 1].plot(history_optimized.history['val_loss'], label='Val Loss')
axes[0, 1].set_title('Optimized Model Training Curves')
axes[0, 1].legend()

# Multi-class Confusion Matrix for Optimized Model
cm = confusion_matrix(y_true, opt_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], xticklabels=class_names, yticklabels=class_names, cmap='Blues')
axes[1, 0].set_title('Optimized Model Confusion Matrix')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('evaluation_visualizations.png')
print("\nVisualizations saved to 'evaluation_visualizations.png'.")
