import tensorflow as tf
import keras
from keras import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras.preprocessing import image


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
#load in images from directory with a size of 64x64 pixels, batch size of 16, so 2 batches per epoch (32:16)
#after loading images from dir, save some images for training and others for validation data
train_generator = train_datagen.flow_from_directory(
    "car_data/support_set/",
    target_size=(128,128),
    batch_size=16,
    class_mode="categorical",
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    "car_data/support_set/",
    target_size=(128,128),
    batch_size=16,
    class_mode="categorical",
    subset="validation")

# Extract class names
class_names = list(train_generator.class_indices.keys())

#plot images
def plot_imgs(images, labels, class_names, title):
    num_images = min(len(images), 16) # adjust to batch size
    plt.figure(figsize=(12,12))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.show()


#display training images
train_images, train_labels = next(train_generator)
plot_imgs(train_images, train_labels, class_names, 'Training Data')

validation_images, validation_labels = next(validation_generator)
plot_imgs(validation_images, validation_labels, class_names, 'Validation Data')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# After training your model
history = model.fit(train_generator, 
                    steps_per_epoch=train_generator.samples,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples,
                    epochs=50,
                    callbacks=[early_stopping])

# Save the model
model.save('car_model.h5')
print("Model saved as 'car_model.h5'.")

# Evaluate the model
scores = model.evaluate(validation_generator)
print(f'Accuracy: {scores[1]*100}%')






# Create function to predict image
def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class name
    return img, predicted_class

# Create function to plot prediction result
def predict_and_plot_images(img_paths, model, class_names):
    plt.figure(figsize=(12, 8))
    for i, img_path in enumerate(img_paths, start=1):
        img, predicted_class = predict_image(img_path, model, class_names)
        plt.subplot(2, 2, i)
        plt.imshow(img)
        plt.title(f'Predicted Class: {predicted_class}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# List of paths to the four images
img_paths = [
    'car_data/query_set/cybertruck.jpeg',
    'car_data/query_set/testChevy.jpg'
]

# Predict and plot the images
predict_and_plot_images(img_paths, model, class_names)