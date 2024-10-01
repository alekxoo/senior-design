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
    "car_data/processed_support_set/",
    target_size=(224,224),
    batch_size=8,
    class_mode="categorical",
    subset="training")

validation_generator = train_datagen.flow_from_directory(
    "car_data/processed_support_set/",
    target_size=(224,224),
    batch_size=8,
    class_mode="categorical",
    subset="validation")

# Extract class names
class_names = list(train_generator.class_indices.keys())

#plot images
def plot_imgs(images, labels, class_names, title):
    num_images = min(len(images), 8) # adjust to batch size
    plt.figure(figsize=(12,12))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.suptitle(title, fontsize=16)  # Corrected 'fontsize'
    plt.show()


#display training images
train_images, train_labels = next(train_generator)
plot_imgs(train_images, train_labels, class_names, 'Training Data')

validation_images, validation_labels = next(validation_generator)
plot_imgs(validation_images, validation_labels, class_names, 'Validation Data')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
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
# model.save('car_model.h5')
# print("Model saved as 'car_model.h5'.")

# Evaluate the model
scores = model.evaluate(validation_generator)
print(f'Accuracy: {scores[1]*100}%')





# Create function to predict image
def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
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




###################other script

# import tensorflow as tf
# from tensorflow import keras
# from keras._tf_keras.keras import Sequential
# from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
# from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras._tf_keras.keras.optimizers import Adam
# from keras._tf_keras.keras.applications import EfficientNetB0
# from keras._tf_keras.keras.callbacks import LearningRateScheduler, EarlyStopping
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from rembg import remove
# from PIL import Image

# # Function to remove background and save the processed image
# def remove_background(input_path, output_path):
#     with open(input_path, 'rb') as i:
#         with open(output_path, 'wb') as o:
#             input = i.read()
#             output = remove(input)
#             o.write(output)

# # Process all images in the dataset
# def process_dataset(input_dir, output_dir):
#     for item in os.listdir(input_dir):
#         item_path = os.path.join(input_dir, item)
#         if os.path.isdir(item_path):  # Check if it's a directory
#             class_input_dir = item_path
#             class_output_dir = os.path.join(output_dir, item)
#             os.makedirs(class_output_dir, exist_ok=True)
            
#             for image_name in os.listdir(class_input_dir):
#                 if not image_name.startswith('.'):  # Skip hidden files
#                     input_path = os.path.join(class_input_dir, image_name)
#                     output_path = os.path.join(class_output_dir, image_name)
#                     remove_background(input_path, output_path)

# # Process the dataset
# input_dir = "car_data/support_set/"
# output_dir = "car_data/processed_support_set/"
# # process_dataset(input_dir, output_dir)

# # Set up image parameters
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# # Enhanced data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.8, 1.2],
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     output_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="training"
# )

# validation_generator = train_datagen.flow_from_directory(
#     output_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation"
# )

# class_names = list(train_generator.class_indices.keys())

# def plot_imgs(images, labels, class_names, title):
#     num_images = min(len(images), 16)
#     plt.figure(figsize=(12,12))
#     for i in range(num_images):
#         plt.subplot(4, 4, i+1)
#         plt.imshow(images[i])
#         plt.title(class_names[np.argmax(labels[i])])
#         plt.axis('off')
#     plt.suptitle(title, fontsize=16)
#     plt.show()

# train_images, train_labels = next(train_generator)
# plot_imgs(train_images, train_labels, class_names, 'Training Data')

# validation_images, validation_labels = next(validation_generator)
# plot_imgs(validation_images, validation_labels, class_names, 'Validation Data')

# # Use transfer learning with EfficientNetB0
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
# base_model.trainable = False

# # model = Sequential([
# #     base_model,
# #     GlobalAveragePooling2D(),
# #     Dense(512, activation='relu'),
# #     Dropout(0.5),
# #     Dense(256, activation='relu'),
# #     Dropout(0.3),
# #     Dense(len(class_names), activation='softmax')
# # ])

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
#     MaxPooling2D(pool_size=(2,2)),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(6, activation='softmax')
# ])

# # Use a lower learning rate
# optimizer = Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Learning rate scheduler
# def lr_schedule(epoch):
#     return 0.0001 * (0.1 ** int(epoch / 10))

# lr_scheduler = LearningRateScheduler(lr_schedule)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train for more epochs
# history = model.fit(
#     train_generator, 
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     epochs=100,
#     callbacks=[early_stopping, lr_scheduler]
# )

# # model.save('improved_car_model.h5')
# # print("Model saved as 'improved_car_model.h5'.")

# scores = model.evaluate(validation_generator)
# print(f'Validation Accuracy: {scores[1]*100:.2f}%')

# # Plot training history
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

# def predict_image(img_path, model, class_names):
#     # Remove background
#     output_path = 'temp_processed.png'
#     remove_background(img_path, output_path)
    
#     img = load_img(output_path, target_size=IMG_SIZE)
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
    
#     # Clean up
#     os.remove(output_path)
    
#     return img, predicted_class

# def predict_and_plot_images(img_paths, model, class_names):
#     plt.figure(figsize=(12, 8))
#     for i, img_path in enumerate(img_paths, start=1):
#         img, predicted_class = predict_image(img_path, model, class_names)
#         plt.subplot(2, 2, i)
#         plt.imshow(img)
#         plt.title(f'Predicted Class: {predicted_class}')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# img_paths = [
#     'car_data/query_set/cybertruck.jpeg',
#     'car_data/query_set/testChevy.jpg'
# ]

# predict_and_plot_images(img_paths, model, class_names)