import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import jaccard_score

size = 128

#Chargement du dataset  
dataset, info = tfds.load('oxford_iiit_pet:4.0.0', with_info=True)

def prepare(example):
    image = tf.image.resize(example['image'], (size, size))
    image = tf.cast(image, tf.float32) / 255.0  # Normalisation [0,1]
    mask = tf.image.resize(example['segmentation_mask'], (size, size), method='nearest')
    mask = tf.cast(mask, tf.float32)
    mask = tf.where(mask == 1.0, 1.0, 0.0)
    return image, mask

train_data = dataset['train'].map(prepare)
val_data   = dataset['test'].map(prepare)

X_train = np.array([img.numpy()  for img, _    in train_data])
y_train = np.array([mask.numpy() for _,   mask in train_data])
X_val   = np.array([img.numpy()  for img, _    in val_data])
y_val   = np.array([mask.numpy() for _,   mask in val_data])

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")

#Architecture U-Net 
input_layer = Input(shape=(size, size, 3))

#Encoder
conv1 = Conv2D(64,  (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(input_layer)
conv1 = Conv2D(64,  (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
pool1 = MaxPooling2D((2,2))(conv1)
pool1 = Dropout(0.1)(pool1)

conv2 = Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
conv2 = Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
pool2 = MaxPooling2D((2,2))(conv2)
pool2 = Dropout(0.1)(pool2)

conv3 = Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
conv3 = Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
pool3 = MaxPooling2D((2,2))(conv3)
pool3 = Dropout(0.2)(pool3)

conv4 = Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
conv4 = Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
pool4 = MaxPooling2D((2,2))(conv4)
pool4 = Dropout(0.2)(pool4)

#Bottleneck
bottleneck = Conv2D(1024, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
bottleneck = Conv2D(1024, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(bottleneck)

#Decoder
upconv1 = Conv2DTranspose(512, (2,2), strides=2, padding="same", kernel_initializer="he_normal")(bottleneck)
concat1 = concatenate([upconv1, conv4])
conv5 = Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(concat1)
conv5 = Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

upconv2 = Conv2DTranspose(256, (2,2), strides=2, padding="same", kernel_initializer="he_normal")(conv5)
concat2 = concatenate([upconv2, conv3])
conv6 = Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(concat2)
conv6 = Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

upconv3 = Conv2DTranspose(128, (2,2), strides=2, padding="same", kernel_initializer="he_normal")(conv6)
concat3 = concatenate([upconv3, conv2])
conv7 = Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(concat3)
conv7 = Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

upconv4 = Conv2DTranspose(64, (2,2), strides=2, padding="same", kernel_initializer="he_normal")(conv7)
concat4 = concatenate([upconv4, conv1])
conv8 = Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(concat4)
conv8 = Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

output_layer = Conv2D(1, (1,1), activation="sigmoid", padding="same")(conv8)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_unet_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

#Entraînement
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

#Courbes d'apprentissage
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

#Évaluation IoU et Dice
pred = model.predict(X_val, verbose=1)
pred_binary = (pred > 0.5).astype(int)
y_true = y_val.astype(int)

#IoU
iou = jaccard_score(pred_binary.flatten(), y_true.flatten())
print(f"IoU (Jaccard Score): {iou:.4f}")

#Dice
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

dice = dice_score(y_true, pred_binary)
print(f"Dice Score: {dice:.4f}")

#Affichage avec contour rouge sur image couleur
import cv2

i = 0  #Choisissez l'index de l'image que vous voulez afficher

image = X_val[i]
mask_real = y_val[i, :, :, 0]

pred_i = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0, :, :, 0]
mask_pred = (pred_i > 0.5).astype(np.uint8)

#Contours en rouge sur image couleur
contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_contour = image.copy()
cv2.drawContours(image_contour, contours, -1, (1, 0, 0), 2)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image originale")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Masque réel")
plt.imshow(mask_real, cmap="gray")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Masque prédit")
plt.imshow(mask_pred, cmap="gray")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Contour sur image")
plt.imshow(image_contour)
plt.axis('off')

plt.tight_layout()
plt.show()