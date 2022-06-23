import tensorflow
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
#Build model
model = Sequential()
model.add(Conv2D(32,(3,3), activation= 'relu', input_shape = (150,150,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3), activation= 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128,activation= 'relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

#View model
model.summary()

#Tao input #Batch_size  = 32, them mot so augmented
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=0.2,
                                   shear_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_data',
                                                 target_size =(150,150),batch_size = 32,
                                                 shuffle=True,
                                                 class_mode = 'binary')
testing_set = train_datagen.flow_from_directory('dataset/validation_data',
                                                target_size =(150,150),batch_size = 32, class_mode = 'binary')

#FIT MODEL
    #Tao checkpoint
model_path='Model_copy_opt2.h5'
    #Luu lai val_acc tot nhat
checkpointer = ModelCheckpoint(model_path, monitor='val_accuracy',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')
callbacks=[EarlyStopping(monitor='val_loss',patience=10), checkpointer]

history  = model.fit(
                    training_set,
                    steps_per_epoch=len(training_set),
                    epochs=100,
                    validation_data=testing_set,
                    validation_steps=len(testing_set),
                    callbacks= callbacks
                    )
import matplotlib.pyplot as plt
def plot_loss_curves(history) :
  history = history.history
  acc,val_acc = history["accuracy"], history["val_accuracy"]
  loss,val_loss = history["loss"], history["val_loss"]

  plt.figure(figsize=(16,6))
  plt.subplot(121)
  plt.plot(acc, label="train accuracy")
  plt.plot(val_acc, label="val accuracy")
  plt.title("Accuracy")
  plt.legend()

  plt.subplot(122)
  plt.plot(loss, label="train loss")
  plt.plot(val_loss, label="val loss")
  plt.title("Loss")
  plt.legend()
plot_loss_curves(history)
