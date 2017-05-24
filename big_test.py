import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

#InceptionV3 input size

#COMMON PARAMS
img_width, img_height= 299,299
#path to top model weights
top_model_weights_path = 'bottleneck_fc_model.h5'

no_classes=4
no_train_per_class=1000  #REFRACTOR..
no_val_per_class=200    #REFRACTOR.. laterz
epochs = 1
batch_size = 10
train_data_dir = 'small_test/train'
validation_data_dir = 'small_test/val'


nb_train_samples = no_train_per_class*no_classes
nb_validation_samples = no_train_per_class*no_classes




def get_bottleneck_features():
    train_data_dir = 'small_test/train'
    validation_data_dir = 'small_test/val'

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    #build the inceptionV3 network w/o top layers
    model = InceptionV3(include_top=False, weights='imagenet')




    train_generator = datagen.flow_from_directory(
       train_data_dir,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       class_mode=None,
       shuffle=False)
    print("Starting predict gen")

    train_labels=to_categorical(train_generator.classes.tolist(),num_classes=no_classes)
    np.save("train_labels.npy",train_labels)
    train_data = model.predict_generator(train_generator, nb_train_samples // batch_size)
    np.save("train_data.npy",train_data)
    val_generator = datagen.flow_from_directory(
       validation_data_dir,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       class_mode="categorical",
       shuffle=False)
    print("lul")
    val_labels=to_categorical(val_generator.classes.tolist(),num_classes=no_classes)
    np.save("val_labels.npy",val_labels)
    validation_data = model.predict_generator(val_generator, nb_validation_samples // batch_size)
    np.save("val_data.npy",validation_data)



def fit_top_model():
        top_model = Sequential()
        top_model.add(Flatten(input_shape=(8, 8, 2048)))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(no_classes, activation='softmax'))

        top_model.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy', metrics=['accuracy'])

        train_data=np.load("train_data.npy")
        train_labels=np.load("train_labels.npy")
        validation_data=np.load("val_data.npy")
        validation_labels=np.load("val_labels.npy")
        top_model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(validation_data, validation_labels))
        #preds=model.predict(validation_data)
        top_model.save("top_layer_all.h5")

get_bottleneck_features()
#fit_top_model()
