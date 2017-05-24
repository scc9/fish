import numpy as np
import glob
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

'''
Workaround script for model joining (seems to be an open issue with keras applications).

'''
no_clases=4
dir_path='/Users/Mathias/Desktop/HELCON_PILOT/code/small_test/test/skadad_test/*.JPG'

model1=InceptionV3(include_top=False, weights='imagenet')
model2=load_model("top_layer_all_20.h5")

def predict(img_path):
    img = image.load_img(img_path, target_size=(299, 299))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    top_features=model1.predict(x)
    pred=model2.predict(top_features)
    #print(pred)
    return pred

a=np.zeros(no_classes)
for filename in glob.glob(dir_path):
    p=predict(filename)
    a[np.argmax(p)]+=1
    print(a)
print("Done")
print(a/sum(a))
