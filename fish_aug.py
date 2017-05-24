from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

'''
Simple script to impose small variations on Images..

As of now this should be run after cropping.

TODO_LIST:
 1, combine with cropping.
 2, augment before crop but still keep fish inside the box. (less wierd filling etc.)
'''

datagen = ImageDataGenerator(
        rotation_range=6,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        channel_shift_range=10,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        rescale=None)

path='/Users/Mathias/Desktop/HELCON_PILOT/code/small_test/val/sell/'

i=0
for batch in datagen.flow_from_directory(path , batch_size=1,
                          save_to_dir='preview', save_prefix='sill_aug', save_format='jpg',
                          target_size=(480,180)):
    i += 1
    if i > 99:
        break  # otherwise the generator would loop indefinitely
