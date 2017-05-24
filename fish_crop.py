from PIL import ImageFile
from PIL import Image
import glob
'''
Simple fish crop script.

Output dim: 480x180

Should be run before fish_aug.

WARNING: Crop will break if fed with images of other dim / fish position than those we have
so far... This needs to be fixed manually since the we dont have nor will ever get box coords.
'''


dir_path='/Users/Mathias/Desktop/HELCON_PILOT/data/Makrill_OK/*.JPG'
crop_path='/Users/Mathias/Desktop/HELCON_PILOT/data/Makrill_OK_crop/'
image_list=[]

ImageFile.LOAD_TRUNCATED_IMAGES = True  #This is important.
i=0

for filename in glob.glob(dir_path):

    with Image.open(filename) as img:

        i+=1
        print(i)
        img.load()
        new_file= crop_path + 'makrill' + '{0:04}'.format(i) + '.JPG'
        #image_list.append(im)
        width = img.size[0]
        height = img.size[1]
        im= img.crop(
        (width - 390,0,width-210,height))
        #im.transpose(Image.FLIP_LEFT_RIGHT).save(new_file)
        im.save(new_file)
    if i>9990:
        print("lul")
        break
