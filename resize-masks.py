# Resize ground truth segmentation masks
# for training U-net

import tensorflow as tf
import glob
from keras.preprocessing.image import load_img, img_to_array, save_img

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

DIR = 'dataset/mask-big'
FILES = glob.glob(f'{DIR}/*.png')
TARGET_SIZE = (256, 256)
SAVE_DIR = 'dataset/mask'

with tf.device('/GPU:0'):
    for f in FILES:
        img = load_img(f, target_size = TARGET_SIZE)
        img = img_to_array(img)
        filename = f.split('/')[2]
        print(f'Resize image {filename}')
        save_img(f'{SAVE_DIR}/{filename}', img)