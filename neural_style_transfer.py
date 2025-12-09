import tensorflow as tf
from keras.applications.vgg19 import VGG19

vgg = VGG19(weights='imagenet', include_top=True)

content_layer = [vgg.get_layer('block5_conv2')]
style_layers = [vgg.get_layer('block1_conv1'), vgg.get_layer('block2_conv1'), vgg.get_layer('block3_conv1'), vgg.get_layer('block4_conv1'), vgg.get_layer('block5_conv1')]

content_seq = tf.keras.Sequential(content_layer)
style_seq = tf.keras.Sequential(style_layers)