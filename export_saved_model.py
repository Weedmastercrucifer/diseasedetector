# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:49:49 2021

@author: Supratim
"""
import tensorflow as tf
tf.keras.backend.set_learning_phase(0)
model=tf.keras.models.load_model('./vgg16_1.h5')
export_path='../my_image_classifier/1' 


with tf.python.keras.backend.get_session()as sess:
    tf.compat.v1.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image':model.input},
        outputs={t.name: t for t in model.outputs})