import settings
import model
from utils import VOC

import os
import time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"

sess = tf.InteractiveSession(config=config)
model = model.Model()
utils = VOC('train')

toContinue = True


#var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'yolo')#yolo/conv_2


if not toContinue :
    the_scope = 'yolo/(?!fc_36)' 
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = the_scope)#yolo/conv_2
else : 
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'yolo')
#print var_list
saver2=tf.train.Saver(var_list)
sess.run(tf.global_variables_initializer())
    
try:
    saver2.restore(sess, os.getcwd() + '/model.ckpt')

    print('load from past checkpoint')
except Exception as e:
    print(e)
    try:
        print('load yolo small')
        saver2.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print('loaded from YOLO small pretrained')
    except Exception as e:
        print(e)
        print('exit, atleast need a pretrained model')
        exit(0)
        
saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'yolo'),max_to_keep=2,save_relative_paths=True)
for i in range(settings.epoch):
        
    last_time = time.time()
    total_loss = 0
    
    for x in range(0, len(utils.gt_labels) - settings.batch_size, settings.batch_size):
        images = np.zeros((settings.batch_size, settings.image_size, settings.image_size, 3))
        labels = np.zeros((settings.batch_size, settings.cell_size, settings.cell_size, 13))
        
        for n in range(settings.batch_size):
            imname = utils.gt_labels[x + n]['imname']
            #print(imname)
            flipped = utils.gt_labels[x + n]['flipped']
            images[n, :, :, :] = utils.image_read(imname, flipped)
            #print  utils.gt_labels[x + n]['label'].shape
            labels[n, :, :, :] = utils.gt_labels[x + n]['label']
            
            #print(utils.gt_labels[x + n]['label'])
        #print(model.labels)
        loss, _ = sess.run([model.total_loss, model.optimizer], feed_dict = {model.images: images, model.labels: labels})
        total_loss += loss

        if (x + 1) % settings.checkpoint == 0:
            print('checkpoint reached: ' + str(x + 1))
    
    np.random.shuffle(utils.gt_labels)
    print('epoch: ' + str(i + 1) + ', loss: ' + str(loss / (len(utils.gt_labels) - settings.batch_size / (settings.batch_size * 1.0))) + ',  s / epoch: ' + str(time.time() - last_time),' e : ',str(loss))
    if i % 20 == 0 : 
        saver.save(sess, os.getcwd() + '/model.ckpt')
        print ("Save success")
       

