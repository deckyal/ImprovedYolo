import settings
import model
from utils import VOC
import os
import time
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"

sess = tf.InteractiveSession(config = config)
model = model.Model(training = False)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
boundary1 = settings.cell_size * settings.cell_size * settings.num_class
boundary2 = boundary1 + settings.cell_size * settings.cell_size * settings.box_per_cell

tryOriginal = settings.try_original

try:
    if tryOriginal : 
        saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print('load from YOLO small pretrained')
    else : 
        saver.restore(sess, os.getcwd() + '/model.ckpt')
        print('load from past checkpoint')
except:     
    try:
        saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print('load from YOLO small pretrained')
    except:
        print('you must train first, exiting..')
        exit(0)

def draw_result(img, result,color=(0, 255, 0)):
    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
def detect(img):
    img_h, img_w, _ = img.shape
    inputs = cv2.resize(img, (settings.image_size, settings.image_size))
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs = (inputs / 255.0) * 2.0 - 1.0
    inputs = np.reshape(inputs, (1, settings.image_size, settings.image_size, 3))
    
    r1,r2 = detect_from_cvmat(inputs)
    result = r1[0]#detect_from_cvmat(inputs)[0]
    result2 = r2[0]
    #print(result)

    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / settings.image_size)
        result[i][2] *= (1.0 * img_h / settings.image_size)
        result[i][3] *= (1.0 * img_w / settings.image_size)
        result[i][4] *= (1.0 * img_h / settings.image_size)
        
        result2[i][0] *= (1.0 * img_w / settings.image_size)
        result2[i][1] *= (1.0 * img_h / settings.image_size)
        result2[i][2] *= (1.0 * img_w / settings.image_size)
        result2[i][3] *= (1.0 * img_h / settings.image_size)
        
    return result,result2

def detect_from_cvmat(inputs):
    net_output = sess.run(model.logits, feed_dict = {model.images: inputs})
    results = []
    results2 = []
    for i in range(net_output.shape[0]):
        r1,r2 = interpret_output(net_output[i])
        results.append(r1)
        results2.append(r2)
        #print(' -',i)
    '''print('dfc : ',results)
    print('dfc2 : ',results2)'''
    return results,results2

def iou(box1, box2):
    #tb=min(b1[x2
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
    #return [intersection, (box1[2] * box1[3] + box2[2] * box2[3] - intersection)]

def iou_mine(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    print(intersection,box1[2] * box1[3])
    return [intersection/(box1[2] * box1[3]) , (box1[2] * box1[3] + box2[2] * box2[3] - intersection)]

def interpret_output(output):
    probs = np.zeros((settings.cell_size, settings.cell_size, settings.box_per_cell, len(settings.classes_name)))
    class_probs = np.reshape(output[0 : boundary1], (settings.cell_size, settings.cell_size, settings.num_class))
    scales = np.reshape(output[boundary1 : boundary2], (settings.cell_size, settings.cell_size, settings.box_per_cell))
    boxes = np.reshape(output[boundary2 :], (settings.cell_size, settings.cell_size, settings.box_per_cell, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(settings.cell_size)] * settings.cell_size * settings.box_per_cell), [settings.box_per_cell, settings.cell_size, settings.cell_size]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / settings.cell_size
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

    boxes *= settings.image_size

    for i in range(settings.box_per_cell):
        for j in range(settings.num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= settings.threshold, dtype = 'bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis = 3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > settings.IOU_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype = 'bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    result2 = []
    for i in range(len(boxes_filtered)):
        result.append([settings.classes_name[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
        result2.append([boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3],classes_num_filtered[i]])
    
    #print("The result : ",result)
    #print("The result2 : ",result2)
    return result,result2

def read_image(image, name,list_bb):
    result,result2 = detect(image)
    
    print(result,'\n',result2)
    #print('list_bb : ',list_bb[0])
    
    img_h, img_w, _ = image.shape
    
    list_bb_compare = []
    list_bb_compare2 = []
    
    
    for i in range(0,len(list_bb)) :
        #first convert the l_bb to the real image size 
     
        list_bb[i][0] *= (1.0 * img_w / settings.image_size)
        list_bb[i][1] *= (1.0 * img_h / settings.image_size)
        list_bb[i][2] *= (1.0 * img_w / settings.image_size)
        list_bb[i][3] *= (1.0 * img_h / settings.image_size)
        
        #Now convert to the x,y,w,h format
        x = (list_bb[i][0] + list_bb[i][2])/2
        y = (list_bb[i][1] + list_bb[i][3])/2
        w = list_bb[i][2] - list_bb[i][0] ;
        h = list_bb[i][3] - list_bb[i][1] ;
        
        #Now add them 
        list_bb_compare.append([x,y,w,h,list_bb[i][4]])
        list_bb_compare2.append(['test',x,y,w,h,0])
    
    print("lbb compare : ",list_bb_compare)
    
    draw_result(image, result,(0,255,0))
    draw_result(image, list_bb_compare2,(255,0,0))
    
    num_objects = len(list_bb_compare)
    
    num_cor = 0;
    num_false = 0;
    
    #classes_name =  ["van","truck", "bus", "car", "bicycle", "motorbike",'cyclist', "person"]
    #Now calculate the iou 
    for i in range(len(list_bb_compare)) :
        if np.max(np.asarray([list_bb_compare[i][0],list_bb_compare[i][1],list_bb_compare[i][2],list_bb_compare[i][3]])) <= 0 : 
            print("Skipping zero")
            continue
        else : 
            print("Checking : ",list_bb_compare[i])
        
        t_threshold = .7
        if list_bb_compare[i][4] == 6 : 
            t_threshold = .5 
        elif  list_bb_compare[i][4]== 7 : 
            t_threshold = .5
         
        bb_gt = [list_bb_compare[i][0],list_bb_compare[i][1],list_bb_compare[i][2],list_bb_compare[i][3]]
        det = False
        
        for j in range(len(result)) :
            bb_res = [result[j][1],result[j][2],result[j][3],result[j][4]]
            t_iou = iou_mine(bb_gt,bb_res)[0]
            
            print("\t \t IOU : ",t_iou)
            
            if t_iou > t_threshold : 
                det = True;
        
        if det : 
            num_cor += 1
        else : 
            num_false+=1
                
    cv2.imshow('test',image)
    cv2.waitKey(0)
    
    #plt.imshow(image)
    #plt.savefig(os.getcwd() + '/' + name + 'output.png')
    return num_cor,num_false
    

if settings.output == 1:
    image = cv2.imread(settings.picture_name)
    read_image(image, settings.picture_name[-10:])
    
if settings.output == 2:
    labels = VOC('test').load_labels()
    #print(labels)
    
    totalData = 0
    t_tr = 0
    
    for i in range(len(labels)):
        #print(labels[i]['list_bb'])
        #print(labels[i]['imname'])
        image = cv2.imread(labels[i]['imname'])
        n_tr, n_fl = read_image(image, labels[i]['imname'][-10:],labels[i]['list_bb'])
        
        totalData+= (n_tr + n_fl)
        t_tr += n_tr
        
        print("True : ",n_tr,"False : ",n_fl,"/n***********************/n")
    
    print("total acc : ",(t_tr/totalData)*100,'%')
         
if settings.output == 3:
    cap = cv2.VideoCapture(settings.video_name)
    ret, _ = cap.read()
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 640)) 
        result = detect(frame)
        draw_result(frame, result)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)
