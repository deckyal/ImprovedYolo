import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import settings
"""
#for kitti class
import pykitti
import tracklet 
from tracklet import Tracklet 
"""


class VOC:
    def __init__(self, phase):
        if settings.dataType == 0 : 
            self.data_path = '/home/deckyal/eclipse-workspace/YOLO-Object-Detection-Tensorflow-master/YOLO-Object-Detection-Tensorflow-master/VOCdevkit/VOC2012/'
        elif settings.dataType == 1 : 
            self.data_path = '/dos/VOC_KITTI_Object/VOC2012/'#
        elif settings.dataType == 2 : 
            self.data_path = '/dos/VOC_KITTI_Tracking/VOC2012/'#
        self.image_size = settings.image_size
        self.cell_size = settings.cell_size
        self.classes = settings.classes_name
        self.class_to_ind = settings.classes_dict
        self.flipped = settings.flipped
        self.phase = phase #problemas con phase con python3!
        self.gt_labels = None
        self.prepare()

    def image_read(self, imname, flipped = False):
        #print imname
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                #print(gt_labels_cp[idx]['label'] )
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = 'pascal_' + self.phase + '_labels.pkl'

        if os.path.isfile(cache_file):
            print(('Loading gt_labels from: ' + cache_file))
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
       
        print(('Processing gt_labels from: ' + self.data_path))
        
        
        if settings.dataType == 0 : 
            self.image_index = os.listdir(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages'))
            self.image_index = [i.replace('.jpg', '') for i in self.image_index]
        elif settings.dataType == 1 :
            self.image_index = os.listdir('/dos/VOC_KITTI_Object/VOC2012/JPEGImages/')
            self.image_index = [i.replace('.png', '') for i in self.image_index]
        elif settings.dataType == 2 :
            self.image_index = os.listdir('/dos/VOC_KITTI_Tracking/VOC2012/JPEGImages/')
            self.image_index = [i.replace('.png', '') for i in self.image_index]
            
        print(self.image_index)
        
        import random
        random.shuffle(self.image_index)
        
        if self.phase == 'train':
            val = int(len(self.image_index) * (1 - settings.test_percentage))
            self.image_index = self.image_index[: val]
        else:
            val = int(len(self.image_index) * settings.test_percentage)
            self.image_index = self.image_index[: val]

        gt_labels = []
        for index in self.image_index:
            label, num,l_bb = self.load_pascal_annotation(index)
            #print(l_bb)
            if num == 0:
                continue
            if settings.dataType == 0 : 
                imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            elif settings.dataType  in [1,2] : 
                imname = os.path.join(self.data_path, 'JPEGImages', index + '.png')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False,'list_bb' : l_bb})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        if settings.dataType == 0 : 
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        elif settings.dataType in [1,2] : 
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.png')
            
        im = cv2.imread(imname)
        '''cv2.imshow("test",im)
        cv2.waitKey(0)'''
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, 13))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        
        l_bb = np.zeros((len(objs),5))
        
        i = 0
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            
            try:  
                cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
                
                l_bb[i,0] = x1;l_bb[i,1] = y1; l_bb[i,2] = x2; l_bb[i,3] = y2;l_bb[i,4] = cls_ind;
                
                #print("cls ind : ",cls_ind)
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                x_ind = int(boxes[0] * self.cell_size / self.image_size)
                y_ind = int(boxes[1] * self.cell_size / self.image_size)
                if label[y_ind, x_ind, 0] == 1:
                    continue
                label[y_ind, x_ind, 0] = 1
                label[y_ind, x_ind, 1:5] = boxes
                label[y_ind, x_ind, 5 + cls_ind] = 1
                i+=1;
                #if cls_ind=='bicycle' or cls_ind=='bus' or cls_ind=='car' or cls_ind=='motorbike'or cls_ind=='person':
                   # continue
            except Exception as e:
                #print e
                print("Skipping "+str(e))
                pass
           
           

        return label, len(objs),l_bb
    
    
""" Kitti class    
class Kitti:
    def __init__(self, phase):
        self.data_path = os.path.join('KITTI')
        self.image_size = settings.image_size
        self.cell_size = settings.cell_size
        self.classes = settings.classes_name
        self.class_to_ind = settings.classes_dict
        self.flipped = settings.flipped
        self.phase = phase #problemas con phase con python3!
        self.gt_labels = None
        self.kittiDir='KITTI';
        self.drive = '2011_09_26_drive_0009_sync'
        self.date='2011_09_26';
        self.drive1= '0009'; 
     
        self.prepare()
   
    def kitti_read(self, imname, flipped = False):
        
        dataset = pykitti.raw(self.kittiDir, self.date, self.drive1)
        n_frames=len(list(dataset.velo));
        immname=os.path.join(self.data_path,'KITTI')
        self.n_frames=n_frames;
            #for idx in range(n_frames):
        images=dataset.get_cam0(idx)
        return n_frames    
             
    def kitti_loadlabels(self, index):
        dataset = pykitti.raw(self.kittiDir, self.date, self.drive1)
        n_frames=len(list(dataset.velo));
        immname=os.path.join(self.data_path,'KITTI')
        myTrackletFile = join(kittiDir,'2011_09_26/2011_09_26_drive_0009_sync', 'tracklet_labels.xml')
        tracklets =Tracklet.parseXML(myTrackletFile)
        immname=os.path.join(self.data_path,'KITTI')
        for i in range(n_frames):
            frame_tracklets[i] = []
            frame_tracklets_types[i] = []

    # loop over tracklets
        for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
            h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
            trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
            ])
        # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
                if truncation not in (Tracklet.TRUNC_IN_IMAGE, Tracklet.TRUNC_TRUNCATED):
                    continue
            # re-create 3D bounding box in velodyne coordinate system
                    yaw = rotation[2]  # other rotations are supposedly 0
                    assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                    rotMat = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]
                        ])
                    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
                frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]
        return frame_tracklets, frame_tracklets_types 
           
 """       
        
        
        
        
        
        
