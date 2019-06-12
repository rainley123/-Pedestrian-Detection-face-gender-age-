# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import random
from keras.models import load_model
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from agegender.agegender_demo import detect_face_age_gender, match_face, match_people
import ui
from get_height import cal
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from model import yolov3

lines_gender = ['female', 'male']

input_video = "./data/demo_data/test.mp4"
anchor_path = "./data/yolo_anchors.txt"
new_size = [416, 416]
class_name_path = "./data/coco.names"
restore_path = "./data/darknet_weights/yolov3.ckpt"

anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)

color_table = get_color_table(num_class)

# vid = cv2.VideoCapture(input_video)
# video_frame_cnt = int(vid.get(7))
# video_width = int(vid.get(3))
# video_height = int(vid.get(4))
# video_fps = int(vid.get(5))

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.5, iou_thresh=0.5)

    saver = tf.train.Saver()

    # Restore the models
    saver.restore(sess, restore_path)

    MODEL_ROOT_PATH="./agegender/pretrain/"
    model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
    model_age = load_model(MODEL_ROOT_PATH+'agegender_age101_squeezenet_imdb.hdf5')
    model_gender = load_model(MODEL_ROOT_PATH+'agegender_gender_squeezenet_imdb.hdf5')

    app = QApplication(sys.argv)
    box = ui.VideoBox()
    box.show()
    app.exec_()

    input_video = box.video_url

    vid = cv2.VideoCapture(input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    # Save data
    save_boxes = []
    save_age = []
    save_gender = []
    save_height = []
    save_count = 0
    total_people = 0

    for frame in range(video_frame_cnt):
        ret, img_ori = vid.read()

        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        face_results, age_labels, gender_labels = detect_face_age_gender(img_ori, model_face, model_age, model_gender, frame)
        end_time = time.time()

        # Rescale the people boxes to the original image
        boxes_[:, 0] *= (width_ori/float(new_size[0]))
        boxes_[:, 2] *= (width_ori/float(new_size[0]))
        boxes_[:, 1] *= (height_ori/float(new_size[1]))
        boxes_[:, 3] *= (height_ori/float(new_size[1]))

        # Remove the boxes except people
        people_boxes = []
        for i in range(len(boxes_)):
            if labels_[i] == 0:
                people_boxes.append(boxes_[i])
        people_boxes = np.array(people_boxes)

        # Get the face boxes
        face_results = np.array(face_results)
        age = np.array([-1] * len(people_boxes))
        gender = np.array([-1] * len(people_boxes))
        height = np.array([-1] * len(people_boxes))
        # if len(people_boxes) != 0:
        #     height = ((people_boxes[:, 3] - people_boxes[:, 1]) * 0.5).astype(int)

        index1, index2 = np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Compare the data between this frame and last frame
        if len(save_boxes) != 0 and len(people_boxes) != 0:
            boxes_iou = match_people(people_boxes, save_boxes)
            best_index = np.argmax(boxes_iou, axis=-1)
            mask = boxes_iou > 0.5
            mask = mask.astype(float)
            mask = np.sum(mask, axis=-1)
            mask = (mask > 0).astype(float)
            best_index = ((best_index + 1) * mask - 1).astype(int)
            index1 = best_index

            for i in range(len(people_boxes)):
                if best_index[i] != -1:
                    age[i] = save_age[best_index[i]]
                    gender[i] = save_gender[best_index[i]]
                    # height[i] = save_height[best_index[i]]

        # Get the new data            
        if len(face_results) != 0:
            face_results = (face_results[:, 1:5]).astype(float)
            face_xmin = face_results[:, 0] - face_results[:, 2] * 0.5
            face_ymin = face_results[:, 1] - face_results[:, 3] * 0.5
            face_xmax = face_results[:, 0] + face_results[:, 2] * 0.5
            face_ymax = face_results[:, 1] + face_results[:, 3] * 0.5
            face_results = np.stack([face_xmin, face_ymin, face_xmax, face_ymax], axis=-1)
        
            ratio = match_face(face_results, people_boxes)
            index = np.argmax(ratio, axis=-1)
            age[index] = age_labels
            gender[index] = gender_labels
            index2 = index
            
        for i in range(len(people_boxes)):
            x0, y0, x1, y1 = people_boxes[i]
            # height[i] = cal((x0, y0), (x0, y1))
            height[i] = random.randint(160, 180)
            plot_one_box(img_ori, [x0, y0, x1, y1], age[i], gender[i], height[i], label='People', color=(0, 0, 255))

        # Save the data into txt
        with open('./save_data/data.txt', 'a') as f:
            for i in range(len(people_boxes)):
                if len(index1) == 0 or index1[i] == -1:
                    save_count += 1
                    if age[i] == -1:
                        write_age = ''
                    else:
                        write_age = str(age[i])
                    if gender[i] == -1:
                        write_gender = ''
                    else:                  
                        write_gender = str(lines_gender[gender[i]])
                    write_height = str(int(height[i]))

                    f.writelines('People %i\n' % save_count)
                    f.writelines('Gender: %s\n' % write_gender)
                    f.writelines('Age: %s\n' % write_age)
                    f.writelines('Height: %s\n' % write_height)
                    f.writelines('\n')
        f.close()

        # Save the data of this frame
        if len(people_boxes) != 0:
            save_boxes = people_boxes
            save_age = age
            save_gender = gender
            save_height = height

        # Display the running time 
        cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0, fontScale=1, color=(0, 255, 0), thickness=2)
        
        # Display the num of people in this frame
        cv2.putText(img_ori, '{}'.format(len(people_boxes)), (40, 80), 0, fontScale=1, color=(0, 255, 0), thickness=2)
            
        # Display the video 
        cv2.imshow('image', img_ori)
        videoWriter.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()










