# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import sys
import time

from t_utils.misc_utils import parse_anchors, read_class_names
from t_utils.nms_utils import *
from t_utils.plot_utils import plot_one_box
from t_utils.data_aug import letterbox_resize
from darknet_tensorflow import Darknet


def arg_parse():
    parser = argparse.ArgumentParser(description="Tensorflow Yolov3")
    parser.add_argument("--video", dest='video', help="Path where video is located",
                        default="assets/cars.mp4", type=str)
    parser.add_argument("--conf", dest="confidence", help="Confidence threshold for predictions", default=0.5)
    parser.add_argument("--nms", dest="nmsThreshold", help="NMS threshold", default=0.4)
    parser.add_argument("--anchor_path", type=str, default="darknet/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--resolution", dest='resol', help="Input resolution of network. Higher "
                                                      "increases accuracy but decreases speed",
                        default=416, type=int)
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--restore_path", type=str, default="darknet/yolov3.ckpt",
                        help="The path of the weights to restore.")
    return parser.parse_args()


args = arg_parse()

PATH_TO_LABELS = 'labels/coco.names'

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(PATH_TO_LABELS)
args.num_class = len(args.classes)
VIDEO_PATH = args.video

inp_dim = args.resol

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, inp_dim, inp_dim, 3], name='input_data')
    model = Darknet(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class,
                                    max_boxes=200, score_thresh=args.confidence, nms_thresh=args.nmsThreshold)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    # Set window
    winName = 'YOLO-Tensorflow'

    try:
        # Read Video file
        cap = cv2.VideoCapture(VIDEO_PATH)
    except IOError:
        print("Input video file", VIDEO_PATH, "doesn't exist")
        sys.exit(1)

    while cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(frame, inp_dim, inp_dim)
        else:
            height_ori, width_ori = frame.shape[:2]
            img = cv2.resize(frame, tuple([inp_dim, inp_dim]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(inp_dim))
            boxes_[:, [1, 3]] *= (height_ori/float(inp_dim))

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(frame, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100))
            cv2.putText(frame, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow(winName, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
