# coding: utf-8
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import sys
import time

from t_utils.data_process import letterbox_resize, parse_anchors, read_class_names
from t_utils.nms_utils import gpu_nms
from models.darknet_tensorflow import Darknet
from t_utils import detection_boxes_tensorflow as vis


def arg_parse():
    parser = argparse.ArgumentParser(description="Tensorflow Yolov3")
    parser.add_argument("--video", help="Path where video is located",
                        default="assets/cars.mp4", type=str)
    parser.add_argument("--ckpt",  type=str, default="darknet/yolov3.ckpt",
                        help="The path of the weights to restore.")
    parser.add_argument("--conf", dest="confidence", help="Confidence threshold for predictions", default=0.5)
    parser.add_argument("--nms", dest="nmsThreshold", help="NMS threshold", default=0.4)
    parser.add_argument("--anchor_path", type=str, default="darknet/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--resolution", dest='resol',
                        help="Input resolution of network. Higher increases accuracy but decreases speed",
                        default=416, type=int)
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--webcam", help="Detect with web camera", default=False)

    return parser.parse_args()


def run_inference_for_single_image(frame, lbox_resize, sess, input_data, inp_dim, boxes, scores, labels):
    if lbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(frame, inp_dim, inp_dim)
    else:
        height_ori, width_ori = frame.shape[:2]
        img = cv2.resize(frame, tuple([inp_dim, inp_dim]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if lbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori / float(inp_dim))
        boxes_[:, [1, 3]] *= (height_ori / float(inp_dim))

    return boxes_, scores_, labels_


def main():
    args = arg_parse()

    PATH_TO_LABELS = 'labels/coco.names'

    # anchors and class labels
    anchors = parse_anchors(args.anchor_path)
    classes = read_class_names(PATH_TO_LABELS)
    num_classes = len(classes)
    VIDEO_PATH = args.video if not args.webcam else 0

    inp_dim = args.resol

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, inp_dim, inp_dim, 3], name='input_data')
        model = Darknet(num_classes, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_classes,
                                        max_boxes=200, score_thresh=args.confidence, nms_thresh=args.nmsThreshold)

        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt)

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

            # Actual Detection
            start = time.time()
            boxes_, scores_, labels_ = run_inference_for_single_image(frame,
                                                                      args.letterbox_resize,
                                                                      sess,
                                                                      input_data,
                                                                      inp_dim,
                                                                      boxes,
                                                                      scores,
                                                                      labels)
            end = time.time()

            # Visualization of the results of a detection
            vis.visualize_boxes_and_labels_yolo(frame,
                                                boxes_,
                                                classes,
                                                labels_,
                                                scores_,
                                                use_normalized_coordinates=False)

            cv2.putText(frame, '{:.2f}ms'.format((end - start) * 1000), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.imshow(winName, frame)
            print("FPS {:5.2f}".format(1 / (end - start)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Video ended")

        # releases video and removes all windows generated by the program
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
