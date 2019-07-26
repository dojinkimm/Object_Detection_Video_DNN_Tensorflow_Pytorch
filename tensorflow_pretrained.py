import numpy as np
import os
import tensorflow as tf
import cv2
import argparse
import imutils

from t_utils import ops as utils_ops, detection_boxes_tensorflow as vis
from t_utils import label_map_util
import time
import sys


def arg_parse():
    """ Parsing Arguments for detection """

    parser = argparse.ArgumentParser(description='Tensorflow Pretrained')
    parser.add_argument("--video", help="Path where video is located",
                        default="assets/cars.mp4", type=str)
    parser.add_argument("--frozen", help="Frozen inference pb file",
                        default="faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb")
    parser.add_argument("--conf", dest="confidence", help="Confidence threshold for predictions", default=0.5)
    parser.add_argument("--webcam", help="Detect with web camera", default=False)

    return parser.parse_args()


def run_inference_for_single_image(image, tensor_dict, sess):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    out_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    out_dict['num_detections'] = int(out_dict['num_detections'][0])
    out_dict['detection_classes'] = out_dict[
        'detection_classes'][0].astype(np.uint8)
    out_dict['detection_boxes'] = out_dict['detection_boxes'][0]
    out_dict['detection_scores'] = out_dict['detection_scores'][0]
    if 'detection_masks' in out_dict:
        out_dict['detection_masks'] = out_dict['detection_masks'][0]
    return out_dict


def main():
    args = arg_parse()

    VIDEO_PATH = args.video if not args.webcam else 0

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('labels', 'mscoco_label_map.pbtxt')

    # Load a Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(args.frozen, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Set window
    winName = 'Faster-RCNN-Tensorflow'

    try:
        # Read Video file
        cap = cv2.VideoCapture(VIDEO_PATH)
    except IOError:
        print("Input video file", VIDEO_PATH, "doesn't exist")
        sys.exit(1)

    gpu = tf.test.is_gpu_available()
    try:
        with detection_graph.as_default():
            config = tf.ConfigProto() if gpu else tf.ConfigProto(device_count={"GPU": 0})

            with tf.Session(config=config) as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)

                while cap.isOpened():
                    hasFrame, image_np = cap.read()
                    if not hasFrame:
                        break

                    image_np = imutils.resize(image_np, width=450)

                    # Actual Detection
                    start = time.time()
                    output_dict = run_inference_for_single_image(image_np, tensor_dict, sess)
                    end = time.time()

                    # Visualization of the results of a detection.
                    vis.visualize_boxes_and_labels_rcnn(
                                            image_np,
                                            output_dict['detection_boxes'],
                                            output_dict['detection_classes'],
                                            output_dict['detection_scores'],
                                            category_index,
                                            instance_masks=output_dict.get('detection_masks'),
                                            use_normalized_coordinates=True,
                                            min_score_thresh=args.confidence)

                    cv2.putText(image_np, '{:.2f}ms'.format((end - start) * 1000), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                    cv2.imshow(winName, image_np)
                    print("FPS {:5.2f}".format(1/(end - start)))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                print("Video ended")

                # releases video and removes all windows generated by the program
                cap.release()
                cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        cap.release()


if __name__ == "__main__":
    main()

