import cv2
import sys
import argparse
from detection_boxes import DetectBoxes
import time
from thread_w_return import *


def arg_parse():
    """ Parsing Arguments for detection """

    parser = argparse.ArgumentParser(description='Pytorch Yolov3')
    parser.add_argument("--video", help="Path where video is located",
                        default="assets/cars.mp4", type=str)
    parser.add_argument("--config", help="Yolov3 config file", default="darknet/yolov3.cfg")
    parser.add_argument("--weight", help="Yolov3 weight file", default="darknet/yolov3.weights")
    parser.add_argument("--conf", dest="confidence", help="Confidence threshold for predictions", default=0.5)
    parser.add_argument("--nms", dest="nmsThreshold", help="NMS threshold", default=0.4)
    parser.add_argument("--resol", dest='resol', help="Input resolution of network. Higher "
                                                      "increases accuracy but decreases speed",
                        default="416", type=str)
    parser.add_argument("--webcam", help="Detect with web camera", default=False)
    return parser.parse_args()


def get_outputs_names(net):
    # names of network layers e.g. conv_0, bn_0, relu_0....
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detection_gpu(frame_list, net, detect):
    frame_with_rect = []
    for frame in frame_list:
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (int(416), int(416)), (0, 0, 0), True,
                                     crop=False)

        # Set the input to the network
        net.setInput(blob)

        # Runs the forward pass
        network_output = net.forward(get_outputs_names(net))

        # Extract the bounding box and draw rectangles
        detect.detect_bounding_boxes(frame, network_output)

        # Efficiency information
        t, _ = net.getPerfProfile()
        elapsed = abs(t * 1000.0 / cv2.getTickFrequency())
        label = 'Time per frame : %0.0f ms' % elapsed
        cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        frame_with_rect.append(frame)

    return frame_with_rect


def main():
    args = arg_parse()

    VIDEO_PATH = args.video if not args.webcam else 0

    print("Loading network.....")
    net = cv2.dnn.readNetFromDarknet(args.config, args.weight)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Network successfully loaded")

    # class names ex) person, car, truck, and etc.
    PATH_TO_LABELS = "labels/coco.names"

    # load detection class, default confidence threshold is 0.5
    detect = DetectBoxes(PATH_TO_LABELS, confidence_threshold=args.confidence, nms_threshold=args.nmsThreshold)

    # Set window
    winName = 'YOLO-Opencv-DNN'

    try:
        # Read Video file
        cap = cv2.VideoCapture(VIDEO_PATH)
    except IOError:
        print("Input video file", VIDEO_PATH, "doesn't exist")
        sys.exit(1)

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 내가 원하는 개수에 맞춰서 frame을 나눈다
    div = frame_length // 2
    divide_point = [i for i in range(frame_length) if i != 0 and i % div == 0]
    divide_point.pop()

    frame_list = []
    fragments = []
    count = 0
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            frame_list.append(fragments)
            break
        if count in [0, 48]:
            frame_list.append(fragments)
            fragments = []
        fragments.append(frame)
        count += 1
    cap.release()

    # Threading을 통해서 여러 frame들을 detection하게 한다
    thread_detection = [ThreadWithReturnValue(target=detection_gpu,
                                              args=(frame_list[i], net, detect))
                        for i in range(2)]

    final_list = []
    # Threading 을 시작한다
    for th in thread_detection:
        th.start()

    # Threading 이 끝나면 return 받은 값을 새로운 리스트에 담는다
    for th in thread_detection:
        final_list.extend(th.join())

    # return 받은 value 를 video 에 작성한다
    for f in final_list:
        videoWriter.write(f)


if __name__=="__main__":
    s = time.time()
    main()
    e = time.time()
    print((e-s)*1000)

# 34.600
# 17340