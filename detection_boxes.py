import cv2
import numpy as np
from colors import *


def get_class_names(label_path):
    with open(label_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes if classes else None


class DetectBoxes:
    def __init__(self, label_path, confidence_threshold=0.5, nms_threshold=0, mask_threshold=0, has_mask=False):
        self.classes = get_class_names(label_path)
        self.confThreshold = confidence_threshold
        self.nmsThreshold = nms_threshold
        self.maskThreshold = mask_threshold
        self.hasMask = has_mask
        self.maskColor = [255, 178, 50]

    # detect bounding boxes from given frame
    def detect_bounding_boxes(self, frame, output, masks=None):
        height = frame.shape[0]
        width = frame.shape[1]

        if self.nmsThreshold is not 0:
            self.detect_yolo(frame, output, width, height)
        elif self.maskThreshold is not 0:
            self.detect_maskrcnn(frame, output, width, height, masks)
        else:
            self.detect_fast_rcnn(frame, output, height, width)

    def detect_fast_rcnn(self, frame, output, width, height):
        for detection in output[0, 0, :, :]:
            score = float(detection[2])
            if score > self.confThreshold:
                class_id = int(detection[1])

                left = int(detection[3] * height)
                top = int(detection[4] * width)
                right = int(detection[5] * height)
                bottom = int(detection[6] * width)

                self.draw_boxes(frame, class_id, score, left, top, right, bottom)

    def detect_yolo(self, frame, output, frame_width, frame_height):
        # Search for all bounding boxes
        # Save bounding box that have higher score than given confidence threshold
        class_ids = []
        confidences = []
        boxes = []
        for out in output:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Using non-maximum suppression remove overlapping boxes
        # with low confidence
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_boxes(frame, class_ids[i], confidences[i], left, top, left + width, top + height)

    def detect_maskrcnn(self, frame, output, width, height, masks):
        numDetections = output.shape[2]
        for i in range(numDetections):
            box = output[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.confThreshold:
                class_id = int(box[1])
                left = int(width * box[3])
                top = int(height * box[4])
                right = int(width * box[5])
                bottom = int(height * box[6])

                left = max(0, min(left, width - 1))
                top = max(0, min(top, height - 1))
                right = max(0, min(right, width - 1))
                bottom = max(0, min(bottom, height - 1))

                class_mask = mask[class_id]

                self.draw_boxes(frame, class_id, score, left, top, right, bottom)
                if self.hasMask:
                    self.draw_masks(frame, class_mask, left, top, right, bottom)

    # draw boxes higher than confidence threshold
    def draw_boxes(self, frame, class_id, conf, left, top, right, bottom):
        color, txt_color = ((0, 0, 0), (0, 0, 0))
        label = '{}%'.format(round((conf*100), 1))
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s %s' % (self.classes[class_id], label)
            color = STANDARD_COLORS[class_id % len(STANDARD_COLORS)]

        if sum(color) < 500:
            txt_color = (255, 255, 255)

        # draw a bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color=color, thickness=3)

        # put label on top of detected bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + base_line),
                      color=color, thickness=cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=txt_color, thickness=2)

    def draw_masks(self, frame, class_mask, left, top, right, bottom):
        class_mask= cv2.resize(class_mask, (right - left + 1, bottom - top + 1))
        mask= (class_mask > self.maskThreshold)
        roi = frame[top:bottom+1, left:right+1][mask]

        frame[top:bottom+1, left:right+1][mask] = ([0.3*self.maskColor[0], 0.3*self.maskColor[1],
                                                    0.3*self.maskColor[2]] + 0.7 * roi).astype(np.uint8)
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame[top:bottom+1, left:right+1], contours, -1, self.maskColor, 3, cv2.LINE_8, hierarchy, 100)