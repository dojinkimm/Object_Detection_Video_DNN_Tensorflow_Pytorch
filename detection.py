import cv2


def get_class_names(label_path):
    with open(label_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes if classes else None


class DetectBoxes:
    def __init__(self, label_path, confidence_threshold = 0.5):
        self.classes = get_class_names(label_path)
        self.confThreshold = confidence_threshold

    # draw boxes higher than confidence threshold
    def draw_boxes(self, frame, class_id, conf, left, top, right, bottom):
        # draw a bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '{}%'.format(round((conf*100), 1))
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s %s' % (self.classes[class_id], label)

        # put label on top of detected bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])), (left + round(1.5 * label_size[0]), top + base_line),
                     (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # detect bounding boxes from give frame
    def detect_bounding_boxes(self, frame, output):
        width = frame.shape[0]
        height = frame.shape[1]

        for detection in output[0, 0, :, :]:
            score = float(detection[2])
            if score > self.confThreshold:
                class_id = int(detection[1])

                left = int(detection[3] * height)
                top = int(detection[4] * width)
                right = int(detection[5] * height)
                bottom = int(detection[6] * width)

                self.draw_boxes(frame, class_id, score, left, top, right, bottom)
