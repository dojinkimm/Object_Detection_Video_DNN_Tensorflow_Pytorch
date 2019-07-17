from torchvision import transforms
import torch
import cv2

def get_class_names(label_path):
    with open(label_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    classes.insert(0, '__background__')
    return classes if classes else None


class DetectBoxes:
    def __init__(self, label_path, conf_threshold=0.5):
        self.classes = get_class_names(label_path)
        self.confThreshold = conf_threshold

    # detect bounding boxes from give frame
    def detect_bounding_boxes(self, frame, model):
        """
        get_prediction
          parameters:
            - image - frame of video
            - threshold - threshold value for prediction score
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
              are chosen.

        """
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(frame)
        if torch.cuda.device_count() > 0:
            img = img.cuda()
        pred = model([img])
        pred_class = [self.classes[i] for i in list(pred[0]['labels'].cpu().clone().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.confThreshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        self.draw_boxes(frame, pred_boxes, pred_class)
        # return pred_boxes, pred_class

    def draw_boxes(self, frame, boxes, pred_cls):
        for i in range(len(boxes)):
            cv2.rectangle(frame, boxes[i][0], boxes[i][1], (255, 178, 50), 3)
            cv2.putText(frame, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)