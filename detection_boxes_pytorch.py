from torchvision import transforms
import torch
import cv2
from torch.autograd import Variable
from p_utils.util import write_results, prep_image


def get_class_names(label_path):
    with open(label_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    classes.insert(0, '__background__')
    return classes if classes else None


class DetectBoxes:
    def __init__(self, label_path, conf_threshold=0.5, nms_threshold=0):
        self.classes = get_class_names(label_path)
        self.confThreshold = conf_threshold
        self.nmsThreshold = nms_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def bounding_box_yolo(self, frame, inp_dim, model):
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2).to(self.device)
        img = img.to(self.device)

        with torch.no_grad():
            output = model(Variable(img), self.device)
        output = write_results(output, self.confThreshold, len(self.classes), nms=True, nms_conf=self.nmsThreshold)

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        for index, out in enumerate(output):
            outs = out.tolist()
            left = int(outs[1])
            top = int(outs[2])
            right = int(outs[3])
            bottom = int(outs[4])

            cls = int(outs[-1])

            self.draw_boxes(frame, self.classes[cls+1], outs[5],  left, top, right, bottom)

    # detect bounding boxes rcnn
    def bounding_box_rcnn(self, frame, model):
        # Image is converted to image Tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(frame).to(self.device)

        # The image is passed through model to get predictions
        pred = model([img])

        # classes, bounding boxes, confidence scores are gained
        # only classes and bounding boxes > confThershold are passed to draw_boxes
        pred_class = [self.classes[i] for i in list(pred[0]['labels'].cpu().clone().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.confThreshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        for i in range(len(pred_boxes)):
            left = int(pred_boxes[i][0][0])
            top = int(pred_boxes[i][0][1])
            right = int(pred_boxes[i][1][0])
            bottom = int(pred_boxes[i][1][1])

            self.draw_boxes(frame, pred_class[i], pred_score[i], left, top, right, bottom)

    def draw_boxes(self, frame, class_id, score, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '{}%'.format(round((score * 100), 1))
        if self.classes:
            label = '%s %s' % (class_id, label)

        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + base_line), (255,255,255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)