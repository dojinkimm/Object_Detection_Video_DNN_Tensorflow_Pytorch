import cv2 as cv
import sys
from torchvision import models, transforms
from imutils.video import FPS
import torch

fileName = "assets/cars.mp4"

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
if torch.cuda.device_count() > 0:
    model.cuda()
model.eval()

classesFile = "labels/mscoco_labels.names"
COCO_INSTANCE_CATEGORY_NAMES = None
with open(classesFile, 'rt') as f:
    COCO_INSTANCE_CATEGORY_NAMES = f.read().rstrip('\n').split('\n') # classes variable contains names
COCO_INSTANCE_CATEGORY_NAMES.insert(0,'__background__')


# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]



def get_prediction(image, threshold):
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
    img = transform(image)
    if torch.cuda.device_count() > 0:
        img = img.cuda()
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().clone().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


# Process inputs
winName = 'Faster-RCNN-Pytorch'
try:
    # 파일 있으면 시작을 한다
    cap = cv.VideoCapture(fileName)
except IOError:
    print("Input video file", fileName, "doesn't exist")
    sys.exit(1)

fps = FPS().start()
while cv.waitKey(1) < 0:
    # frame에 video의 frame copy 한다
    # 연산을 최소화하기 위해 frame을 resizing을 한다
    hasFrame, frame = cap.read()

    # frame 없으면 종료 한다
    if not hasFrame:
        print("Video ended")
        fps.stop()
        print("elapsed time {}".format(fps.elapsed()))
        print("approximate FPS {}".format(fps.fps()))
        cv.waitKey(100)
        # device release 한다
        cap.release()
        break

    # frame = imutils.resize(frame, width=450)

    boxes, pred_cls = get_prediction(frame, threshold=0.8)
    # img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv.rectangle(frame, boxes[i][0], boxes[i][1], (255, 178, 50), 3)
        cv.putText(frame, pred_cls[i], boxes[i][0], cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

    cv.imshow(winName, frame)
    fps.update()
    if cv.waitKey(25) & 0xFF == ord('q'):
        fps.stop()
        print("{}".format(fps.fps()))
        cap.release()
        cv.destroyAllWindows()
        break


