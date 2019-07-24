# coding: utf-8

from __future__ import division, print_function

import cv2


def draw_bounding_box_yolo(img, coord, label=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    '''
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, (255, 178, 50), 3)
    if label:
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(int(coord[1]), label_size[1])
        cv2.rectangle(img, (int(coord[0]), top - round(1.5 * label_size[1])),
                      (int(coord[0]) + round(1.5 * label_size[0]), top + base_line), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, label, (int(coord[0]), top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)


