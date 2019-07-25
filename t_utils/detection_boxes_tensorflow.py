# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
# Set headless-friendly backend.
import matplotlib;

import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import six
import tensorflow as tf
import cv2

from colors import *

matplotlib.use('Agg')  # pylint: disable=multiple-statements

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
    """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
    """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_bounding_box_on_image(image,
                               coord,
                               display_str_list,
                               color=(0, 0, 0),
                               score=0,
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: cv2 frame
    coord: [ymin, xmin, ymax, xmax] of bounding box
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    color: color to draw bounding box. Default is black.
    score: score of detected box. If not yolo, the default is 0.
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """

    im_height = image.shape[0]
    im_width = image.shape[1]
    if use_normalized_coordinates:
        normalized_coord = [coord[0] * im_height, coord[1] * im_width,
                            coord[2] * im_height, coord[3] * im_width]
        (top, left, bottom, right) = list(map(int, normalized_coord))
    else:
        (top, left, bottom, right) = list(map(lambda x: int(x), coord))

    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=3)

    if use_normalized_coordinates and score == 0:
        draw_label_box(image, display_str_list[0], top, left, color)
    else:
        label = '{} {}%'.format(display_str_list, round((score * 100), 1))
        draw_label_box(image, label, top, left, color)


def draw_label_box(image, display_str, top, left, color):
    txt_color = (0, 0, 0)
    if sum(color) < 500:
        txt_color = (255, 255, 255)

    label_size, base_line = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.rectangle(image, (left, top - round(1.5 * label_size[1])),
                  (left + round(1.5 * label_size[0]), top + base_line), color=color, thickness=cv2.FILLED)
    cv2.putText(image, display_str, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=txt_color, thickness=2)


def _resize_original_image(image, image_shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(
        image,
        image_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=True)
    return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
    """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
    """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_rcnn(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=50,
        min_score_thresh=.5,
        groundtruth_box_visualization_color='black'):
    """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}

    # if max_boxes_to_draw is not defined, use number of detected boxes
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]

    # use smaller number between max_boxes_to_draw and detected boxes
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):

        if scores is None or scores[i] > min_score_thresh:
            # score of detected box is higher than threshold
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:

                # assign value to class_name if exists in category_index dictionary
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                display_str = str(class_name)
                if not display_str:
                    display_str = '{}%'.format(round(100 * scores[i], 1))
                else:
                    # if class_name and score both exist
                    display_str = '{}: {}%'.format(display_str, round(100 * scores[i], 1))
                # append label string ex) car: 90%
                box_to_display_str_map[box].append(display_str)

                # give unique color to a specific class
                box_to_color_map[box] = STANDARD_COLORS[
                    classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    # Tuple of coord are key of box_to_color_map dictionary
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        # mask exists
        if instance_masks is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_masks_map[box],
                color=color
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_boundaries_map[box],
                color='red',
                alpha=1.0
            )
        # actually draws bounding box
        draw_bounding_box_on_image(
            image,
            [ymin, xmin, ymax, xmax],
            box_to_display_str_map[box],
            color=color,
            use_normalized_coordinates=use_normalized_coordinates)

    return image


def visualize_boxes_and_labels_yolo(frame, boxes_, classes, labels_, scores_, use_normalized_coordinates):
    """Overlay labeled boxes on an image with formatted scores and label names.

     This function groups boxes that correspond to the same location
     and creates a display string for each detection and overlays these
     on the image. Note that this function modifies the image in place, and returns
     that same image.

     Args:
       frame: cv2 frame
       boxes_: a numpy array of shape [N, 4]
       classes: a numpy array of shape [N]. Note that class indices are 1-based,
         and match the keys in the label map.
       labels_: a numpy array of shape [N]
       scores_: a numpy array of shape [N] or None.
       use_normalized_coordinates: whether boxes is to be interpreted as
         normalized coordinates or not.
     """
    # Visualization of the results of a detection
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        # give unique color to a specific class
        color = STANDARD_COLORS[
            labels_[i] % len(STANDARD_COLORS)]
        draw_bounding_box_on_image(frame,
                                   [y0, x0, y1, x1],
                                   classes[labels_[i]],
                                   color,
                                   score=scores_[i],
                                   use_normalized_coordinates=use_normalized_coordinates)


def add_cdf_image_summary(values, name):
    """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """

    def cdf_plot(values):
        """Numpy function to plot CDF."""
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                                / cumulative_values.size)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
        return image

    cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
    tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
    """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """

    def hist_plot(values, bins):
        """Numpy function to plot hist."""
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        y, x = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(
            fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
        return image

    hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
    tf.summary.image(name, hist_plot)
