# -*- coding:utf8 -*-


import numpy as np
import cv2
from scipy.ndimage.measurements import label


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img


class HeatTracking():

    def __init__(self, shape, threshold=0.6, weight_decayed=0.5):
        # Looking into 5 recent frames
        self._history_len = 5
        self._frames = []
        self._shape = shape
        self._threshold = threshold
        self._weights = []

        # Initialize with a list of heat
        for i in range(self._history_len):
            self._frames.append(self.zero_heat())

        # Initialize weights
        for i in range(self._history_len):
            self._weights.insert(0, weight_decayed ** (i + 1))

    def zero_heat(self):
        return np.zeros(self._shape).astype(np.float32)

    def add_heat_frame(self, windows):
        new_heat = self.zero_heat()
        for window in windows:
            new_heat[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        self._frames.append(new_heat)
        self._frames.pop(0)

    def apply_threshold(self, heatmap):
        heatmap[heatmap <= self._threshold] = 0
        return heatmap

    def get_heats_label(self):
        # Get the weighted average of heat frames
        weighted_sum = np.average(self._frames, axis=0, weights=self._weights)
        # thresholded
        res = self.apply_threshold(weighted_sum)
        labels = label(res)
        return labels
