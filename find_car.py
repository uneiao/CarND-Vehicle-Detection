#! /usr/bin/python
# -*- coding:utf8 -*-


import glob
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

import heat_map
import search
import train


class VehicleTrack():
    def __init__(self):
        self._model = None
        self._scaler = None
        self._heat_tracking = None
        pass

    def save_image(self, name, image):
        pass

    def get_model_and_scaler(self):
        self._model = train.load_svc_model()
        self._scaler = train.load_scaler()

    def find_cars_in_image(self, img, ystart=360, ystop=720, pix_per_cell=8, cell_per_block=8):
        draw_img = np.copy(img)

        img_tosearch = img[ystart:ystop,:,:]
        height, width, _ = img_tosearch.shape

        # Define blocks and steps as above
        nxblocks = (width // pix_per_cell) - cell_per_block + 1
        nyblocks = (height // pix_per_cell) - cell_per_block + 1
        #nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                # Scale features and make a prediction
                test_features = self._scaler.transform(train.FEATURE_(subimg).reshape(1, -1))
                test_prediction = self._model.predict(test_features)

                if test_prediction == 1:
                    scale = 1
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,
                        (xbox_left, ytop_draw+ystart),
                        (xbox_left+win_draw, ytop_draw+win_draw+ystart),
                        (0,0,255), 6)

        cv2.imshow("findcar", draw_img)
        cv2.waitKey()

    def process_image(self):
        self.get_model_and_scaler()
        import matplotlib.pyplot as plt
        import pylab
        fig = plt.figure()
        ind = 231
        for image_path in glob.glob("test_images/*.jpg"):
            image = cv2.imread(image_path)
            windows = self.multi_scale_search(image)
            draw_img = search.draw_boxes(image, windows)
            name = image_path.split("/")[1].split(".")[0]
            cv2.imwrite("%s_output.jpg" % name, draw_img)
            self._heat_tracking = None
            plt.subplot(ind)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cmap="gray")
            plt.title("%s_output.jpg" % name)
            ind += 1
        pylab.show()

    def process_video(self, video_file_path):
        self.get_model_and_scaler()
        def process_image(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            windows = self.multi_scale_search(image)
            if self._heat_tracking is None:
                self._heat_tracking = heat_map.HeatTracking(
                    (image.shape[0], image.shape[1]),
                    weight_decayed=0.5,
                    threshold=0.6)
            self._heat_tracking.add_heat_frame(windows)
            labels = self._heat_tracking.get_heats_label()
            draw_img = heat_map.draw_labeled_bboxes(image, labels)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
            return draw_img
        video_output = "%s_output.mp4" % video_file_path.split(".")[0]
        clip1 = VideoFileClip(video_file_path)
        _clip = clip1.fl_image(process_image)
        _clip.write_videofile(video_output, audio=False)

    def multi_scale_search(self, image, draw=False):
        search_scale_settings = [
            ([None, None], [400, 470], (64, 64), (0.75, 0.75)),
            ([None, None], [410, 520], (96, 96), (0.75, 0.75)),
            ([None, None], [420, 660], (128, 128), (0.75, 0.75)),
        ]
        if draw:
            draw_image = np.copy(image)
        match_windows = []
        for x_start_stop, y_start_stop, xy_window, xy_overlap in search_scale_settings:
            windows = search.slide_window(
                image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                xy_window=xy_window, xy_overlap=xy_overlap)

            hot_windows = search.search_windows(image, windows, self._model, self._scaler, train.FEATURE_)
            match_windows.extend(hot_windows)

            if draw:
                window_img = search.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)

        if draw:
            cv2.imshow("multi_scale", window_img)
            cv2.waitKey()

        return match_windows


if __name__ == "__main__":
    vt = VehicleTrack()
    #vt.process_image()
    vt.process_video("project_video.mp4")
