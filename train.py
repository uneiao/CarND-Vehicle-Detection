#! /usr/bin/python
# -*- coding:utf8 -*-


import glob
import time
import pickle
import random
import feature
import numpy as np


def get_images(sample_size):
    images_pos = glob.glob("vehicles/*/*.png")
    images_neg = glob.glob("non-vehicles/*/*.png")
    cars = []
    notcars = []

    cars = random.sample(images_pos, sample_size)
    notcars = random.sample(images_neg, sample_size)
    return cars, notcars


NAME2FEATURE_ = lambda img: feature.extract_features(
    img, color_space="HSV", hog_channel="GRAY", spatial_feat=False)


FEATURE_ = lambda img: feature.single_img_features(
    img, color_space="HSV", hog_channel="GRAY", spatial_feat=False)


def get_samples(sample_size):
    # Define the labels vector
    car_images, notcar_images = get_images(sample_size)
    car_features = NAME2FEATURE_(car_images)
    notcar_features = NAME2FEATURE_(notcar_images)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    return X_train, X_test, y_train, y_test, X_scaler


def train_linear_svc():
    X_train, X_test, y_train, y_test, X_scaler = get_samples(8000)
    from sklearn.svm import LinearSVC
    model = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4))
    pickle.dump(model, open("svc.model", "wb"))
    pickle.dump(X_scaler, open("x.scaler", "wb"))


def load_svc_model():
    return pickle.load(open("svc.model", "rb"))


def load_scaler():
    return pickle.load(open("x.scaler", "rb"))


if __name__ == "__main__":
    train_linear_svc()
