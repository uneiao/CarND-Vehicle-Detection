#! /usr/bin/python
# -*- coding:utf8 -*-


def get_images(sample_size):
    images_pos = glob.glob("vehicles/*/*.png")
    images_neg = glob.glob("non-vihicles/*/*.png")
    cars = []
    notcars = []
    for image in images_pos:
        cars.append(image)

    for image in images_neg:
        notcars.append(image)

    cars = cars[0: sample_size]
    notcars = notcars[0: sample_size]
    return cars, notcars


def train_svm():
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    pass


if __name__ == "__main__":
    train_svm()
