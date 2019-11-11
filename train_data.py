# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np

import modules
from skimage.feature import hog
from glob import glob
from pyefd import elliptic_fourier_descriptors as efd

# !git clone https://github.com/saulolks/HandTracking

wsize_open = 13
wsize_gaussian = 13

y_min = 54
y_max = 137
crmin = 135
crmax = 174
cbmin = 80
cbmax = 125

batch_size = 64
epochs = 5

headcascade = cv2.CascadeClassifier('face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=30)
kernel = np.ones((wsize_open, wsize_open), np.uint8)
descriptor = 'fourier'
to_train = False


def efd_feature(contour):
    coeffs = efd(np.squeeze(contour), order=10, normalize=True)
    return coeffs.flatten()[3:]


sample = ['open', 'zero', 'palm', 'pinch', 'gun',
          'ok', 'point', 'pinch', 'closed']

data = []
labels = []
data_test = []
labels_test = []

if not data:
    for dir in sample:
        count = 0
        label = dir
        print(dir)
        for filepath in glob('Database/sample/'+dir+'/**'):
            original = cv2.imread(filepath, -1)

            preprocessed = modules.camera_module(source=original.copy())
            images, cnt, approx = modules.detection_module(source=preprocessed)

            """Descrição de imagem"""
            if descriptor == "hu":
                pass

            elif descriptor == "hog":
                hog_features = hog(images[0], orientations=8,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(10, 10),
                                   feature_vector=True)
                data.append(hog_features)
                print(hog_features)

            elif descriptor == "fourier":
                data.append(efd_feature(cnt))

            else:
                M = cv2.moments(approx)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                data.append(modules.signature((cX, cY), cnt))

            labels.append(label)

if to_train:
    input_shape = len(data[0])
    model = modules.mlp(input_shape=input_shape)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=["accuracy"])

    model.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=1)
    scores = model.evaluate(data_test, labels_test,
                            verbose=0, batch_size=batch_size)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
