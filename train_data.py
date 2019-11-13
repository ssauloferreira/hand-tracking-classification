# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import modules
from math import copysign, log10
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from glob import glob
from pyefd import elliptic_fourier_descriptors as efd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
descriptor = 'hog'
to_train = True


def efd_feature(contour):
    coeffs = efd(np.squeeze(contour), order=10, normalize=True)
    return coeffs.flatten()[3:]


sample = {'open': 1, 'zero': 2, 'palm': 3, 'pinch': 4,
          'gun': 5, 'ok': 6, 'point': 7, 'closed': 9}
marcel = {'A': 1, 'B': 2, 'C': 3, 'Five': 4, 'Point': 5, 'V': 6}

data = []
labels = []


def do_print(i, maxi):
    begin = "["
    middle = "|"*int((10*i)/maxi) + " "*int((10*(maxi-i))/maxi)
    end = "]"
    return begin + middle + end


if not data:
    for dir in sample:
        files = glob('Database/sample/'+dir+'/**')

        i = 1
        maxi = len(files)
        label = sample[dir]

        print('\n', dir)

        for filepath in files:
            print(do_print(i, maxi)) if i % 50 == 0 else None
            i += 1
            original = cv2.imread(filepath, -1)

            preprocessed = modules.camera_module(source=original.copy())
            images, cnt, approx = modules.detection_module(source=preprocessed)

            """Descrição de imagem"""
            if cnt is not None:
                cv2.imshow(dir, images[0])
                cv2.waitKey()
                cv2.destroyAllWindows()

                hand_image = np.zeros((80, 80), np.uint8)
                hand = cv2.resize(images[0], (80, 80),
                                  interpolation=cv2.INTER_NEAREST)
                hand_image[:hand.shape[0], :hand.shape[1]] = hand

                if descriptor == "hu":
                    M = cv2.moments(images[0])
                    huMoments = cv2.HuMoments(M).flatten()
                    for i in range(0, 7):
                        huMoments[i] = -1 * copysign(1.0, huMoments[i]) *\
                            log10(abs(huMoments[i]))
                    data.append(huMoments)

                elif descriptor == "hog":
                    hog_features = hog(hand_image, orientations=8,
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(10, 10),
                                       feature_vector=True)
                    data.append(hog_features)

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
    print(input_shape)
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.3,
                                                        random_state=42)
    print(x_train)
    print(y_train)
    # model = modules.mlp(input_shape=input_shape)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=["accuracy"])

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    # scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, predict))
