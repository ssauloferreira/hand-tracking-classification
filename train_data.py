# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import modules
import imutils


from math import copysign, log10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from glob import glob
from pyefd import elliptic_fourier_descriptors as efd
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, \
                            recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

# !git clone https://github.com/saulolks/HandTracking

wsize_open = 21
wsize_gaussian = 13

y_min = 4
y_max = 137
crmin = 147
crmax = 186
cbmin = 0
cbmax = 125

batch_size = 64
epochs = 5

headcascade = cv2.CascadeClassifier('face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=30)
kernel = np.ones((wsize_open, wsize_open), np.uint8)
descriptor = 'fourier'
classifier = 'knn'
to_train = True


def efd_feature(contour):
    coeffs = efd(np.squeeze(contour), order=10, normalize=True)
    return coeffs.flatten()[3:]


sample = {'open': 1, 'zero': 2, 'palm': 3, 'pinch': 4,
          'gun': 5, 'ok': 6, 'point': 7, 'closed': 9}
marcel = {'Five': 4, 'V': 6, 'B': 2, 'A': 1, 'C': 3, 'Point': 5}

data = []
labels = []


def do_print(i, maxi):
    begin = "["
    middle = "|"*int((10*i)/maxi) + " "*int((10*(maxi-i))/maxi)
    end = "]"
    return begin + middle + end


if not data:
    if True:
        files = glob('Database/hgr1/**')

        count = 1
        maxi = len(files)

        # print('\n', dir)

        for filepath in files:

            # print(do_print(i, maxi)) if i % 10 == 0 else None
            # print(filepath)
            print(f"{count}/{maxi}")
            count += 1

            label = str(filepath).split('/')[-1][0]
            if label.isdigit():

                original = cv2.imread(filepath, -1)
                original = imutils.resize(original, 600)
                # cv2.imshow("original", original)

                preprocessed = modules.camera_module(source=original.copy())
                # cv2.imshow("dir", preprocessed)
                # cv2.waitKey()

                images, cnt, approx = modules\
                    .detection_module(source=preprocessed)

                """Descrição de imagem"""
                if cnt is not None:

                    hand_image = np.zeros((80, 80), np.uint8)
                    hand = cv2.resize(images[0], (80, 80),
                                      interpolation=cv2.INTER_NEAREST)
                    hand_image[:hand.shape[0], :hand.shape[1]] = hand

                    contours, _ = cv2.findContours(hand_image, 1, 2)

                    if len(contours) > 0:
                        epsilon = 0.01 * cv2.arcLength(contours[0], True)
                        approx = cv2.approxPolyDP(contours[0], epsilon, True)

                        # cv2.imshow("dir", hand_image)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()

                        if descriptor == "hu":
                            M = cv2.moments(hand_image)
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
                            data.append(efd_feature(contours[0]))
                        else:
                            M = cv2.moments(approx)

                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])

                            data.append(modules.signature((cX, cY),
                                                          contours[0]))

                        labels.append(label)

if to_train:
    input_shape = len(data[0])
    print(input_shape)
    print(f"Descritor: {descriptor}")
    print(f"Dataset: {len(data)} itens")
    # print(len(labels))

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2,
                                                        random_state=42)

    # model = modules.mlp(input_shape=input_shape)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=["accuracy"])

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    # scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    if classifier == 'svm':
        model = SVC(gamma='auto')
    elif classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=9)
    elif classifier == 'mlp':
        model = MLPClassifier(activation='relu', alpha=1e-05,
                              batch_size='auto', beta_1=0.9,
                              beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant',
                              learning_rate_init=0.001, max_iter=200,
                              momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5,
                              random_state=1, shuffle=True, solver='lbfgs',
                              tol=0.0001, validation_fraction=0.1,
                              verbose=False, warm_start=False)

    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, predict))
    print("Precision: ", f1_score(y_test, predict, average='micro'))
    print("Recall: ", recall_score(y_test, predict, average='micro'))
    confMatrix = confusion_matrix(y_test, predict)
    print(confMatrix)
