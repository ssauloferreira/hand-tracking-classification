import cv2
import numpy as np
from keras import Sequential
from keras.layers import Dropout, Dense, Activation
from scipy.spatial import distance

wsize_open = 15
wsize_gaussian = 15

y_min = 0
y_max = 137
crmin = 139
crmax = 174
cbmin = 0
cbmax = 125

# y_min = 4
# y_max = 137
# crmin = 147
# crmax = 186
# cbmin = 0
# cbmax = 125

headcascade = cv2.CascadeClassifier('face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=30)
kernel = np.ones((wsize_open, wsize_open), np.uint8)


def signature(center, points):
    description = []

    for cnt in points:
        dist = distance.euclidean(cnt[0][0], center)
        description.append(dist)

    description.sort(reverse=True)
    description = description[:50]
    # print(description)

    return description


def camera_module(source):
    image = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

    image_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    faces = headcascade.detectMultiScale(image_gray, 1.3, 5)

    """### Divide imagem YCrCb em 3 canais distintos"""

    y, cr, cb = cv2.split(image)

    """### Binarização"""

    _, y = cv2.threshold(y, y_min, y_max, cv2.THRESH_BINARY)
    _, cr = cv2.threshold(cr, crmin, crmax, cv2.THRESH_BINARY)
    _, cb = cv2.threshold(cb, cbmin, cbmax, cv2.THRESH_BINARY)

    """### Morfologia
    Utiliza transformações morfológicas de abertura para
    remoção de pequenos artefatos na imagem.
    """

    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
    cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
    cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)

    """### Juntando os filtros e binarizando"""

    img = cv2.merge((y, cr, cb))
    img1 = img.copy()

    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, img3 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)

    """### Eliminação de ruídos e face"""

    # mask = backSub.apply(image)
    # img4 = cv2.bitwise_and(mask, img3)
    img5 = cv2.medianBlur(img3, wsize_gaussian)
    img5 = cv2.morphologyEx(img5, cv2.MORPH_OPEN, kernel)

    marge = 30
    for (x, y, w, h) in faces:
        cv2.rectangle(img5, (x - marge, y - h + marge),
                      (x + marge + w, y + marge + h), (0, 0, 0), -1)

    return img5


def detection_module(source):
    segment = source.copy()
    segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
    boundbox = source.copy()
    boundbox = cv2.cvtColor(boundbox, cv2.COLOR_GRAY2BGR)
    binary = source.copy()
    hand_only = None
    cnt = None
    approx = None

    contours, _ = cv2.findContours(binary, 1, 2)

    if len(contours) > 0:
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(approx)
        x, y, w, h = cv2.boundingRect(hull)

        # x = x-10 if x > 10 else x
        # y = y-10 if y > 10 else y
        # w = w+10 if w < segment.shape[0] - 10 else w
        # h = h+10 if h < segment.shape[1] - 10 else h

        hand_only = binary[y:y + h, x:x + w]
        cv2.drawContours(segment, [approx], -1, (0, 0, 255), 5)
        cv2.rectangle(boundbox, (x, y), (x + w, y + h), (255, 0, 0),  5)

    return (hand_only, segment, boundbox), cnt, approx


def mlp(input_shape, num_layers=1000):
    a = num_layers
    b = int(num_layers/2)
    c = int(num_layers/10)

    model = Sequential()
    model.add(Dense(a, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(b))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(c))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
