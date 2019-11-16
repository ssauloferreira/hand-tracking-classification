# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np

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


def nothing(pos):
    pass


source = cv2.imread('Database/marcel/Five/Five-train001.ppm', -1)

while True:
    cv2.namedWindow('YRB_calib')
    cv2.createTrackbar('Ymin', 'YRB_calib', 54, 255, nothing)
    cv2.createTrackbar('Ymax', 'YRB_calib', 137, 255, nothing)
    cv2.createTrackbar('CRmin', 'YRB_calib', 135, 255, nothing)
    cv2.createTrackbar('CRmax', 'YRB_calib', 174, 255, nothing)
    cv2.createTrackbar('CBmin', 'YRB_calib', 0, 255, nothing)
    cv2.createTrackbar('CBmax', 'YRB_calib', 125, 255, nothing)
    cv2.namedWindow('Windows sizes')
    cv2.createTrackbar('OpenSize', 'Windows sizes', 3, 10, nothing)
    cv2.createTrackbar('Gaussian', 'Windows sizes', 5, 30, nothing)
    cv2.createTrackbar('Connected', 'Windows sizes', 150, 500, nothing)

    ymin = cv2.getTrackbarPos('Ymin', 'YRB_calib')
    ymax = cv2.getTrackbarPos('Ymax', 'YRB_calib')
    crmin = cv2.getTrackbarPos('CRmin', 'YRB_calib')
    crmax = cv2.getTrackbarPos('CRmax', 'YRB_calib')
    cbmin = cv2.getTrackbarPos('CBmin', 'YRB_calib')
    cbmax = cv2.getTrackbarPos('CBmax', 'YRB_calib')
    wsize_open = cv2.getTrackbarPos('OpenSize', 'Windows sizes')
    wsize_gaussian = cv2.getTrackbarPos('Gaussian', 'Windows sizes') + 1
    wsize_gaussian = wsize_gaussian if wsize_gaussian % 2 == 1 \
        else wsize_gaussian + 1

    image = cv2.cvtColor(source.copy(), cv2.COLOR_BGR2YCR_CB)

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

    mask = backSub.apply(image)
    img4 = cv2.bitwise_and(mask, img3)
    img5 = cv2.medianBlur(img4, wsize_gaussian)
    img5 = cv2.morphologyEx(img5, cv2.MORPH_OPEN, kernel)

    marge = 30
    for (x, y, w, h) in faces:
        cv2.rectangle(img5, (x - marge, y - h + marge),
                      (x + marge + w, y + marge + h), (0, 0, 0), -1)

    cv2.imshow("", img5)
    cv2.imshow("", cr)
