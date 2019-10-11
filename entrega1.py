import cv2
import numpy as np
#==============================================================================================
wsize_open = 3
wsize_gaussian = 5

y_min = 54
y_max = 137
crmin = 135
crmax = 174
cbmin = 0 
cbmax = 125
#=============================================================================================
headcascade = cv2.CascadeClassifier('./face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=30)
kernel = np.ones((wsize_open, wsize_open), np.uint8)
#=============================================================================================
original = cv2.imread('')
image = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
# ------------------------ remove face -------------------------------
image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
faces = headcascade.detectMultiScale(image_gray, 1.3, 5)
# ------------------- splitting into 3 channels ----------------------
y, cr, cb = cv2.split(image)
# ------------------------- binarization -----------------------------
_, y = cv2.threshold(y, ymin, ymax, cv2.THRESH_BINARY)
_, cr = cv2.threshold(cr, crmin, crmax, cv2.THRESH_BINARY)
_, cb = cv2.threshold(cb, cbmin, cbmax, cv2.THRESH_BINARY)
# ------------------------- morphology -------------------------------
y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)
# ---------------------------  merge ---------------------------------
img = y + cr + cb
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

mask = backSub.apply(image)
img = cv2.bitwise_and(mask, img)
img = cv2.medianBlur(img, wsize_gaussian)
img = remove_concomponent(img=img, min_value=min_value)
# ------------------------- removing face ---------------------------
marge = 30
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x - marge, y - h + marge), (x + marge + w, y + marge + h), (0, 0, 0), -1)

