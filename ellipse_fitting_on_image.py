import numpy as np
import cv2
from src.ellipse_fitting import fit_ellipse3

# Equation defining the ellipse
# X = (x^2, xy, y^2, x, y, 1)
# A = (a, b, c, d, e, f)
# Return value:
#  > 0 ->  point outside the ellipse
#  = 0 ->  point on the ellipse
#  < 0 ->  point inside the ellipse
def model_ellipse(X, A):
    return X.dot(A)

# Read in image and perform adaptive thresholding
img = cv2.imread("images/calib.PNG")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                 # Convert the input image to a grayscale
img_threshold = cv2.adaptiveThreshold(src=img_gray, maxValue=255, \
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=9)

# retrieve connected components in the image
# labels array shows pixel correspondence to a connected component
_, labels, stats, centroids = cv2.connectedComponentsWithStats(img_threshold)

# Set index of connected component to extract
cc_index = 2

# show connected component
while not cv2.waitKey(1) > 0:
    img_display = cv2.cvtColor(img_threshold.copy(), cv2.COLOR_GRAY2BGR)
    x = stats[cc_index, cv2.CC_STAT_LEFT]
    y = stats[cc_index, cv2.CC_STAT_TOP]
    w = stats[cc_index, cv2.CC_STAT_WIDTH]
    h = stats[cc_index, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Ellipse to fit", img_display)

# Extract points of the connected component and fit ellipse
X, Y = np.nonzero(labels==cc_index)
points = np.hstack([Y[:,np.newaxis],X[:,np.newaxis]])
e = fit_ellipse3(points)

# Display pixels that lie inside extracted ellipse
X, Y = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
Xf = X.flatten()
Yf = Y.flatten()
Zf = model_ellipse(np.asarray([Xf**2, Xf*Yf, Yf**2, Xf, Yf, np.ones_like(Xf)]).T, e).flatten()
Z = np.reshape(Zf, X.shape)

while not cv2.waitKey(1) > 0:
    img_display = cv2.cvtColor(np.float32((np.abs(Z)<0.0005)*255), cv2.COLOR_GRAY2BGR)
    cv2.imshow("Ellipse Fitted", img_display)

