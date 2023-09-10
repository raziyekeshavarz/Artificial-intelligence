import cv2
from gaussion import gaussian
from gradient import gradient
from nonmax_suppression import maximum
from double_thresholding import thresholding
import matplotlib.pyplot as plt

# select image
IMAGE_FILE = './data/lena.jpeg'

# Read image and convert it to gray
image = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)

# Gaussian
gImage = gaussian(image)

# Gradient
G, theta = gradient(gImage)

# non-maximum
nonMax = maximum(G, theta)

# double threshold
result, _ = thresholding(nonMax)

# TODO: edge tracking

# Show result
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')

plt.show()

import cv2
from gaussion import gaussian
from gradient import gradient
from nonmax_suppression import maximum
from double_thresholding import thresholding
import matplotlib.pyplot as plt

# select image
IMAGE_FILE = './data/lena.jpeg'

# Read image and convert it to gray
image = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)

# Gaussian
gImage = gaussian(image)

# Gradient
G, theta = gradient(gImage)

# non-maximum
nonMax = maximum(G, theta)

# double threshold
result, _ = thresholding(nonMax)

# TODO: edge tracking

# Show result
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')

plt.show()
