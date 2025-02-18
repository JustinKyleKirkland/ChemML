import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def pil_image_to_pixmap(pil_image):
	# Convert the PIL image to a NumPy array
	img_array = np.array(pil_image)

	# Check if the image has an alpha channel
	if img_array.shape[2] == 4:  # If RGBA
		img_array = img_array[..., [2, 1, 0, 3]]  # Convert RGBA to BGRA
	else:  # If RGB
		img_array = np.dstack((img_array, np.full((img_array.shape[0], img_array.shape[1]), 255)))  # Add alpha channel

	# Create a QImage from the NumPy array
	height, width, channel = img_array.shape
	bytes_per_line = 4 * width
	q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_ARGB32)

	# Convert QImage to QPixmap
	return QPixmap.fromImage(q_image)
