import numpy as np
import cv2
import os

# ================Useful Variables===========
DATA_PATH = './data'
TEXT_FILE_PATH = os.path.join(DATA_PATH, 'text.png')
A_PATH = os.path.join(DATA_PATH, 'a.png')
A_ROTATE_PATH = os.path.join(DATA_PATH, 'a-rotate.png')
OFFSET = 2

# ==================Read Image===================
text = cv2.imread(TEXT_FILE_PATH)
gray = cv2.cvtColor(text, cv2.COLOR_RGB2GRAY)
a = cv2.imread(A_PATH, cv2.IMREAD_GRAYSCALE)
a_rotate = cv2.imread(A_ROTATE_PATH, cv2.IMREAD_GRAYSCALE)


def frequency_domain():
    print("Frequency Domain")

    def matching(template, color):

        # Width and Height of each Template
        template_w, template_h = template.shape[::-1]

        # Calculate FFT of each Image (Main Image and Templates)
        text_fft = np.fft.fft2(gray)
        template_fft = np.fft.fft2(template, gray.shape)

        template_phase_correlation = np.real(
            np.fft.ifft2(text_fft * np.conj(template_fft) / np.abs(text_fft * np.conj(template_fft)))
        )

        # Recognition
        peak = list(
            zip(*np.unravel_index(
                np.argsort(template_phase_correlation, axis=None),
                template_phase_correlation.shape))
        )[::-1]

        for index, point in enumerate(peak):
            point = point[::-1]
            if index > 2:
                break
            cv2.rectangle(
                text,
                (point[0] - OFFSET, point[1] - OFFSET),
                (point[0] + template_w + OFFSET, point[1] + template_h + OFFSET),
                color,
                2
            )

    matching(a, (0, 255, 0))
    matching(a_rotate, (0, 0, 255))

    # Show Image
    cv2.imshow('Frequency Domain', text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def spatial_domain():
    def matching(template, color):
        w, h = template.shape[::-1]
        match = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.7
        loc = np.where(match >= threshold)

        for point in zip(*loc[::-1]):
            cv2.rectangle(
                text,
                (point[0] - OFFSET, point[1] - OFFSET),
                (point[0] + w + OFFSET, point[1] + h + OFFSET),
                color,
                2
            )

    matching(a, (0, 255, 0))
    matching(a_rotate, (0, 0, 255))

    cv2.imshow('Spatial Domain', text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frequency_domain()
    spatial_domain()
