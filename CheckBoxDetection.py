import pytictoc
from boxdetect import config
from boxdetect.pipelines import get_boxes, get_checkboxes
import cv2
import matplotlib.pyplot as plt
import numpy as np

def checked_boxes(i):
    # Start timer using tic()
    tic = pytictoc.TicToc()
    tic.tic()

    cfg = config.PipelinesConfig()

    # Important to adjust these values to match the size of boxes on your image
    cfg.width_range = (16, 30)
    cfg.height_range = (14, 30)

    # The more scaling factors the more accurate the results, but also it takes more time to process
    # Too small scaling factor may cause false positives
    # Too big scaling factor will take a lot of processing time
    cfg.scaling_factors = [6.0]

    # w/h ratio range for boxes/rectangles filtering
    cfg.wh_ratio_range = (0.5, 1.7)

    # group_size_range starting from 2 will skip all the groups
    # with a single box detected inside (like checkboxes)
    cfg.group_size_range = (1, 1)

    # Num of iterations when running dilation transformation (to enhance the image)
    cfg.dilation_iterations = 0

    # Detect boxes and group them
    rects, grouping_rects, image, output_image = get_boxes(
        i, cfg=cfg, plot=False)

    print("All boxes found:", grouping_rects)

    # Uncomment the following lines to see the output image
    plt.figure(figsize=(20, 20))
    plt.imshow(output_image)
    plt.show()

# Load the image and run the function
img = cv2.imread("D:/ML_Questions/Data/dell_3-0056-press-DOWN.png")
checked_boxes(img)
