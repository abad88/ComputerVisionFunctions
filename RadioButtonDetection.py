import cv2
import numpy as np
import glob

ref_folder = "D:/ML_Questions/Data"

ref_data = glob.glob(ref_folder+"/*.*")

for i in ref_data:
    print(i)

    # Read image.
    img = cv2.imread(i, cv2.IMREAD_COLOR)

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using a 3x3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=20, minRadius=8, maxRadius=15
    )

    # Draw circles that are detected and filled.
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Create a mask to check the filled area of the circle.
            mask = np.zeros_like(gray)
            cv2.circle(mask, (a, b), r, 255, -1)

            # Calculate the contour area.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area = cv2.contourArea(contours[0])

            # Draw the filled circle if the contour area matches the expected area.
            if abs(contour_area - np.pi * r ** 2) < 48:  # Adjust the threshold as needed.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

                # Print the coordinates of the circle.
                print("Center: ({}, {})".format(a, b))
                print("Radius:", r)

                # Count the black pixels within the circle region.
                circle_region = gray[b - r : b + r, a - r : a + r]
                black_pixel_count = np.sum(circle_region == 0)
                print("Black Pixel Count:", black_pixel_count)

    # Display the image with filled circles.
    cv2.imshow("Detected Filled Circles", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
