import os
import sys
import cv2
import imutils
import numpy as np

from tqdm import tqdm
from enum import Enum
from math import sqrt
from time import time

# Define a custom ENUM to specify which text-extraction function should be executed
# later on on the last step of the pipeline
class FTYPE(Enum):
    ALL = 0
    EXACT = 1
    SMOOTH = 2
    BINARY = 3
    GRAYSCALE = 4
    SINGLECHAR = 5

# If the user should choose SINGLECHAR as type function, he should also specify
# from which kind of processed plate he'd like to extract the single character.
class STYPE(Enum):
    BINARY = 1
    EXACT = 2

# We now define the main class of this project, the PlateExtractor.
# This class will hold inside the methods which compose the pipeline.
class PlateExtractor:

    #### 1) PREPROCESSING
    # Function that, given a plate image, will resize it to our standard (80x240),
    # approximating the result with the best interpolation possibile.
    def optimal_resize(self, plate):

        # Check if we need to enlarge the image or shrink it, because in those different
        # scenario we'll need different interpolation methods.
        if plate.shape[0] < 80 or plate.shape[1] < 240:

            # If the image (row,col) is lesser than the maximum dimension, we shrink it using
            # the cubic interpolation, whihc we'll give us the best result (at cost of
            # some computational speed).
            plate = cv2.resize(plate, (240,80), cv2.INTER_CUBIC)

        # Otherwise, if the image (row,col) exceed the maximum dimension
        elif plate.shape[0] > 80 or plate.shape[1] > 240:

            # we'll use the INTER_AREA interpolation.
            plate = cv2.resize(plate, (240,80), cv2.INTER_AREA)

        return plate

    # End

    # Simple function to display a given image with a given name
    def display_pipeline(self, name, plate):
        cv2.imshow(name, plate)
        cv2.waitKey(0)
        
    # End

    # Function that, given a plate image, will do various preprocessing operations to
    # optimally binarize and clarify the plate text in the image.
    def adaptive_preprocessing(self, plate, adaptive_bands=True, display=False):

        # If the display flag is set to true, we'll going to display the ENTIRE PIPELINE
        # of our methods. This will be the only comments explaining that.
        if display: self.display_pipeline("Raw Plate", plate)

        # Before going into the preprocessing, we need to scale our image ot a fixed
        # size of (80, 240) to be consistent in our steps.
        plate = self.optimal_resize(plate)

        # First thing, we convert the plate from RGB to HSV to generate a precise
        # blue mask that will be later applied to the grayscale image to cut
        # unnecessary elements from the whole plate image.
        hsv_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)

        # PIPELINE SHOW
        if display: self.display_pipeline("HSV Plate", hsv_plate)

        # define a custom range for lower and upper blue values in HSV space; this
        # was taken by analyze the specific type of image to preprocess, in our case
        # 240x80 matrixes that present the same scene, using a HSV Color Picker
        # made with OpenCV. That said, with the Saturation channel max range imposed
        # to 170, we're excluding most of the blues from the final image range.
        lower = np.array([0,0,0])
        upper = np.array([179,170,255])

        # We then use inRange, that checks if array elements lie between the elements
        # of two other arrays. In this case, our first array is the plate to threshold,
        # while the two others are the lower and upper blue ranges.
        # inRange check if a pixel (x,y) of the image is contained inside the range:
        # if so, it puts a 255 value inside a dst image to attest that the (x,y) pixel
        # result in the range of the lower/upper bound. If not, that means the pixel
        # (x,y) hasn't passed the test and so it contains a blue value we want to mask.
        mask_blue = cv2.inRange(hsv_plate, lower, upper)

        # PIPELINE SHOW
        if display: self.display_pipeline("Blue mask", mask_blue)

        # Using a simple and fairly strong median blur to suppress white noise into
        # the mask image
        mask_blue_filtered = cv2.medianBlur(mask_blue, 7)

        # PIPELINE SHOW
        if display: self.display_pipeline("Blue mask filtered", mask_blue_filtered)

        # Once the mask has been cleared, we fill eventual remaining holes in the mask
        # using a closing on the result of the median filtering. To do so, we use an
        # elliptical 7x7 kernel and perform an opening operation on the inverted mask.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_blue_filled = cv2.morphologyEx(mask_blue_filtered, cv2.MORPH_CLOSE, kernel)


        # PIPELINE SHOW
        if display: self.display_pipeline("Blue mask filled", mask_blue_filled)

        # Before moving on the pipeline, we're gonna use the blue mask to extract the outer
        # (x,y) coordinates of the left and right blue band, since those will be useful
        # for considering only the central white plate content excluding the blue bands.
        # First thing, we extrapolate the max height (rows) and width (cols) of the image.
        # There will be cases where the blue band will be present only at one side (i.e.:
        # GERMAN PLATES) or at both sides (i.e.: ITALIAN PLATES).
        max_height = mask_blue_filled.shape[0]
        max_width = mask_blue_filled.shape[1]

        # If we'd like to have the coordinates of the maximum extension and starting point
        # of the bands to be automatic:
        # Before entering in the adaptive_bands scope, we initialize the optimal left and right
        # coordinate relative to the band that will be modified in the if/else
        # Optimal left and right are equal to the most-inner left and most-outer right
        # coordinate (in our default case: 0 and 240)
        optimal_left_width = 0
        optimal_right_width = plate.shape[1]

        # We also initialize an optimal upper height and optimal lower height: that because
        # an image COULD NOT BE centered on a plate, but taking the surrounding car elements too.
        # With optimal height (lower and upper) we aim to extract ONLY the region containing the plate
        # alongisde with optimal width. Note that, since the image could instead contain ONLY the plate area,
        # we initialize those value at 0/max height and try to compute them with the adaptive bands logic.
        optimal_lower_height = 0
        optimal_upper_height = plate.shape[0]

        # If the user would like adaptive bands
        if adaptive_bands:

            # Here, we're going to use find contours to check where the blue bands lie
            # in terms of pixel coordinates of the image we're analyzing.
            # Before finding the contours, we'd like to superimpose a white frame on the blue
            # mask image to detach rectangle mask from the border, since we're going to use
            # findContours to fulfill our scope. We achieve this by using numpy slicing on the
            # rows as follow, to replace on the bottom and top row (0-max cols) white pixels.
            mask_blue_filled[0][0:max_width-1] = 255
            mask_blue_filled[max_height-1][0:max_width-1] = 255

            # We do the same on the leftmost and rightmost column, replacing their values
            # with blue pixels: we do this by a simple iteration for the sake of simplicity
            for i in range(max_height):
                mask_blue_filled[i][0] = 255
                mask_blue_filled[i][max_width-1] = 255

            # We now call findContours to find the contours of the mask element (that should be
            # composed by the two blue mask rectangle. Note that we could have a decomposed black
            # mask, so we could deal with more contours than the expected).
            mask_contours = cv2.findContours(mask_blue_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Using the grab_contours function we return instead of a multi-dimensional array
            # containing both contours and hierarchy only the contours of an image, correctly
            # organized by the nested relationship between them.
            mask_contours = imutils.grab_contours(mask_contours)

            # PIPELINE SHOW
            if display: self.display_pipeline("Framed blue mask", mask_blue_filled)

            # For drawing purposes in the debug section later on
            mask_blue_copy = None
            if display: mask_blue_copy = cv2.cvtColor(mask_blue_filled, cv2.COLOR_GRAY2BGR)

            # Now, we need to find the most inner width coordinate in which the blue boxes
            # extends. We do such thing to use those coordinate to exclude everything before
            # those coordinates, since they represent a blue band.
            # We initialize the midpoint and the leftmost and rightmost width coordinate.
            image_midpoint = round(plate.shape[1]/2)

            # For every contour in the found contours in the mask
            for contour in mask_contours:

                # extract the x,y,w,h coordinates
                [x,y,w,h] = cv2.boundingRect(contour)

                # Check for noise: after the blue mask post-processing (median and closing)
                # the contour should contains only the contour representing the blue bounding
                # boxes. That should not be true due to small noise areas remained. To skip those
                # areas, we calculate the area of the bounding box: if the area is lesser than
                # the 2% of the area (empirical), we're dealing with a potential noise: we just
                # skip that.
                if (w*h) < (2*(plate.shape[1]*plate.shape[0]))/100:
                    continue

                # check if the x coordinate (width) is placed left or right the midpoint
                # and if the box isn't the entire image (w=image width)
                if x > image_midpoint and w != plate.shape[1]:

                    # now check if the found coordinate is lesser than the right optimal width
                    # and if so, assign the new found coordinate: x alone is enough because
                    # it represents the starting point of the blue band.
                    if x <= optimal_right_width:
                        optimal_right_width = x
                        optimal_lower_height = y
                        optimal_upper_height = h

                # Else, x is lesser than the midpoint: it's then located left in the image;
                # assure, as above, that we're not dealing with the entire image contour box.
                elif x < image_midpoint and w != plate.shape[1]:

                    # As above, we use the inverse logic: if x is bigger than the current coordinate,
                    # we found a better approximation for the blue band coordinate. In this case,
                    # since the x alone represent the starting point and doesn't give us the extension
                    # information of the blue box, we need to sum x and w to obtain the real coordinate
                    # in which the blue band ends.
                    if x >= optimal_left_width:
                        optimal_left_width = x + w
                        optimal_lower_height = y
                        optimal_upper_height = h

                # DEBUG PURPOSES:
                if display:
                    print("CONTOUR BOX FOUND: {}".format(cv2.boundingRect(contour)))
                    cv2.rectangle(mask_blue_copy, (x,y), ((x + w), (y+h)+5), (0,255,0), 3)
                    cv2.imshow("contours", mask_blue_copy)
                    cv2.waitKey(0)

        else:
            # If not, set the default values relative to the italian plates: 25 and 215
            optimal_left_width = 25
            optimal_right_width = 215

        # OPTIONAL: Draw correct blue band mask rectangles: this has been proved to decrease
        # the quality of the output sometimes. We'll just use the adaptive coordinates found
        # in the extract contours to avoid taking elements before or past the found coordinates.
        if display:

            # DIsplay infos about coordinates
            print("Max extension of left band: {} - Starting point of right band: {}".format(
            optimal_left_width, optimal_right_width
            ))

            # Getting a copy
            copy_mask = mask_blue_filled.copy()

            # Draw left blue band: We'd like to draw the rectangle starting at (0,0) and finishing
            # at the optimal left width found with the maximum height possible (80 by default)
            cv2.rectangle(copy_mask, (0, 0), ((0 + optimal_left_width), (0+plate.shape[0])), 0, -1)

            # Draw right blue band: same concept here, but using the rightmost values
            cv2.rectangle(copy_mask, (optimal_right_width, 0), (plate.shape[1], plate.shape[0]), 0, -1)

            # Display it
            self.display_pipeline("Final Mask", copy_mask)

        # Now we're going to apply the mask on the plate: convert the latter to grayscale
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # PIPELINE SHOW
        if display: self.display_pipeline("Gray plate", gray_plate)

        # Normalize the grayscale image to enhance the dark and white areas
        cv2.normalize(gray_plate, gray_plate, 0, 255, cv2.NORM_MINMAX)

        # PIPELINE SHOW
        if display: self.display_pipeline("Gray plate normalized", gray_plate)

        # Apply the generated blue mask on the grayscale plate image
        gray_plate_masked = cv2.bitwise_and(gray_plate, gray_plate, mask=mask_blue_filled)

        # PIPELINE SHOW
        if display: self.display_pipeline("Gray plate masked", gray_plate_masked)

        # Now we need to use an adaptive threshold to generate a good approssimation
        # of the binarized image, useful to binarize text in an optimal way: we then
        # use adaptiveThreshold using GAUSSIAN_C as the gaussian summation of the neighborhood,
        # with a blocksize of 15 (neighborhood size) and a C constant equal to the
        # square of the standard deviation of the normalized grayscale image / 2, because
        # this value give us an adaptive way to threshold the image based on the content
        # of the analyzed plate (darker, brigther, blurred etc). We then invert the
        # binarized image to retrieve the text in white (more precise)
        binarized = cv2.adaptiveThreshold(gray_plate_masked,
                                          255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          15,
                                          sqrt(gray_plate_masked.std())/2)


        # PIPELINE SHOW
        if display: self.display_pipeline("Gray plate binarized", binarized)

        # To suppress eventual noise in the binarization, we do use another media filter,
        # this time fairly weak to delete white noise in the binarization.
        binarized_filtered = cv2.medianBlur(binarized, 3)

        # PIPELINE SHOW
        if display: self.display_pipeline("Binarized filtered", binarized_filtered)

        # Having a good filtered image, we now need to apply the blue mask computed earlier
        # to isolate even more the plate center containing the text, suppressing the remaining
        # holes containing useless details of the image.
        binarized_masked = cv2.bitwise_and(binarized_filtered, binarized_filtered, mask=mask_blue_filtered)

        # PIPELINE SHOW
        if display: self.display_pipeline("Binarized masked", binarized_masked)

        # We then return both the image filtered, binarized and with the mask applied on,
        # and the grayscale image for further analysis.
        return gray_plate, binarized_masked, (optimal_left_width, optimal_right_width, optimal_lower_height, optimal_upper_height)

    # END


    #### 2) CONTOURS EXTRACTION
    # Function that, given a correct preprocessed and binarized plate, will find the
    # contours of the text inside the image, generating a mask that will cover only
    # the plate characters identified. Note that, if precise_masking is set to true,
    # the contours of the final image will be, instead of the bounding box, the precise
    # contours of the characters into the image.
    def extract_contours(self, preprocessed_plate, precise_masking=False, band_coordinates=(25, 215), display=False):

        # Using the function findContours, we aim to extract contours and borders
        # from a given image. This function, since take in input the RETR_TREE flag,
        # will build a hierarchy of nested contours in form of an array, that will store
        # the relationship between bounds. Specifying only one returning elements we
        # store in contours both the contour of the input and the hierarchy, fundamental
        # for our next step.
        contours = cv2.findContours(preprocessed_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Using the grab_contours function we return instead of a multi-dimensional array
        # containing both contours and hierarchy only the contours of an image, correctly
        # organized by the nested relationship between them.
        contours = imutils.grab_contours(contours)

        # Optional: sorting contours to extract the first 10 elements (most sensed contours)
        # and then take the first 10 elements that should contains text border: this can cause
        # a OUT OF BOUNDS exception since some images can present 8 or 9 total contours.
        # contours = (sorted(contours, key = cv2.contourArea)[::-1])[:10]

        # Create a black image with same size of the input image: this will be useful to
        # draw filled white boxes using the contour coordinate. Those boxes will be useful
        # in the next step when we'll use this generated image as a mask to extract characters.
        text_mask = np.zeros((preprocessed_plate.shape[0], preprocessed_plate.shape[1]), dtype=np.uint8)

        # To achieve better modularity we'll store the character bounding box inside
        # an array.
        bounding_contours = []

        # Iterate through all the possible contour in the contours array
        for i,contour in enumerate(contours):

            # Given a set of coordinate points stored in a single contour object,
            # we tehn use boundingRect to calculates the up-right bounding rectangle of
            # a point set. This will return the x,y coordinate alongside with height and width
            # of a given contour point set.
            [x, y, w, h] = cv2.boundingRect(contour)

            # First thing, we check for meaningless contours: a meaningless contour is
            # just a contour that has nothing to do with the actual character contours:
            # those contours are the result of a not-too-much aggressive binarization,
            # that leave trails of white pixels on the left and right side of the plate.
            # To suppress those useless contours we just check the position found with
            # bounding rect of the contour analyzed. Remember that, in OpenCV, rows
            # and columns are inverted, so x = cols and y = rows. We then check if x
            # is lesser than a certain spatial position (left side) or exceed it from
            # the right. Since every plate got two blue band at it's side, we just
            # exclude a safe amount of pixel both from left and right (25 pixels is
            # a mean average found on some test images). Since our plate images are
            # of a fixed size of 80x240, we just add 25 + the first col and subtract
            # 25 - the last col, resulting in a min 25 and max 215 sensful bound.
            # Note that, if we override the band_coordinates with the adaptive blue band
            # coordinates found in the preprocessing, the result will be different based
            # on which extension of the blue bands found earlier.
            if x < band_coordinates[0] or x > band_coordinates[1]:
                # If the test not pass, the contour is meaningless and we can continue
                # to the next iteration
                continue

            # Now, since we got a bounding rectangle of a possible character contour,
            # we want to check if the size of this box is contained inside a specific
            # interval: this because, the plate extraction algorithm in this project
            # produces a costant image result in which characters of the plate got
            # similar size. With this knowledge we can then define a maximum and minimum
            # range in which a bounding box result in a potential character.
            # To assure that, we check the membership of widht and height to that interval.
            # NOTE: the interval was taken analyzing characters of various plate images.
            if 5 <= w <= 28 and 35 <= h <= 63:

            # Now, before performing whatever operation, we check which kind of contouring
            # we'd like to have: if precise_masking is set to True, that means we do not
            # want to draw bounding boxes around the characters but instead we want to draw
            # a precise mask of the character itself filled with a white solid color.
            # Instead, if precise_masking is set to False, we then use the given positions
            #`in terms of coordinate and width/height to draw a filled rectangle on the
            # text_mask black empty image, aimed to draw white boxes that later will be
            # used as a mask to extract characters from the plate.
                if precise_masking:
                    # If precise_masking is True, then we just draw a filled character
                    # contour in the mask destination
                    # In the function, text_mask is the image where we'll going to draw
                    # the contours, [contour] is the single occurrence of contours
                    # passed as a matrix, 0 is the element we're going to analyze (since
                    # countour is composed just by one element, the 0th) 255 is the color
                    # we're going to draw the contour and -1 means "fill the shape".
                    cv2.drawContours(text_mask, [contour], 0, 255, -1)

                    # then append the current contours point into bounding_contours
                    # alongside with his bounding coordinates x and y
                    bounding_contours.append((x, y, contour))

                    # PIPELINE SHOW
                    if display:
                        print("{} CONTOUR PASSED: H = {} / W = {} / x = {} / y = {}".format(i, h, w, x, y))
                        self.display_pipeline("Text Mask", text_mask)

                else:
                    # Instead, if precise_masking is False, we just compute the bounding
                    # box based on x,y,w,h computed before. Note that, for precision
                    # we give a 3 pixel room more in the height.
                    # Here, text_mask is the image where we'll going to draw the rectangle,
                    # (x,y) is the first vertex (uperr left) of the rectangle, (x+w) and
                    # (y+h) is the opposite vertex (bottom right) 255 is the is the color
                    # we're going to draw and -1 means "fill the rectangle".
                    cv2.rectangle(text_mask, (x, y), ((x + w), (y + h) + 3), 255, -1)

                    # then append the current bounding box points into bounding_contours
                    # alongside with his bounding coordinates x and y
                    bounding_contours.append((x, y, w, h))

                    # PIPELINE SHOW
                    if display:
                        print("{} CONTOUR PASSED: H = {} / W = {} / x = {} / y = {}".format(i, h, w, x, y))
                        self.display_pipeline("Text Mask", text_mask)
            else:
                # If here, that means the boundingRect coordinates did not pass the test;
                # we're then dealing with a FALSE POSITIVE that should be ignored. Display
                # information and skip the iteration cycle
                # PIPELINE SHOW
                if display: print("{} CONTOUR NOT PASSED: H = {} / W = {} / x = {} / y = {}".format(i, h, w, x, y))
                continue


        # We want to remove unwanted contours. That is to say, contours nested one in another.
        # So we sort the bounding boxes by x position and we proceed to check a constraint
        # on the top left and bottom right corners of the squares. We assume external square
        # always antecede internal squares as their "x" is smaller.
        bounding_contours = sorted(bounding_contours, key = lambda x: x[0])

        # Having the contours sorted by the x position (i.e.: columns) we now check
        # for every contour in bounding_contours if there are some nested contours
        # (i.e.: imagine a D where the first contour are the outer D points and the second
        # contour are the internal D points).
        # We start our enumeration at one because the first contour is represented by
        # the left-most character.
        for i, contour in enumerate(bounding_contours[1:], start=1):

            # Initialize empty x,y,w,h values needed later for nested contours check:
            # we do this both for the current contour and previous one
            [x_c, y_c, w_c, h_c] = [0,0,0,0]
            [x_p, y_p, w_p, h_p] = [0,0,0,0]

            # Check now if we're dealing with a precise masking or not: remember that,
            # a precise masking bounding_contours array is composed by 3 main elements,
            # (x,y,[contours_point]) while a not precise masking is composed by 4: (x,y,w,h)
            if precise_masking:

                # Extracting the boundingRect coordinate from the contous points, both
                # current and previous
                [x_c, y_c, w_c, h_c] = cv2.boundingRect(contour[2])
                [x_p, y_p, w_p, h_p] = cv2.boundingRect(bounding_contours[i-1][2])


            # Else, precise masking is set to false, and the len of contour is four:
            # x,y,w,h given by the non-precise masking bounding boxes.
            else:

                # Assigning the current bounding box coordinates to the existing one,
                # alongside with the previous one
                [x_c, y_c, w_c, h_c] = contour
                [x_p, y_p, w_p, h_p] = bounding_contours[i-1]

            # Extracted the points, we now check if the current contour box is inside
            # the previous one (i.e.: the D example). To do that, we check if the current
            # x is major than the previous x (upper left corner), and if the current x+w
            # is lesser than the previous x+w (upper right corner)
            if x_c > x_p and (x_c + w_c) < (x_p + w_p):
                # if so, remove the current element because it represent a nested
                # contour
                bounding_contours.remove(contour)

        # PIPELINE SHOW
        if display: self.display_pipeline("Final mask", text_mask)

        # We then return, after the character mask generation, both the masked text image
        # and the array containing the boxes that represent the characters.
        return (bounding_contours, text_mask)

    # END

    # Function that fiven an image name and its bounding contours, will create the .box
    # file associated with his associated syntax capable to be passed into the tesseract
    # train.
    def generate_boxfile(self, filename, bounding_contours):

        # If the OUTPUT_BOX file does not exists, create one to store the .box files
        if not os.path.isdir("OUTPUT_BOX"):
            os.mkdir("OUTPUT_BOX")


        # We use a try to catch an eventual error due to an inconsisten name of the plate
        # NOTE: for a correct .box file generation, the plate name should be like this:
        # ID-PLATECHARACTER.EXT -> 0-DZ234EW, where 0 is the numerical ID and the character
        # following are the plate characters contained in it. if the plate does not have 7
        try:

            # Since the filename is in format ID-PLATECHARACTER.EXTENSION we need to split
            # the string to retrieve ONLY the ID and the characters of the plate.
            # Retrieving the ID
            ID = filename.split("-")[0]

            # Retrieving the entire plate characters name
            plate = (filename.split("-")[1]).split(".")[0]

        except:

            # If an error occurred, just return: the name isn't in the correct format.
            return

        # Initializing an empty x,y,w,h coordinates tuple; this will be modified later
        # based on the lenght of the bounding_contours: 3 for the precise masking, that
        # will require a boundingRect function on the third element to retrieve [x,y,w,h]
        # 4 for the not-precise masking (and so already in the [x,y,w,h] format)
        [x,y,w,h] = [0,0,0,0]

        # If the bounding_contours len is > 7 (characters of the plate) proceed with the
        # .box extraction (so if the bounding_contours failed to retrieve all the 7 boxes
        # it will be not created)
        if len(bounding_contours) >= len(plate):

            # We know that the height of the image is a fixed value of 80
            height = 80

            # We then open and create a file with write permission inside the output path
            # of the OUTPUT_BOX folder: it's name will be identical to the PLATE ID to
            # mantain consistence between box files and plates.
            with open(os.path.join("OUTPUT_BOX", "{}.box".format(ID)), 'w') as file:

                # Enumerating the plate lenght (7 characters)
                for ind,char in enumerate(plate):

                    # We now check the bounding_contours lenght: if 3, this means the
                    # array was obtained using precise_masking, and so we need to use
                    # the function boundingRect to extract (x,y,w,h)
                    if len(bounding_contours[ind]) == 3:

                        # Extract the coordinates in format [x,y,w,h] using the contours
                        # points
                        [x,y,w,h] = cv2.boundingRect(bounding_contours[ind][2])

                    # Otherwise the non-precise masking was used and we can simply extract
                    # the coordinates tuple by accessing the bounding_contours at that very index
                    else:
                        [x,y,w,h] = bounding_contours[ind]

                    # We now get the coordinates: since tesseract uses the y axis inverted
                    # (not like openCV that uses (y,x), tesseract do use (x,y)) we need to
                    # invert coordinates in such manner. We use the height to invert them.
                    left = x
                    bottom = height - (y+h)
                    right = left + w
                    top = height - y

                    # We then write in the file the char followed by it's coordinate just fetched.
                    file.write("%s %d %d %d %d 0 \n" % (char,left,bottom,right,top))

    # End

    ##### 3) TEXT EXTRACTION METHODS
    # Function that, given an image both binarized and grayscale, the mask, his bounding contours,
    # it's name and the type of function to be applied, will both write result on disk and
    # return the image if specified. Note that, by default, both return and write are set
    # to false; this flags will be activated by the appropriate function.
    # Also, a stype (singletype) flag is passed, to let the grayscale_sametext_single which
    # function has to be applied to retrieve the single characters.
    def extract_text(self, bin_plate, gray_plate, plate, adaptive_coord, contours_coordinates, mask, name, ftype, stype, write=False, ret=False, display=False):

        # Now check whatever funciton the user needs with some ifs
        if ftype == FTYPE.ALL:
            # if all was choosen, return should be set to false
            # because it's useless to return all the images; just write them on disk.
            # Same reason apply to display: we need to write down, not display them.
            ret = False
            display= False

            # Calling every text-extraction function: since the flag is ALL, we need
            # to write on the disk EVERY possible kind of text-extraction method.
            # We proceed to call every method passing the relative image.
            self.binarized_smooth(bin_plate, name, mask, write, ret, display)
            self.binarized_text(bin_plate, name, mask, write, ret, display)
            self.grayscale_text(plate, name, mask, write, ret, display, adaptive_coord)
            self.grayscale_sametext(plate, name, mask, write, ret, display)

            # For the single character extraction, we need to know which kind of final
            # output we need: if a single character extracted from the binary plate
            # or extracted from the enhanced grayscale one.
            # To do that we just check which kind of SINGLE CHARACTER TYPE we'd like,
            # binary or grayscale. We then pass the appropriate image to the function.
            if stype == STYPE.BINARY:
                # If the extraction needed is from binary, pass the binary plate
                self.grayscale_sametext_single(bin_plate, contours_coordinates, name, mask, stype)
            else:
                # Else, the extraction if from the grayscale image.
                self.grayscale_sametext_single(plate, contours_coordinates, name, mask, stype)

        elif ftype == FTYPE.SMOOTH:
            # return smooth (returns none in case ret is false)
            return self.binarized_smooth(bin_plate, name, mask, write, ret, display)

        elif ftype == FTYPE.BINARY:
            # return binary text (returns none in case ret is false)
            return self.binarized_text(bin_plate, name, mask, write, ret, display)

        elif ftype == FTYPE.GRAYSCALE:
            # return grayscale text binarized (returns none in case ret is false)
            return self.grayscale_text(plate, name, mask, write, ret, display, adaptive_coord)

        elif ftype == FTYPE.EXACT:
            # return exact text of grayscale (returns none in case ret is false)
            return self.grayscale_sametext(plate, name, mask, write, ret, display)

        elif ftype == FTYPE.SINGLECHAR:
            # return exact text of grayscale subdivided by a single character
            # (doesn't return anything; it just write single characters on disk)
            # so both write and ret aren't passed because this function ONLY writes.
            # As above, we must check which kind of plate is needed from the extraction.
            if stype == STYPE.BINARY:
                # If the extraction needed is from binary, pass the binary plate
                self.grayscale_sametext_single(bin_plate, contours_coordinates, name, mask, stype)
            else:
                # Else, the extraction is made on the rgb plate
                self.grayscale_sametext_single(plate, contours_coordinates, name, mask, stype)

        else:
            # if ftype was wrong return -1 as error
            return -1

    # END

    def write_on_path(self, output_path, image, image_name):
        # Write output on disk: check if an output folder already exists
        if not os.path.isdir(output_path):
            # if not, create it and write the image on
            os.mkdir(output_path)
            cv2.imwrite(os.path.join(output_path, image_name), image)

        else:
            # If the path exists, just write the image inside
            cv2.imwrite(os.path.join(output_path, image_name), image)

    # END

    # Function that, given an RGB plate will enhance the color of the characters and
    # the background using the rgb planes for a accurate text extraction.
    def enhance_plate(self, plate):

        # Before going into the preprocessing, we need to scale our image ot a fixed
        # size of (80, 240) to be consistent in our steps.
        plate = self.optimal_resize(plate)

        # Splitting the RGB Planes into 3 images
        rgb_planes = cv2.split(plate)

        # Arrays containg the i-th step of the algorithm
        result_planes = []
        result_norm_planes = []

        # For each plane in the planes extracted
        for plane in rgb_planes:

            # Dilate the image to obtain a thicker version of the plate in that specific
            # plane
            dilated_img = cv2.dilate(plane, np.ones((5, 5), np.uint8))

            # Applly a median blur to obtain the background for that very plane
            bg_img = cv2.medianBlur(dilated_img, 31)

            # Do the absolute difference between the current plane and the calculated image
            # to retrieve a clean image
            diff_img = 255 - cv2.absdiff(plane, bg_img)

            # Normalize the result to enhance differences
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            # Append the processed plane to the arrasy
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        # After the operations are done, merge back the planes
        plate = cv2.merge(result_norm_planes)

        # And then return the grayscale version of the enchanced plate
        return cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Function that given the binarized optimal image and the mask, will produce a
    # clean image containing only the characters of the plate, smoothed and thickened
    # with some post-processing. It will also write the image with the name passed
    # on the specified directory.
    def binarized_smooth(self, binarized_plate, image_name,  text_mask, write, ret, display):

        # Simply applying the text mask on the filtered image will give us
        # an image resulting in only characters of the plate
        masked_binarized = cv2.bitwise_and(binarized_plate, binarized_plate, mask=text_mask)

        # PIPELINE SHOW
        if display: self.display_pipeline("Masked binarized", masked_binarized)

        # Inverting the image to convert white text to black text
        text_image = cv2.threshold(masked_binarized, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # PIPELINE SHOW
        if display: self.display_pipeline("Binarized not smoothed", text_image)

        # Blurring the image with a medium gaussian filter
        text_blur = cv2.GaussianBlur(text_image,(5,5),0)

        # Adding the weighted sum of the base image and the gaussian blur to retrieve
        # a smoother contour
        text_weighted = cv2.addWeighted(text_blur,1.5,text_image,-0.5,0)

        # Binarizing the resulting image to remove blurry areas
        text_binarized = cv2.threshold(text_weighted, 230, 255, cv2.THRESH_BINARY)[1]

        # thicken the image with a closing (dilation+erosion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_final = cv2.morphologyEx(text_binarized, cv2.MORPH_CLOSE, kernel, iterations=1)

        # PIPELINE SHOW
        if display: self.display_pipeline("Binarized Smoothed", text_final)

        # Write output on disk
        if write: self.write_on_path("OUTPUT_SMOOTH", text_final, image_name)

        # Return the text image if specified
        if ret: return text_final

    # END

    # Function that given the grayscale image and the mask, will produce a
    # clean image containing only the characters of the plate contained into the
    # binarized image. It will also write the image with the name passed
    # on the specified directory.
    def binarized_text(self, binarized_plate, image_name, text_mask, write, ret, display):

        # Since we got the binarized plate, we just apply the mask on it
        # to retrieve the plate characters
        masked_binarized = cv2.bitwise_and(binarized_plate, binarized_plate, mask=text_mask)

        # PIPELINE SHOW
        if display: self.display_pipeline("Mask binarized", masked_binarized)

        # Wr then invert the color to have the text in black instead of white
        text_final = cv2.threshold(masked_binarized, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # PIPELINE SHOW
        if display: self.display_pipeline("Output", text_final)

        # Write output on disk if specified
        if write: self.write_on_path("OUTPUT_BINARIZED", text_final, image_name)

        # Return the text image if specified
        if ret: return text_final

    # END

    # Function that given the grayscale image and the mask, will produce a
    # clean and enhanced image.
    def grayscale_text(self, plate, image_name, text_mask, write, ret, display, adaptive_coord):

        # Apply the enhancement on the plate
        grayscale_plate = self.enhance_plate(plate)

        # PIPELINE SHOW
        if display: self.display_pipeline("Grayscale enhanced output", grayscale_plate)

        # Given the enhanced plate, now we'll cut the blue band with the coordinates found
        # in the preprocessing step: note that, the coordinates could be adaptive (found by
        # the band contour method) or static (Defaultly assigned).
        # Cut from the left band coordinate found to the end
        grayscale_plate_crop = grayscale_plate[adaptive_coord[2]:adaptive_coord[3], adaptive_coord[0]:grayscale_plate.shape[1]]

        # Now cut from 0 to the rightmost starting band coordinate: remember that, the image has
        # been cropped from the left so we need to subtract to the rightmost coordinate the
        # pixels that now are missing (from 0 to adaptive_coord[0]).
        grayscale_plate_crop = grayscale_plate_crop[:, 0:(adaptive_coord[1]-adaptive_coord[0])]

        # PIPELINE SHOW
        if display: self.display_pipeline("Grayscale cropped", grayscale_plate_crop)

        # Reshape to original size
        grayscale_plate = self.optimal_resize(grayscale_plate_crop)

        # PIPELINE SHOW
        if display: self.display_pipeline("Grayscale cropped resized", grayscale_plate)

        # Write output on disk
        if write: self.write_on_path("OUTPUT_GRAYSCALE_BIN", grayscale_plate, image_name)

        # test
        cv2.imwrite("img.png", grayscale_plate)

        # Return the text image if specified
        if ret: return grayscale_plate

    # END

    # Function that given the grayscale image and the mask, will produce a
    # clean image containing only the characters of the plate returning
    # the exact same characters of the plate without any postprocessing.
    # It will also write the image with the name passed on the specified directory.
    def grayscale_sametext(self, plate, image_name, text_mask, write, ret, display):

        # Enhance plate
        grayscale_plate = self.enhance_plate(plate)

        # We first do an AND masking with the grayscale enhanced plate to extract the desired
        # zone containing the characters.
        masked_binarized_and = cv2.bitwise_and(grayscale_plate, grayscale_plate, mask=text_mask)

        # PIPELINE SHOW
        if display: self.display_pipeline("Masked binarized and", masked_binarized_and)

        # We then do the negation (not) of the and mask to retrieve the inverse mask
        masked_binarized_notand=cv2.bitwise_not(masked_binarized_and)

        # PIPELINE SHOW
        if display: self.display_pipeline("Masked binarized not and", masked_binarized_notand)

        # We then apply the text mask on the negation of the and mask retrieved before
        masked_binarized_andnotand = cv2.bitwise_and(masked_binarized_notand, masked_binarized_notand, mask=text_mask)

        # PIPELINE SHOW
        if display: self.display_pipeline("Masked binarized and not nad", masked_binarized_andnotand)

        # We now invert the image to retrieve the original and exact pixel values in their
        # right colors: we extracted ONLY the wanted masked areas into the grayscale image.
        masked_grayscale = ~masked_binarized_andnotand

        # Normalize final result for better color distinction
        cv2.normalize(masked_grayscale, masked_grayscale, 0, 255, cv2.NORM_MINMAX)

        # PIPELINE SHOW
        if display: self.display_pipeline("Output", masked_grayscale)

        # Write output on disk if specified
        if write: self.write_on_path("OUTPUT_GRAYSCALE_EXACT", masked_grayscale, image_name)

        # Return the text image if specified
        if ret: return masked_grayscale


    # END

    # Function that given the grayscale image, the mask and the spatial countour coord,
    # will produce a clean image containing only the characters of the plate, that will
    # further divide into seven images, each of them containing, in a sequential organization
    # (first spatial character will come first: i.e: having EX 4573A as plate, this function
    # will generate seven images, in the order: E(1), X(2), 4(3) etc..).
    # NOTE: this modality doesn't show return anything, it will ONLY write the single character
    # of a given plate on a folder.
    def grayscale_sametext_single(self, plate, contours_coordinates, image_name, text_mask, stype):

        # First thing we're going to do, is sort the contours_coordinate by the x coordinate
        # (the first element of both the precise masking and not precise masking result).
        # Doing this allow to organize the bounding box in terms of spatial relationship,
        # giving a 'sequence-like' order respecting the order of the plate string.
        # Remember that since we extracted the coordinates with OpenCV x means columns
        # and y means row. We gonna use a lambda function centered on the first element
        # to sort our array for the column position; lesser column value means precedent
        # position in the plate string.
        contours_coordinates = sorted(contours_coordinates, key=lambda x:x[0])

        # We then check which type of character we'd like to have in output: extracted
        # from a binary plate or from the enhanced grayscale image. We then initialize
        # an empty image at first
        text_image = None

        # And then check which type of function we need for the character extraction
        if stype == STYPE.BINARY:
            # If binary, we need to get the optimal binary image from the plate in input
            text_image = self.binarized_text(plate, image_name, text_mask, write=False, ret=True, display=False)

        else:
            # Otherwise, we need to extract the single character wanted from a grayscale
            # image.
            text_image = self.grayscale_sametext(plate, image_name, text_mask, write=False, ret=True, display=False)

        # If not existing, make a folder for the single character output
        if not os.path.isdir("OUTPUT_SINGLE"):
            os.mkdir("OUTPUT_SINGLE")

        # Since we're going to extract the single character from the grayscale plate
        # applying a bounding box on it, we need to nest all the operation on a for loop
        # iterating seven times (size of contours_coordinate) extracting the single
        # characters into single images.
        for i, contour in enumerate(contours_coordinates):

            # Before assuring in which case we're on, we need to define two empty structure
            # that will save the current character image and the coordinates in which the
            # bounding box around this caracter is stored. We do this to avoid reduntant
            # operation in the code.
            cur_character = None
            [x,y,w,h] = [0,0,0,0]

            # Now, first thing we need to check, is the lenght of the contour passed:
            # we have two cases. 1) is when precise_masking was set to true, so the
            # contour is composed by (x,y, [contour_points]). In this case, the lenght
            # is equal to three. 2) is when precise_masking was set to true, so the
            # contour is composed by (x,y,w,h) because a rectangle was draw.
            # Different cases requires then different approached.
            # Case when the precise_masking is true (we got only character contours)
            if len(contour) == 3:

                # using the contour itself, we're going to generate a bounding rectangle
                # around our contour to define the area of the output image, since boundingRect
                # calculates the up-right bounding rectangle of a point set. We pass into
                # the function the third argument of contour, the contour points array.
                [x, y, w, h] = cv2.boundingRect(contour[2])

            # if the lenght of the current contour isn't equal to three, that means we
            # got 4 elements (and we know that for sure since the extract_contours function
            # returns an array composed by [x,y,w,h] elements instead of [x,y,[contours_point])
            # and we just need to write the single character as an image extracting the box.
            else:

                # Extract the [x,y,w,h] elements from the current contour element, saved
                # previously with boundingRect into the contour extraction function
                [x, y, w, h] = contour

            # We then need to extract the current character from the grayscale image:
            # to achieve this fast, we use the numpy slicing on the image: note that,
            # only OpenCV treats x as y and y as x, so in a numpy array x = row and
            # y = col. With that in mind, we invert the coordinates using y as row
            # and x as col since we got x,y,w,h with the OpenCV function boundingRect.
            # We use y and x as starting index, and then crop our ROI using the widht
            # and height to generate the bounding box of the text_image, already processed
            # and ready to be used for character extraction purposes.
            # Note that, Using the slicing we'll select ONLY the image part relative to
            # the current analyzed character.
            cur_character = text_image[y: y+h, x: x+w]

            # Make padding border on the image using copyMakeBorder and passing the
            # desired padding size in every direction
            bordersize = 10
            cur_character = cv2.copyMakeBorder(cur_character, top=bordersize, bottom=bordersize,
                                               left=bordersize,  right=bordersize,
                                               borderType=cv2.BORDER_CONSTANT, value=255)

            # Having the current character box, we now need to write it into the
            # folder defined by the current image name passed in input: for that,
            # we'll simply use the function write_on_path to write the current
            # character as a single image.
            self.write_on_path(os.path.join("OUTPUT_SINGLE", image_name.split('.')[0]), cur_character, "{}.png".format(i))


    # END

    #### END TEXT EXTRACTION METHODS

    #### DISPLAY FUNCTIONS
    # Function that, given a path, will write the corresponding image/s found in the path
    # (if the path is an image, it will write only the passed image) into an output folder.
    # NOTE: if the input path is a folder, it must contains only images and folders.
    # Defaultly, This function will apply the binarization function as default one,
    # with precise_masking=false (bounding boxes will be returned instead of precise contours)
    def apply_extraction_onpath(self, input_path=None, desired_ext='png', precise_masking=True, adaptive_bands=True, ftype=FTYPE.BINARY, stype=STYPE.BINARY, ret=False, write=True):

        # Check if path is none then exit
        if input_path is None:
            print("Path must be a folder containing images or a single plate image.")
            exit(1)

        # First thing, we check if the passed input path is a directory:
        if os.path.isdir(input_path):

            print("Going to extract {} images from: {}".format(len(os.listdir(input_path)), input_path))

            # If so, we are going to extract image by image the files in that directory.
            for file in tqdm(sorted(os.listdir(input_path))):

                # We then check if the fhe file extracted is a folder
                if os.path.isdir(file):
                    # if so, continue with the next for iteration
                    continue

                # If not, we then extract the text from the file and write it into
                # it's relative folder
                # We then open the image
                plate = cv2.imread(os.path.join(input_path, file))

                # extract the grayscale normalized and the binarized optimal plate
                gray_plate, bin_plate, adaptive_coord = self.adaptive_preprocessing(plate)

                # we then extract the contours mask
                contours_coordinate, contours_mask = self.extract_contours(bin_plate, precise_masking=precise_masking, band_coordinates=adaptive_coord)

                # Generating a .BOXFILE for an optional tesseract (OCR) training
                self.generate_boxfile(file, contours_coordinate)

                # Split the filename to extract name and extension and subsitute the
                # extracted extensione with the desired one in the latter function
                filename, _ = file.split('.')

                # and extract the text from the image: note that, this function will write
                # by default results on disk.
                result = self.extract_text(bin_plate, gray_plate, plate, adaptive_coord, contours_coordinate, contours_mask, "{}.{}".format(filename, desired_ext), ftype, stype, True, False, False)
        
        else:

            print("Going to extract characters from: {}".format(input_path))

            # Else, the path is not a folder but just an image. Then, proceed
            # to apply preprocessing and text extraction.
            # Reading the file
            plate = cv2.imread(input_path)

            # extract the grayscale normalized and the binarized optimal plate
            gray_plate, bin_plate, adaptive_coord = self.adaptive_preprocessing(plate)

            # we then extract the contours mask
            contours_coordinate, contours_mask = self.extract_contours(bin_plate, precise_masking=precise_masking, band_coordinates=adaptive_coord)

            # Generating a .BOXFILE for an optional tesseract (OCR) training
            self.generate_boxfile(input_path, contours_coordinate)

            # Split the filename to extract name and extension and subsitute the
            # extracted extensione with the desired one in the latter function
            filename, _ = input_path.split('.')

            # and extract the text from the image: note that, this function will write
            # by default results on disk.
            result = self.extract_text(bin_plate, gray_plate, plate, adaptive_coord, contours_coordinate, contours_mask, "{}.{}".format(filename, desired_ext), ftype, stype, write, ret, False)
    
            # If ret was set to true, return the processed plate to the user 
            if ret: return result
            
        print("All done!")

    # END

    # Function aimed to show a preprocessing cycle on a given image
    # Note that precise_masking is set to false; we'll then get the bounding box mask.
    # We also use SMOOTH as standard function and return set to true to retrieve
    # the image processed with the extract_text function.
    # NOTE: this function works only for FTYPE GRAYSCALE, FTYPE EXACT, FTYPE SMOOTH and FTYPE BINARY.
    def display_result(self, input_path=None, precise_masking=True, adaptive_bands=True, ftype=FTYPE.BINARY, display=True):

        # Exit if no path specified or is a folder
        if input_path is None or os.path.isdir(input_path) is True:
            print("Path missing OR path inserted is a folder.\nPlease usa a single image when using function 'display_result'.")
            exit(1)

        if ftype==FTYPE.SINGLECHAR:
            print("Single characters can not be displayed. Please, select another extraction method to display.")
            exit(1)

        # Reading the plate
        plate = cv2.imread(input_path)

        # Apply preprocessing on it to binarize information
        gray_plate, bin_plate, adaptive_coord = self.adaptive_preprocessing(plate, adaptive_bands, display)

        # We then extract the contours passing the specified precise_masking property
        # and taking only the second of the returned arguments (the image)
        contours_coordinate, contours_mask = self.extract_contours(bin_plate, precise_masking=precise_masking, band_coordinates=adaptive_coord, display=True)

        # we then process the final image containing only the clean text.
        text = self.extract_text(bin_plate, gray_plate, plate, adaptive_coord, contours_coordinate, contours_mask, "image.png", ftype, None, False, True, display)

    # End

# Endclas
