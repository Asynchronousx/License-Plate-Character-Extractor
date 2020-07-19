import sys
from lpce import PlateExtractor, FTYPE, STYPE

# Generating our istance
extractor = PlateExtractor()

# Fetching the user path: can be an image or an entire folder containing images
path = sys.argv[1]

### NOTE: use ONE method at time, commeting others not used. ###
### VARIOUS EXAMPLE CASES: more info at https://github.com/Asynchronousx/License-Plate-Character-Extractor: 

# 0) Calling the function display_result to display the pipeline to the user of a given image.
# We pass precise_masking = True that means the contours of the character will be taken instead
# of the bounding box they rely in. We pass the FTYPE BINARY to get the binary image as result
# and display = True to display each step of the process.
extractor.display_result(path, precise_masking=True, ftype=FTYPE.GRAYSCALE, display=True)

# 1) Extracting the characters from a single image/entire folder with a single method:
# Note that, the default method is the binarization text.
# With apply_extraction_onpath we apply the extraction of the characters on the desired path
# (single image or an entire folder), using precise masking (contour of the characters instead
# of the bounding boxes. Note that, precise_masking is set default to false) and the generated
# extension as png (default: tif)
extractor.apply_extraction_onpath(input_path=path, desired_ext='png', precise_masking=True)

# 2) Extracting the characters from a single image/entire folder with a different method
# I.E: grayscale exact characters, using default ext (tif) and default precise_masking.
extractor.apply_extraction_onpath(input_path=path, ftype=FTYPE.EXACT)

# 3) Extracting the characters from a single image/entire folder with the SINGLE CHARACTER
# method, in which we'll extract into several single images each of the characters present in the
# plate. Default, the type of the output will be BINARY but we can choose between that and EXACT,
# that will return the same masked character of the original plate.
extractor.apply_extraction_onpath(input_path=path, ftype=FTYPE.SINGLECHAR, stype=STYPE.EXACT)

# 3) Extracting the characters from a single image/entire folder with the ALL method,
# in which all the postprocessing methods for characters extraction will be called on the
# given images, and specifying the SINGLE CHARACTER EXTRACTION TYPE (default: binary)
extractor.apply_extraction_onpath(input_path=path, precise_masking=False, ftype=FTYPE.ALL, stype=STYPE.EXACT)
