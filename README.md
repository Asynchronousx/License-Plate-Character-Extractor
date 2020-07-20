# License Plate Character Extractor
<b>LPCE</b>: A simple yet useful tool built to extract only the alphanumerical characters from a license plate image.<br><br>
<img src="https://i.ibb.co/fFvsPN6/unnamed.jpg" title="Optional title" alt="Original Plate"> <img src="https://i.ibb.co/hXm4hc4/resized.png" alt="Binarized Plate"><br><br>
The reason behind this script was given by the necessity of extracting only the characters from a license plate to fine-tune an OCR neural network model like <b>Tesseract</b>, both for train the netowrk on how to recognize and correctly identify characters from a license plate and for applying a solid preprocessing before perform the OCR.<br><br>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Features
This script comes with <b>five</b> different methods of extraction, based on different needs:<br>

#### Extraction methods             
+ <b>Plate Exact Grayscale</b>: Given a license plate, will produce an image containing only the exact character from the original plate. Note: This method works really good with good resolution LP since it will build a mask containing the cropped characters from the original LP.<br>
<img src="https://i.ibb.co/FBVBfbW/unnamed.png" alt="Exact plate"><br>
+ <b>Plate Binarization</b>: Given a license plate will produce the most accurate binarization.<br>
<img src="https://i.ibb.co/1G2LSpH/unnamed-binary.png" alt="Binarized plate"><br>
+ <b>Plate Enhance Grayscale</b>: Given a license plate will produce a cropped, enhanced original plate containing mostly only the area in which the characters are contained.<br>
<img src="https://i.ibb.co/y8ykrVf/unnamed.png" alt="Grayscale plate"><br>
+ <b>Plate Binarization Smooth</b>: Given a license plate, will produce a smoothed and thickened version of the binarization plate<br>
<img src="https://i.ibb.co/cb0Jkcc/unnamed.png" alt="Smooth plate"><br>
+ <b>Single Character Extraction</b>: It does what you expect; extracts one by one the characters contained into the image, saving each <br>
of them into a separate image. We got two kind of single character extraction:
    + <b>Binary</b>: Will give a similar output as the Plate Binarization, but dividing each character into a single image.<br>
    ![](https://i.ibb.co/j4FTnM5/0-binary.png=250x250),![](https://i.ibb.co/4Njzk9Z/1-binary.png=250x250),![](https://i.ibb.co/xSvfzNP/2-binary.png=250x250),![](https://i.ibb.co/t2ZgKNT/3-binary.png=250x250),![](https://i.ibb.co/w0rdR5s/4-binary.png=250x250),![](https://i.ibb.co/rGkGhyY/5-binary.png=250x250),![](https://i.ibb.co/QMTjSgM/6-binary.png=250x250)
    + <b>Exact</b>: Will give a similar output as the Plate Exact, but dividing each character into a single image.<br>
    ![](https://i.ibb.co/ggTG5BW/image.png=250x250),![](https://i.ibb.co/sR8Kd90/1.png=250x250),![](https://i.ibb.co/hXD8Qct/2.png=250x250),![](https://i.ibb.co/NnGDJbj/3.png=250x250),![](https://i.ibb.co/Y8jy6CD/4.png=250x250),![](https://i.ibb.co/0XZsBzf/5.png=250x250),![](https://i.ibb.co/xDSKzbf/6.png=250x250)<br>
+ <b>ALL</b>: Methods that uses <b>EACH OF THE PREVIOUS</b> methods, calling them one by one and storing the result of each one into the appropriate folder.<br>

#### Functionalities
This script was made with <b>efficience</b> in mind. Indeed, each of the methods proposed are insanely fast and performant, relying mostly on the <b>Numpy</b> and <b>OpenCV</b> library: this because, the preoprocessing step before the OCR should be as fast as possible.<br>

+ <b>Display Result</b>: Function that takes in input an image and display every single step of our pre and post processing, useful to understand what the pipeline does (for curiosity or debug purposes). NOTE: <b>can't</b> be used with SINGLECHAR.
+ <b>Extraction on path</b>: Given a path (a single image or an entire folder), this function will take care of apply the extraction methods on the given input/s. <b>NOTE</b> If a folder has been passed, be sure that the folder itself contains ONLY IMAGES. Others file will make the script not working.
+ <b>.BOX File creation for Tesseract</b>: Since this script was created having in mind the <b>training of tesseract</b> as main purpose, it will take care of generate <b>.BOX</b> file containing the coordinates of each character into the given plate, that will be needed later into the tesseract training to identify the characters. To take advantage of this functionality, you need to name your file as follow: ID-PLATECHARACTERS.EXT<br>
-> Example: Given the plate used as example, the name should be: <b>0-FI764WL.ext</b> where ID are progressive into the main folder (0,1,2,3,4...N), FI764WL are the characters contained into the plate and ext can be <b>any extension</b> (jpg,png,tif etc). It will then produce a .BOX file containing coordinates useful for tesseract in training phase. <b>Be careful to the syntax</b> if you want to generate .BOX files!<br>

## Usage

First of all, be sure to download/clone the repo and extract the <b>lcpe.py</b> in a folder of your choice (default: current folder) and then follow up the usage section. If you want to extract the lcpe.py file in another directory, be sure to specify that into your script. Additionally, the file <b>example.py</b> contains some of usage examples ready to be executed.<br>
Note that, the script <b>works on image containing only the plate</b>. Images with an entire car or with partially visible plates will not work well.<br> 

### Basic
This script was also made with an extreme <b>ease of use</b> in mind.<br>
All the functions are organized into a class, which you'll need to import into your python script to perform the extraction.<br>
There are two main methods which can use, and both of them can be used out-of-the-box without any tweaking to it: <b>display_result</b> to show pipeline outputs at each step and <b>apply_extraction_on_path</b> that given a folder or a single image, will take care of creating for you all the output folders in which results will be contained; if specified, can also return the processed plate.<br>
By default, the extraction method used without any tweaking on the flags are both Binarization for the entire plate and the single characters extraction. Let's look for a practical example: <br>
<b>Extraction on a given path:</b>
```python
# Importing class module
from lpce import PlateExtractor
# Generating our istance
extractor = PlateExtractor()
# Apply extraction on a given path (image or an entire folder containing ONLY images)
extractor.apply_extraction_onpath(input_path=path)
```
<b>Extraction on a single image specifying the return flag (plate must be returned)</b>
```python
# Importing class module
from lpce import PlateExtractor
# Generating our istance
extractor = PlateExtractor()
# Apply extraction on a given path (image)
extracted_plate = extractor.apply_extraction_onpath(input_path=path, ret=True)
```
<b>Display Pipeline results:</b>
```python
# Importing class module
from lpce import PlateExtractor
# Generating our istance
extractor = PlateExtractor()
# Display each step of the pipeline on a SINGLE image
extractor.display_result(input_path=path)
```

### Advanced
Instead, if you'd like to tweak the extraction or the display method, there are some flags you should know about: let's look for the entire function call in the class, both for extraction and display:<br>
```python
def apply_extraction_onpath(self, input_path=None, desired_ext='png', precise_masking=True, adaptive_bands=True, ftype=FTYPE.BINARY, stype=STYPE.BINARY, write=True, ret=False)
def display_result(self, input_path=None, precise_masking=True, adaptive_bands=True, ftype=FTYPE.BINARY, stype=STYPE.BINARY, write=False, ret=True, display=True)
```

Let's now breakdown the parameter list and explain carefully what each of them does:
#### List of parameters: apply_extraction_on_path
+ <b>input_path</b>: The path resolving to an image or an entire folder (containing only images)
+ <b>desired_ext</b>: Desired extension for the output file. <b>default='png'</b>.
+ <b>precise_masking</b>: Precise masking indicates which type of output will be made by the script. Setting to ON, will produce an image in which the precision of the extraction is built around the character contour. Instead, setting to OFF, will produce an output in which the precision of the result character will be the bounding box containing the character itself. <b>default=True</b>.
+ <b>adaptive_bands</b>: Adaptive bands will automatically fetch the coordinates of the blue band (if present) into the image that will be used in the script to crop automatically the image considering only the area of the plate in which the character are contained. <b>default: True</b>
+ <b>ftype</b>: Type of extraction method used to process the plate into an image containing only the characters. <b>default=FTYPE.BINARY</b>
+ <b>stype</b>: Type of extraction method used to extract the single characters from the plate into single images. <b>default=STYPE.BINARY</b><br>
#### List of parameters: display_result
+ <b>input_path</b>: Same as above
+ <b>precise_masking</b>: Same as above
+ <b>adaptive_bands</b>: Same as above
+ <b>ftype</b>: Same as above
+ <b>display</b> Flag that indicates if displaying or not the additional pre and post processing steps for a <b>full view</b> of the pipeline process. <b>default=True</b><br>

Since we saw different methods, there are different flags for each method contained into a comfy enum class. Let's see the structure, both for <b>FTYPE and STYPE</b>:<br>
```python
class FTYPE(Enum):
    ALL = 0
    EXACT = 1
    SMOOTH = 2
    BINARY = 3
    GRAYSCALE = 4
    SINGLECHAR = 5

class STYPE(Enum):
    BINARY = 1
    EXACT = 2
```
Each of them corresponds to the methods described earlier into the README and will be needed if planning to use the script in a different way than the default behavior. Let's then see an example of an advanced use, with custom flags:<br>
<b>Apply extraction on path</b>:
```python
# Importing class module, FTYPE and STYPE enumS
from lpce import PlateExtractor, FTYPE, STYPE
# Generating our istance
extractor = PlateExtractor()
# Apply extraction on a given path, with precise_masking set to false and grayscale extraction instead of binary
extractor.apply_extraction_onpath(input_path=path, precise_masking=False, ftype=FTYPE.GRAYSCALE)
```
OR
```python
# Apply extraction on a given path, with adaptive_bands set to false, using ALL extraction methods
# and specifying EXACT method for the single character extraction
extractor.apply_extraction_onpath(input_path=path, precise_masking=False, ftype=FTYPE.GRAYSCALE, stype=STYPE.EXACT)
```
OR
```python
# Apply extraction on a given path using the single character method extraction
extractor.apply_extraction_onpath(input_path=path, ftype=FTYPE.SINGLECHAR)
```
<b>Display Result</b>: Basically the approach is the same as above. We'll see just an example to show off:<br>
```python
# Display results of the pipeline of a given image using the Smoothing extraction method, using display=True to show off EVERY step of the pipeline
extractor.display_result(path, ftype=FTYPE.SMOOTH, display=True)
```
<b>NOTE:</b> Obviously, combinations can be made <b>AS YOU PREFER</b>. No limit on that. <br>

#### Dependencies
You'll need this modules to run this script:
+ <b>Opencv (cv2) >3</b>
+ <b>Numpy</b>
+ <b>Imutils</b>
+ <b>TQDM</b>

## Known issues
The script was made to work with Italian plates, but works with almost EVERY kind of CAR plate, as long they presents two common aspects:<br>
+ If bands are present, they must be <b>BLUE</b>
+ Plate background should be </b>WHITE</b> (i.e: not working with different background, like yellow). 


