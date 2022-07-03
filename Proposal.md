**Bahria University**

**Karachi Campus**

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.001.png)

**COURSE: CSC-411       Artificial Intelligence      Term: 6th Department: BCE**

**Proposal Report** 

**Submitted By:** 

Hassan Abbas (02-132192-013)

**Submitted To: Engr. Nabiha Faisal** 

Contents 

[Abstract ............................................................................................................................................... 3 ](#_page2_x69.00_y220.92)[Introduction ........................................................................................................................................ 3 ](#_page2_x69.00_y414.92)[The Pipeline ......................................................................................................................................... 3 ](#_page2_x313.00_y450.92)[Methodology: ...................................................................................................................................... 4 ](#_page3_x69.00_y280.92)[Implementation presentation using Flowchart: ................................................................................. 4 ](#_page3_x313.00_y145.92)[Results: ................................................................................................................................................ 4 ](#_page3_x313.00_y478.92)[Conclusion: .......................................................................................................................................... 5 ](#_page4_x313.00_y504.92)[References .......................................................................................................................................... 5 ](#_page4_x313.00_y680.92)[CODE ................................................................................................................................................... 7 ](#_page6_x69.00_y252.92)

Figure 

[Figure 1 A typical Sudoku puzzle ............................................................................................................. 3 ](#_page2_x313.00_y429.92)[Figure 2 Selected Image read using OpenCV .......................................................................................... 4 ](#_page3_x313.00_y706.92)[Figure 3 Coverting to GrayScale for better processing of image ............................................................ 5 ](#_page4_x69.00_y278.92)[Figure 4 Detecting Corners of the Puzzle ................................................................................................ 5 ](#_page4_x69.00_y511.92)[Figure 5 Extracting the Puzzle from the image ....................................................................................... 5 ](#_page4_x69.00_y736.92)[Figure 6 Removing Number From the Boxes .......................................................................................... 5 ](#_page4_x313.00_y273.92)[Figure 7 Sudoku puzzle as a 9x9 matrix .................................................................................................. 5 ](#_page4_x313.00_y415.92)[Figure 8 Prompting the message when solution not found ................................................................... 5 ](#_page4_x313.00_y472.92)

**Sudoku Solver using Deep Learning** 

**Hassan Abbas** 

Bahria University Karachi Campus 

13 National Stadium Road, Karachi-75260 – Pakistan 

**Abstract**

Sudoku  is  a  well-known  puzzle- solving game, with simple rules of completion yet may require complex reasoning. In this project, we aim to develop  a  system  that  takes  any picture of a Sudoku puzzle, extracts its contents using image processing techniques,  solves  the  puzzle,  and then displays the results. 

**Introduction** 

**Sudoku** 

Sudoku  is  a  logic-based  puzzle-solving  game  introduced  in  Japan. The  name  Sudoku  means  ‘Number digits’  in  classic  Sudoku,  the objective is to fill a 9 × 9 grid with digits so that each column, each row, and each of the nine 3 × 3 sub-grids that  compose  the  grid  (also  called "boxes",  "blocks",  or  "regions") contain all of the digits from 1 to 9. 

**Steps of the Project** 

**Part 1:** Digit Classification Model **Part 2:** Reading and Detecting the Sudoku From an Image 

**Part 3:** Solving the Puzzle 

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.002.png)

*Figure 1 A typical Sudoku puzzle*

**The Pipeline** 

We will divide the project into 3 parts which are as follows: 

**Part 1: Digit Classification Model** 

We  plan  to  first  build  and  train  a neural  network  on  the  Char74k images dataset for digits. This model will help to classify the digits from the images. 

**Part 2: Reading and Detecting the Sudoku From an Image** 

This section contains, identifying the puzzle from an image with the help of OpenCV, classifying the digits in 

the  detected  Sudoku  puzzle  using Part-1, and finally getting the values of the cells from Sudoku and stored in an array. 

**Part3: Solving the Puzzle** 

We are going to store the array that we  got  in  Part-2  in  the  form  of  a matrix  and  finally  run  a  recursion loop to solve the puzzle. 

**Methodology:** 

**Tools:** 

- Python 3.9.7 
- conda 4.13.0 
- anaconda 2021.11 

**Packages:** 

- cv2 4.5.5 
- opencv-python 4.5.5.64 
- numpy 1.20.3 
- pandas 1.3.4 
- seaborn 0.11.2 
- matplotlib 3.4.3 
- imutils 0.5.4 
- pytesseract 0.3.9 
- keras\_ocr 0.9.1 
- skimage 0.18.3 

Platform: 

- IPython          : 7.29.0 
- ipykernel        : 6.4.1 
- ipywidgets       : 7.6.5 
- jupyter\_client : 6.1.12 
- jupyter\_core : 4.8.1 
- jupyter\_server   : 1.4.1 
- jupyterlab       : 3.2.1 
- nbclient         : 0.5.3 
- nbconvert        : 6.1.0 
- nbformat : 5.1.3 
- notebook : 6.4.5 
- qtconsole        : 5.1.1 
- traitlets        : 5.1.0 

**Implementation presentation using Flowchart:** 

**Flowchart:** 

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.003.jpeg)

**Results:** 

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.004.png)

*Figure 2 Selected Image read using OpenCV*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.005.png)

*Figure 3 Coverting to GrayScale for better processing of image*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.006.png)

*Figure 4 Detecting Corners of the Puzzle*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.007.png)

*Figure 5 Extracting the Puzzle from the image*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.008.png)

*Figure 6 Removing Number From the Boxes*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.009.png)

*Figure 7 Sudoku puzzle as a 9x9 matrix*

![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.010.png)

*Figure 8 Prompting the message when solution not found*

**Conclusion:** 

In this project, we performed image processing to recognize and extract the puzzle from the image then we performed  OCR  (Object  Character Recognition)  using  pytesseract. Lastly,  we  use  the  Constraint Satisfaction  Problem  to  solve  the Sudoku puzzle. 

**References** 

1. Akshay  Gupta,  “Solving Sudoku  From  Image  Using Deep Learning – With Python Code”,  [Online].  Available: 

[https://www.analyticsvidhya.c om/blog/2021/05/solving- sudoku-from-image-using- deep-learning-with-python- code/ ](https://www.analyticsvidhya.com/blog/2021/05/solving-sudoku-from-image-using-deep-learning-with-python-code/)

2. “Sudoku”,  [Online]. Available: [https://en.wikipedia.org/wiki/ Sudoku ](https://en.wikipedia.org/wiki/Sudoku)
2. Jonathan  Pritchard,  “Sudoku Image  Solver”,  [Online]. Available: [https://github.com/jpritcha3- 14/sudoku-image-solver ](https://github.com/jpritcha3-14/sudoku-image-solver)
2. [Orhan  G.  Yalçın,](https://blog.orhangaziyalcin.com/?source=post_page-----54c35b77a38d--------------------------------)  “Image Classification  in  10  Minutes with  MNIST  Dataset” [Online] Available: [https://towardsdatascience.co m/image-classification-in-10- minutes-with-mnist-dataset- 54c35b77a38d ](https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d)
2. Nithyashree V, “Step-by-Step guide for Image Classification on Custom Datasets” [Online] Available: [https://www.analyticsvidhya.c om/blog/2021/07/step-by- step-guide-for-image- classification-on-custom- datasets/ ](https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/)
2. SUDARSHAN S MAGAJI, “Sudoku  Box  Detection”  [Online] Available: 

[https://www.kaggle.com/datas ets/macfooty/sudoku-box- detection ](https://www.kaggle.com/datasets/macfooty/sudoku-box-detection)

7. “Sudoku  Solver  using OpenCV & Python” [Online] Available: [https://data- flair.training/blogs/opencv- sudoku-solver/ ](https://data-flair.training/blogs/opencv-sudoku-solver/)
7. “How  to  extract  text  from image using Python” [Online] Available: [https://www.etutorialspoint.co m/index.php/381-text- extraction-from-image-using- opencv-and-ocr-python ](https://www.etutorialspoint.com/index.php/381-text-extraction-from-image-using-opencv-and-ocr-python)
7. “Python  OpenCV  cv2  Find Contours  in  Image”  [Online] Available: [https://pythonexamples.org/py thon-opencv-cv2-find- contours-in-image/ ](https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/)
7. “Binarization  of  image  in opencv” [Online] Available: [https://stackoverflow.com/que stions/34288776/binarization- of-image-in-opencv ](https://stackoverflow.com/questions/34288776/binarization-of-image-in-opencv)
7. “Python  Tesseract  4.0  OCR: Recognize  only  Numbers  / Digits  and  exclude  all  other Characters”  [Online] Available: [https://return2.net/python- tesseract-4-0-get-numbers- only/ ](https://return2.net/python-tesseract-4-0-get-numbers-only/)

12. [“pytesseract using tesseract ](https://stackoverflow.com/questions/46574142/pytesseract-using-tesseract-4-0-numbers-only-not-working)![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.011.png)

[4.0 numbers only not working”](https://stackoverflow.com/questions/46574142/pytesseract-using-tesseract-4-0-numbers-only-not-working) [Online] Available: 

[https://stackoverflow.com/que stions/46574142/pytesseract- using-tesseract-4-0-numbers- only-not-working ](https://stackoverflow.com/questions/46574142/pytesseract-using-tesseract-4-0-numbers-only-not-working)

**CODE** 

#!/usr/bin/env python 

- coding: utf-8 
- In[1]: 

#============ importing the libraries =========== import cv2 

import random, os 

import numpy as np  

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from imutils import contours 

import pytesseract 

import math 

import keras\_ocr  

from skimage import data, restoration 

from skimage.transform import resize 

- In[63]: 

#============== loading the image and reading it into binary form using cv2.imread() ================ folder = 'D:/python projects/sudoku- images/aug/\_213\_9204346.jpeg' 

- a=random.choice(os.listdir(folder)) 
- print(a) 

sudoku\_a = cv2.imread(folder)#+'/'+a) 

plt.figure() 

plt.imshow(sudoku\_a) 

plt.show() 

- In[64]: 

#================ Preprocessing image to be read sudoku\_a = cv2.resize(sudoku\_a, (450,450)) 

- function to greyscale, blur and change the receptive threshold of image 

def preprocess(image): 

`    `gray = cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY)  

`    `blur = cv2.GaussianBlur(gray, (3,3),6)  

`    `threshold\_img = cv2.adaptiveThreshold(blur,255,1,1,11,2) 

`    `return threshold\_img 

threshold = preprocess(sudoku\_a) 

#let's look at what we have got 

plt.figure() 

plt.imshow(threshold) 

plt.show() 

- In[65]: 

#============= Determining the outline of the sudoku puzzle 

contour\_1 = sudoku\_a.copy() 

contour\_2 = sudoku\_a.copy() 

contour, hierarchy = cv2.findContours(threshold,cv2.RETR\_EXTERNAL,cv2.CHAIN\_AP PROX\_SIMPLE) 

cv2.drawContours(contour\_1, contour,-1,(0,255,0),3) #let's see what we got 

plt.figure() 

plt.imshow(contour\_1) 

plt.show() 

- In[66]: 
- =================== Functions for extracting the puzzle,  
- reframing and splitting the cells 

def main\_outline(contour): 

`    `biggest = np.array([]) 

`    `max\_area = 0 

`    `for i in contour: 

`        `area = cv2.contourArea(i) 

`        `if area >50: 

`            `peri = cv2.arcLength(i, True) 

`            `approx = cv2.approxPolyDP(i , 0.02\* peri, True) 

`            `if area > max\_area and len(approx) ==4:                 biggest = approx 

`                `max\_area = area 

`    `return biggest ,max\_area 

def reframe(points): 

`    `points = points.reshape((4, 2)) 

`    `points\_new = np.zeros((4,1,2),dtype = np.int32)     add = points.sum(1) 

`    `points\_new[0] = points[np.argmin(add)] 

`    `points\_new[3] = points[np.argmax(add)] 

`    `diff = np.diff(points, axis =1) 

`    `points\_new[1] = points[np.argmin(diff)] 

`    `points\_new[2] = points[np.argmax(diff)] 

`    `return points\_new 

def splitcells(img): 

`    `rows = np.vsplit(img,9) 

`    `boxes = [] 

`    `for r in rows: 

`        `cols = np.hsplit(r,9) 

`        `for box in cols: 

`            `box = cv2.resize(box, (450, 450))/255.0             cv2.imshow("Splitted block", box) 

`            `cv2.waitKey(50) 

`            `boxes.append(box) 

`    `return boxes 

black\_img = np.zeros((450,450,3), np.uint8) biggest, maxArea = main\_outline(contour) if biggest.size != 0: 

`    `biggest = reframe(biggest) 

cv2.drawContours(contour\_2,biggest,-1, (0,255,0),10) pts1 = np.float32(biggest) 

pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]]) matrix = cv2.getPerspectiveTransform(pts1,pts2) imagewrap = cv2.warpPerspective(sudoku\_a,matrix,(450,450)) imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR\_BGR2GRAY) plt.figure() 

plt.imshow(imagewrap) 

plt.show() 

- In[67]: 

#============ Removing the Numbers from inside the Boxes thresh\_2 = imagewrap.copy() 

thresh\_2 = cv2.adaptiveThreshold(imagewrap,255,cv2.ADAPTIVE\_THRESH\_G AUSSIAN\_C, cv2.THRESH\_BINARY\_INV,57,5) 

cnts = cv2.findContours(thresh\_2, cv2.RETR\_TREE, cv2.CHAIN\_APPROX\_SIMPLE) 

cnts = cnts[0] if len(cnts) == 2 else cnts[1] 

for c in cnts: 

`    `area = cv2.contourArea(c) 

`    `if area < 1000: 

`        `cv2.drawContours(thresh\_2, [c], -1, (0,0,0), -1) plt.figure() 

plt.imshow(thresh\_2) 

print(thresh\_2.shape) 

plt.show()     

- In[68]: 
- Fix horizontal and vertical lines 

vertical\_kernal = cv2.getStructuringElement(cv2.MORPH\_RECT, (1,5)) 

- kernel = np.ones((5,5),np.uint8) 
- thresh\_2 = cv2.erode(imagewrap,kernel,iterations = 1) thresh\_2 = cv2.morphologyEx(thresh\_2, cv2.MORPH\_CLOSE, vertical\_kernal, iterations=5) 

horizontal\_kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (5,1)) thresh\_2 = cv2.morphologyEx(thresh\_2, cv2.MORPH\_CLOSE, horizontal\_kernel, iterations=5) 

plt.figure() 

plt.imshow(thresh\_2) 

plt.show()    

- In[78]: 

invert = 255 - thresh\_2 

cnts = cv2.findContours(invert, cv2.RETR\_TREE, cv2.CHAIN\_APPROX\_SIMPLE) 

cnts = cnts[0] if len(cnts) == 2 else cnts[1] 

(cnts, \_) = contours.sort\_contours(cnts, method="top-to-

bottom") ![](Aspose.Words.6704c83c-96b0-4f31-a80b-c5017e886641.012.png)

- In[79]: 
- model = model = tf.keras.models.load\_model('epic\_num\_reader.model') 
- pipeline = keras\_ocr.pipeline.Pipeline() 

pytesseract.pytesseract.tesseract\_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' 

- In[80]: 

sudoku\_rows = [] 

row = [] 

ltext = [] 

for (i, c) in enumerate(cnts, 1): 

`    `area = cv2.contourArea(c) 

`    `if area < 50000: 

`        `row.append(c) 

`        `if i % 9 == 0:   

`            `(cnts, \_) = contours.sort\_contours(row, method="left-to-right") 

`            `sudoku\_rows.append(cnts) 

`            `row = [] 

- Iterate through each box 

for row in sudoku\_rows: 

`    `for c in row: 

`        `mask = np.zeros(imagewrap.shape, dtype=np.uint8)         cv2.drawContours(mask, [c], -1, (255,255,255), - 1) 

`        `result = cv2.bitwise\_and(imagewrap, mask) 

`        `result[mask==0] = 255 

- result = (cv2.dilate(result,(5,5),iterations = 

1)) 

- sharpen\_kernel = np. array([[-1,-1,-1], [-1,9,-

1], [-1,-1,-1]]) 

- result = cv2. filter2D(result, -1, 

sharpen\_kernel) 

`        `blur = cv2.GaussianBlur(result, (3,3),6) 

- ret,result = 

cv2.threshold(blur,0,255,cv2.THRESH\_BINARY+cv2.THRESH\_OTS U) 

`        `result = cv2.adaptiveThreshold(blur,255,1,1,11,2)         erosion = cv2.erode(result,(5,5),iterations = 1)         cv2.imshow('result', erosion) 

- result = resize(result, (result.shape[0],28, 

28,1)) 

- predictions = model.predict(result) 
- # print(predictions[0]) 
- t = (np.argmax(predictions)) 
- print(t) 

`        `text = pytesseract.image\_to\_string(result,lang='eng',config='-- psm 10 --oem 3 -c tessedit\_char\_whitelist=0123456789-') 

- #pyt.image\_to\_string(result, 

lang="eng",config='--psm 6 --oem 3') #builder=builder) 

`        `text = text.strip() 

`        `ltext.append(text) 

`        `print(text) 

- pred = model.predict(result) 
- print(pred.argmax()) 

`        `cv2.waitKey(15)  

cv2.imshow('thresh', thresh\_2) cv2.imshow('invert', invert) cv2.waitKey() 

- In[81]: 

i = 0 

while i < len(ltext): 

`    `if ltext[i] == '':         ltext[i] = '0' 

`    `if ltext[i] == '-':         ltext[i] = '0' 

ltext[i] = int(ltext[i]) i += 1 

ltextmatrix = np.reshape(ltext,(9,9)) ltextmatrix 

- In[86]: 

def next\_box(quiz): 

`    `for row in range(9): 

`        `for col in range(9): 

`            `if quiz[row][col] == 0:                 return (row, col) 

    return False 

#Function to fill in the possible values by evaluating rows collumns and smaller cells 

def possible (quiz,row, col, n): 

`    `#global quiz 

`    `for i in range (0,9): 

`        `if quiz[row][i] == n and row != i: 

`            `return False 

`    `for i in range (0,9): 

`        `if quiz[i][col] == n and col != i: 

`            `return False 

`    `row0 = (row)//3 

`    `col0 = (col)//3 

`    `for i in range(row0\*3, row0\*3 + 3): 

`        `for j in range(col0\*3, col0\*3 + 3): 

`            `if quiz[i][j]==n and (i,j) != (row, col):                 return False 

`    `return True 

#Recursion function to loop over untill a valid answer is found.  

def solve(quiz): 

`    `val = next\_box(quiz) 

`    `if val is False: 

`        `return True 

`    `else: 

`        `row, col = val 

`        `for n in range(1,10): #n is the possible solution             if possible(quiz,row, col, n): 

`                `quiz[row][col]=n 

`                `if solve(quiz): 

`                    `return True  

`                `else: 

`                    `quiz[row][col]=0 

`        `return  

def Solved(quiz): 

`    `for row in range(9): 

`        `if row % 3 == 0 and row != 0: 

`            `print("....................") 

`        `for col in range(9): 

`            `if col % 3 == 0 and col != 0: 

`                `print("|", end=" ") 

`            `if col == 8: 

`                `print(quiz[row][col]) 

`            `else: 

`                `print(str(quiz[row][col]) + " ", end="") 

- In[87]: 

if solve(ltextmatrix): 

`    `Solved(ltextmatrix) 

else: 

`    `print("Solution don't exist. Or Model misread digits.") 

- In[54]: 
- kernel = np.ones((5,5),np.uint8) 

no = cv2.morphologyEx(result, cv2.MORPH\_CLOSE, (5,5), iterations=5) 

plt.figure() 

plt.imshow(no) 

plt.show() 

- In[ ]: 
