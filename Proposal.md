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
