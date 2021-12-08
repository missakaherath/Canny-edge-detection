import cv2
import math
import numpy as np
import sys

filename = "u" + ".png"   #enter image name here

img = cv2.imread(filename, 0)

if(img is None):
    print("Invalid image name")
    sys.exit()
else:
    print("Detecting edges...")

def getGaussianKernel(sigma = 1.5, size = 5): # generate the gaussian kernel
    
    kernel = [[0.0]*size for i in range(size)]

    h = size//2
    w = size//2

    for x in range(-h, h+1):
        for y in range(-w, w+1):
            normal = 1 / (2.0 * math.pi * sigma**2) 
            hx = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
 
            kernel[x+h][y+w] = hx*normal
            
    kernel_sum = sum(sum(kernel, []))
    gaussian_filter_kernel = [[i / kernel_sum for i in a] for a in kernel]
    
    #print(gaussian_filter_kernel)        
    return kernel


def wrap(img):  # wrap around the edge pixels
    img.insert(0, img[-1])
    img.append(img[1])
    for a in range (len(img)-2):
        img[a].insert(0, img[a][-1])
        img[a].append(img[a][1])
    return(img)

def applyFilter(img, mask): #apply filter mask to img
    offset = len(mask)//2
    outputImg = []

    for i in range(offset, len(img)-offset):
        outputRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val += (img[xn][yn] * mask[x][y])
            outputRow.append(val)
        outputImg.append(outputRow)
    return outputImg

#apply sobel filters
def sobel(img):
    sobelX = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelY = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    wrapedgradientEstimationX = wrap(img)
    gradientEstimationX = applyFilter(wrapedgradientEstimationX, sobelX)
    
    wrapedgradientEstimationY = wrap(gradientEstimationX)
    gradientEstimationY = applyFilter(wrapedgradientEstimationY, sobelY)
    
    #print(len(gradientEstimationX), len(gradientEstimationX[0]), len(gradientEstimationY), len(gradientEstimationY[0]))
    
    angleMatrixRow = [0] * len(gradientEstimationY[0])
    angleMatrix = []
    for i in range(len(gradientEstimationY)):
        angleMatrix.append(angleMatrixRow)
    
    #generate angle matrix
    for row in range(len(angleMatrix)):
        for i in range(len(angleMatrix[0])):
            if(gradientEstimationX[row][i]!=0):
                D = math.atan2(gradientEstimationY[row][i],gradientEstimationX[row][i])
                angleMatrix[row][i] = D * 180 / math.pi
            elif(gradientEstimationX[row][i]==0 and gradientEstimationY[row][i] > 0):
                D = math.pi/2
                angleMatrix[row][i] = D * 180 / math.pi
            else:
                D = -math.pi/2
                angleMatrix[row][i] = D * 180 / math.pi
            if(angleMatrix[row][i] < 0):
                angleMatrix[row][i]+=180
    
    return gradientEstimationY, angleMatrix

def nonMaxSuppression(img, angles): #non maxima suppression
  M, N = img.shape
  suppressedImg = np.zeros((M,N))
  for i in range(1,M-1):
      for j in range(1, N - 1):
          if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
              val = max(img[i, j+1], img[i, j-1])
          elif (22.5 <= angles[i, j] < 67.5):
              val = max(img[i+1, j-1], img[i-1, j+1])
          elif (67.5 <= angles[i, j] < 112.5):
              val = max(img[i+1, j], img[i-1, j])
          elif (112.5 <= angles[i,j] < 157.5):
              val = max(img[i-1, j-1], img[i+1, j+1])
          
          if img[i, j] >= val:
            suppressedImg[i, j] = img[i, j]
          else:
            suppressedImg[i, j] = 0

  return suppressedImg

def doubleThreshold(img, lowerThresholdRat, upperThresholdRat): #applying double threshold
    upperThreshold = img.max() * upperThresholdRat
    lowerThreshold = upperThreshold * lowerThresholdRat
    
    M, N = img.shape
    
    result = np.zeros((M,N))

    for i in range(1, M-1):
        for j in range(1, N-1):
            if(img[i,j] >= upperThreshold): # if strong
                result[i,j] = 255
            elif(img[i,j] >= lowerThreshold and img[i,j] <= upperThreshold):    # if weak
                result[i,j] = 25
    return result

'''--------------------------------------------run----------------------------------------------------'''

#Filter noise using gaussian filtering
kernel = getGaussianKernel(1,3) #getting the gaussian kernel, parametes - (sigma, size)
wrappedimg = wrap(img.tolist()) #wrap the image
noiseFiltered = applyFilter(wrappedimg, kernel)  #apply gaussian filtering

'''nparray = np.array(noiseFiltered)
cv2.imwrite("GaussianApplied.jpg", nparray) #save image '''

#apply gradient estimation
gradientEstimatedImg, angleMatrix = sobel(noiseFiltered)

'''nparray2 = np.array(gradientEstimatedImg)
cv2.imshow("Gradient estimation", nparray2)
cv2.waitKey(0)
cv2.destroyAllWindows()
imagefilename2 = "GradientEstimation.jpg"
cv2.imwrite(imagefilename2, nparray2) #save image'''

#apply Non-maxima suppression
nonmaxsupImg = nonMaxSuppression(np.array(gradientEstimatedImg), np.array(angleMatrix))


'''cv2.imshow("nonmaxsup", nonmaxsupImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

imagefilename3 = "nonmaximasup.jpg"
cv2.imwrite(imagefilename3, nonmaxsupImg) #save image'''

#apply double threshold
doubleThresholdAppliedImg = doubleThreshold(nonmaxsupImg, 0.04, 0.07) # parameters : (image, lower threshold ratio, upper threshold ratio)

'''cv2.imshow("Final Image", doubleThresholdAppliedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#print(doubleThresholdAppliedImg.shape)

cv2.imwrite("result.jpg", doubleThresholdAppliedImg) #save image

print("Output image saved!")
