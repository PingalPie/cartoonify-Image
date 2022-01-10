import cv2
import numpy as np

def ImageImporter(filename):
    img = cv2.imread(filename)
    
    return img

def EdgeDetection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    
    return edges

def ColorQuantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    critirea =  (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, critirea, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

print('Enter The File Name: (Including Extention)')
inpimg = input('')
img = ImageImporter(inpimg)
line_wdth = 9
blurval = 7
totalColors = int(input('How many colors should be: '))

edgeimg = EdgeDetection(img, line_wdth, blurval)
img = ColorQuantisation(img, totalColors)
blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edgeimg)
cv2.imwrite('cartoon.jpg', cartoon)
print('Done!')
