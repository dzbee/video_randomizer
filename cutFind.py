import numpy as np
import cv2

def cutFind(filePath):
    vid = cv2.VideoCapture(filePath)
    
    take = [];

    nFrame = vid.get(CV_CAP_PROP_FRAME_COUNT)
    qux = vid.read()
    for k in xrange(1,nFrame-1):
        vid.set(CV_CAP_PROP_POS_FRAMES,k)
        foo = qux
        qux = vid.read()
        scoreR = corr2(foo[:,:,2],qux[:,:,2])
        scoreG = corr2(foo[:,:,1],qux[:,:,1])
        scoreB = corr2(foo[:,:,0],qux[:,:,0])
        if mean([scoreR,scoreG,scoreB]) < .5:
            k
            close all;figure;imagesc(foo);figure;imagesc(qux);
            bla = input('Take? [1 = Yes / Else = No]');
        if bla ~= 1:
            take = [take k];
    
    return take
        
def corr2(imgA,imgB):
    imgA = imgA - np.mean(imgA)
    imgB = imgB - np.mean(imgB)
    corr = np.sum(imgA*imgB)/np.sqrt(np.sum(imgA*imgA)*np.sum(imgB*imgB))

    return corr
