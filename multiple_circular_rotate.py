#importing packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import halftone
from math import log10, sqrt

#main function that calls other functions
def main():
    path1 = 'image1.png'     #path
    img1 = cv2.imread(path1,0)    #read image
    img1 = np.array(img1) #make np array

    path2 = 'image2.jpg'     #path
    img2 = cv2.imread(path2,0)    #read image
    img2 = np.array(img2) #make np array

    path3 = 'image3.png'     #path
    img3 = cv2.imread(path3,0)    #read image
    img3 = np.array(img3) #make np array

    #defining length (l), breadth (b), number of images to hide (m)
    l = 200
    b = 120
    m = 3
    k = np.random.randint(1,m+1)

    #resizing images to standard length and breadth
    img1 = cv2.resize(img1,(b,l))
    img2 = cv2.resize(img2,(b,l))
    img3 = cv2.resize(img3,(b,l))

    #initializing the np arrays
    secret1 = np.zeros([l,b],dtype=int)
    secret2 = np.zeros([l,b],dtype=int)
    secret3 = np.zeros([l,b],dtype=int)

    final1 = np.zeros([l,b],dtype=int)
    final2 = np.zeros([l,b],dtype=int)
    final3 = np.zeros([l,b],dtype=int)

    r1 = np.zeros([l,b*m],dtype=int)
    r2 = np.zeros([l,b*m],dtype=int)

    #binarize the images
    secret1 = binarize_cv2(img1)
    secret2 = binarize_cv2(img2)
    secret3 = binarize_cv2(img3)

    r1 = create_r1(l,b*m)
    print("r1\n",r1)
    display_as_plt(r1,"r1")

    r2 = create_r2_at_0(secret1,r1,r2,l,b,1) #creating r2 with secret k=1
    print("r2 with secret1 \n",r2)

    r2 = create_r2_at_120(secret2,r1,r2,l,b,2) #creating r2 with secret k=2
    print("r2 with secret2 \n",r2)

    r2 = create_r2_at_240(secret3,r1,r2,l,b,3) #creating r2 with secret k=3
    print("r2 with secret3 \n",r2)

    if (k==1): #rotating r2 by (k = 1) 360/m degrees
        final1 = superimpose_at_0(r1,r2,final1,l,b,k)
        display_as_plt(final1,"Superimposed Image 1 Image")
    
    elif (k==2): #rotating r2 by (k = 2) 2*360/m
        final2 = superimpose_at_60(r1,r2,final2,l,b,k)
        display_as_plt(final2,"Superimposed Image 2 Image")
    
    elif (k==3): #rotating r2 by by (k = 3) 3*360/m
        final3 = superimpose_at_180(r1,r2,final3,l,b,k)
        display_as_plt(final3,"Superimposed Image 3 Image")
    
    psnr1 = calc_psnr(secret1,final1)
    psnr2 = calc_psnr(secret2,final2)
    psnr3 = calc_psnr(secret3,final3)

    print(psnr1,psnr2,psnr3)
#display image as plt
def display_as_plt(img,name):
    plt.figure(name)
    plt.imshow(img,cmap='gray')
    plt.show()
    
#binarize secret image using cv2.THRESH_BINARY
def binarize_cv2(img):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh1

#create a random matrix r1 with arbitrary 0,1 values
def create_r1(l,b):
    r1 = np.random.randint(2, size=(l,b))
    return r1

#create second random matrix r2 that results from secret1 and r1
def create_r2_at_0(secret,r1,r2,l,b,k):
    for i in range(l):
        for j in range(b):
            if secret[i,j]==0:
                r2[i,j] = r1[i,j]
            else:
                r2[i,j] = 1 - r1[i,j] 
    return r2

#create second random matrix r2 that results from secret2 and r1 rotated by 60
def create_r2_at_120(secret,r1,r2,l,b,k):
    for i in range(l):
        for j in range(b,k*b):
            if secret[i,j-b]==0:
                r2[i,j] = r1[i,j]
            else:
                r2[i,j] = 1 - r1[i,j] 
    return r2

#create second random matrix r2 that results from secret2 and r1 rotated by 180
def create_r2_at_240(secret,r1,r2,l,b,k):
    for i in range(l):
        for j in range((k-1)*b,k*b):
            if secret[i,j-(k-1)*b]==0:
                r2[i,j] = r1[i,j]
            else:
                r2[i,j] = 1 - r1[i,j] 
    return r2

#create the superimpose (XOR) of r1 and r2
def superimpose_at_0(r1,r2,final,l,b,k):
    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    
    return final

#create the superimpose (XOR) of r1 and r2 rotated by k=1
def superimpose_at_120(r1,r2,final,l,b,k):
    for i in range(l):
        for j in range(b,k*b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j-b] = 1
            else:
                final[i,j-b] = 0
    
    return final

#create the superimpose (XOR) of r1 and r2 rotated by k=2
def superimpose_at_240(r1,r2,final,l,b,k):
    for i in range(l):
        for j in range((k-1)*b,k*b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j-2*b] = 1
            else:
                final[i,j-2*b] = 0
    
    return final

def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

if __name__ == "__main__":
    main()