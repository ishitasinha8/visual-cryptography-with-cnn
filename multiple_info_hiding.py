#importing packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import halftone
from math import log10, sqrt

#main function that calls other functions
def main():
    path1 = 'image1.png'     #path
    img1 = cv2.imread(path1,0)    #read image
    img1 = np.array(img1) #make np array

    path2 = 'image2.png'     #path
    img2 = cv2.imread(path2,0)    #read image
    img2 = np.array(img2) #make np array

    path3 = 'image3.png'     #path
    img3 = cv2.imread(path3,0)    #read image
    img3 = np.array(img3) #make np array

    path4 = 'image4.png'     #path
    img4 = cv2.imread(path4,0)    #read image
    img4 = np.array(img4) #make np array

    l1, b1 = img1.shape #get shape
    #print(l1,b1)
    l2, b2 = img2.shape #get shape
    #print(l2,b2)
    
    img1 = cv2.resize(img1,(512,512))
    img2 = cv2.resize(img2,(512,512))
    img3 = cv2.resize(img3,(512,512))
    img4 = cv2.resize(img4,(512,512))

    l1, b1 = img1.shape #get shape
    #print(l1,b1)
    l2, b2 = img3.shape #get shape
    #print(l2,b2)

    #display_as_plt(img1,"Secret Image - girl1")
    #display_as_plt(img2,"Secret Image - girl2")
    #display_as_plt(img3,"Secret Image - secret")

    maxl = max(l1,l2)
    maxb = max(b1,b2)

    l = max(maxl,maxb)
    b = l

    print(l,b)
    #create np arrays with 0 values
    secret1 = np.zeros([l,b],dtype=int)
    secret2 = np.zeros([l,b],dtype=int)
    secret3 = np.zeros([l,b],dtype=int)
    secret4 = np.zeros([l,b],dtype=int)

    r1 = np.zeros([l,b],dtype=int)
    r2 = np.zeros([l,b],dtype=int)
    #r3 = np.zeros([l,b],dtype=int)

    final1 = np.zeros([l,b],dtype=int)
    final2 = np.zeros([l,b],dtype=int)
    final3 = np.zeros([l,b],dtype=int)
    final4 = np.zeros([l,b],dtype=int)

    #BINARIZE THE IMAGES

    secret1 = binarize_cv2(img1)  #with default thresholding
    display_as_plt(secret1,"Binarized Image 1")

    secret2 = binarize_cv2(img2)  #with default thresholding
    display_as_plt(secret2,"Binarized Image 2")

    secret3 = binarize_cv2(img3)  #with default thresholding
    display_as_plt(secret3,"Binarized Image 3")

    secret4 = binarize_cv2(img4)  #with default thresholding
    display_as_plt(secret4,"Binarized Image 4")

    #CREATE FIRST RANDOM GRID
    r1 = create_r1(l,b)
    display_as_plt(r1,"First Random Grid")
    
    #CREATE SECOND RANDOM GRID AND SUPERIMPOSE RG1 AND RG2
    #AT 0 DEGREES
    r2 = create_r2_at_0(secret1,r1,r2,l,b)
    display_as_plt(r2,"Second Random Grid 0 degree")

    final1 = superimpose_at_0(r1,r2,final1,l,b)
    print(final1)
    display_as_plt(final1,"Superimposed Image 1 Image")

    #AT 90 DEGREES
    r2 = create_r2_at_90(secret2,r1,r2,l,b)
    display_as_plt(r2,"Second Random Grid 90 degree")

    final2 = superimpose_at_90(r1,r2,final2,l,b)
    display_as_plt(final2,"Superimposed Image 2 Image")

    #AT 180 DEGREES
    r2 = create_r2_at_180(secret3,r1,r2,l,b)
    display_as_plt(r2,"Second Random Grid 180 degree")

    final3 = superimpose_at_180(r1,r2,final3,l,b)
    display_as_plt(final3,"Superimposed Image 3 Image")

    #AT 270 DEGREES
    r2 = create_r2_at_270(secret4,r1,r2,l,b)
    display_as_plt(r2,"Second Random Grid 270 degree")

    final4 = superimpose_at_270(r1,r2,final4,l,b)
    display_as_plt(final4,"Superimposed Image 4 Image")

    #WRITE IN FILE
    write_in_file(final1, "image1_superimp.png")
    write_in_file(final2, "image2_superimp.png")
    write_in_file(final3, "image3_superimp.png")
    write_in_file(final4, "image4_superimp.png")

    #FIND THE PSNR
    find_avg_psnr(secret1,final1,secret2,final2,secret3,final3,secret4,final4)

    calc_contrast("image1_superimp.png")
    calc_contrast("image2_superimp.png")
    calc_contrast("image3_superimp.png")
    calc_contrast("image4_superimp.png")

    print("Pixel wise accuracy theoretical: ",cal_pixel_wise_accuracy(secret1, secret1))
    print("Pixel wise accuracy 1: ",cal_pixel_wise_accuracy(secret1, final1))
    print("Pixel wise accuracy 2: ",cal_pixel_wise_accuracy(secret2, final2))
    print("Pixel wise accuracy 3: ",cal_pixel_wise_accuracy(secret3, final3))
    print("Pixel wise accuracy 4: ",cal_pixel_wise_accuracy(secret4, final4))

#display the image open cv
def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return img

#display image as plt
def display_as_plt(img,name):
    plt.figure(name)
    plt.imshow(img,cmap='gray')
    plt.show()

#binarize secret image (img)
def binarize(img,l,b):
    for i in range(l):
      for j in range(b):
         if(img[i,j] > 128 and img[i,j]<255):
             img[i,j]=1 
         else:
             img[i,j]=0
    return img

#binarize secret image using cv2.THRESH_BINARY
def binarize_cv2(img):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh1


#create a random matrix r1 with arbitrary 0,1 values
def create_r1(l,b):
    #r1 = np.random.rand(l,b)
    r1 = np.random.randint(2, size=(l,b))
    return r1

#create second random matrix r2 that results from secret and r1
def create_r2_at_0(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b):
            if secret[i,j]==0:
                r2[i,j] = r1[i,j]
            else:
                r2[i,j] = 1 - r1[i,j] 
    return r2

#create second random matrix r2 rotated by 90 that results from secret and r1
def create_r2_at_90(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b):
            if(secret[j,l-i-1]==0):
                #print(j,l-i-1,end=",")
                r2[j,l-i-1] = r1[i,j]
            else:
                r2[j,l-i-1] = 1 - r1[i,j]
    return r2

#create second random matrix r2 rotated by 180 that results from secret and r1
def create_r2_at_180(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b):
            if(secret[l-i-1,l-j-1]==0):
                #print(j,l-i-1,end=",")
                r2[l-i-1,l-j-1] = r1[l-j-1,i]
            else:
                r2[l-i-1,l-j-1] = 1 - r1[l-j-1,i]
    return r2

#create second random matrix r2 rotated by 270 that results from secret and r1
def create_r2_at_270(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b):
            if(secret[l-j-1,i]==0):
                #print(j,l-i-1,end=",")
                r2[l-j-1,i] = r1[j,l-i-1]
            else:
                r2[l-j-1,i] = 1 - r1[j,l-i-1]
    return r2

#create the superimpose (XOR) of r1 and r2 at 0 degree
def superimpose_at_0(r1,r2,final,l,b):
    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    
    #display_as_plt(final,"Superimposed Image at 0 degree")
    return final

#create the superimpose (XOR) of r1 and r2 at 90 degree
def superimpose_at_90(r1,r2,final,l,b):
    r2 = np.rot90(r2)

    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0

    #display_as_plt(final,"Superimposed Image at 90 degree")
    final = np.rot90(final,3)
    return final

#create the superimpose (XOR) of r1 and r2 at 180 degree
def superimpose_at_180(r1,r2,final,l,b):
    r2 = np.rot90(r2,3)

    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    
    #display_as_plt(final,"Superimposed Image at 180 degree")

    final = np.rot90(final)
    return final

#create the superimpose (XOR) of r1 and r2 at 270 degree
def superimpose_at_270(r1,r2,final,l,b):
    r2 = np.rot90(r2,2)

    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    
    #display_as_plt(final,"Superimposed Image at 270 degree")
    final = np.rot90(final,2)
    return final

def hist_eq(im):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl1 = clahe.apply(im)
    return cl1

def find_avg_psnr(secret1,final1,secret2,final2,secret3,final3,secret4,final4):
    psnr1 = calc_psnr(secret1,secret1)
    print(psnr1) #100
    
    psnr1 = calc_psnr(secret1,final1)
    print(psnr1) #6.717608412375675

    psnr2 = calc_psnr(secret2,final2)
    print(psnr2) #1.4475864776022838

    psnr3 = calc_psnr(secret3,final3)
    print(psnr3) #10.905255530589526

    psnr4 = calc_psnr(secret4,final4)
    print(psnr4) #4.090313040351356

    avgPsnr = (psnr1 + psnr2 + psnr3 + psnr4)/4
    print("Average psnr = ",avgPsnr)

def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

'''
def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    
    # Calculate the maximum pixel value
    max_pixel_value = np.max(original)
    
    # Calculate the PSNR value
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    
    return psnr
'''

def cal_pixel_wise_accuracy(secret, final):
    secret_flat = secret.flatten()
    final_flat = final.flatten()

    # Calculate the accuracy by comparing each pixel
    accuracy = np.mean(secret_flat == final_flat)

    return accuracy

def calc_contrast(final):
    img = cv2.imread(final)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert to grayscale
    mean_val = np.mean(gray_img) #Calculate the mean pixel value
    contrast = np.mean(np.abs(gray_img - mean_val)) # Calculate the contrast
    print("Contrast:", contrast)

def write_in_file(final, image_name):
    final_norm = final * 255
    cv2.imwrite(image_name,final_norm)

if __name__ == "__main__":
    main()