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

    #display_as_plt(img1,"Secret Image - girl1")
    #display_as_plt(img2,"Secret Image - girl2")
    #display_as_plt(img3,"Secret Image - secret")

    l = 200
    b = 120
    m = 3
    k = np.random.randint(1,m+1)

    img1 = cv2.resize(img1,(b,l))
    img2 = cv2.resize(img2,(b,l))
    img3 = cv2.resize(img3,(b,l))

    #display_as_plt(img1,"Secret Image - 1")
    #display_as_plt(img2,"Secret Image - 2")
    #display_as_plt(img3,"Secret Image - 3")

    secret1 = np.zeros([l,b],dtype=int)
    secret2 = np.zeros([l,b],dtype=int)
    secret3 = np.zeros([l,b],dtype=int)

    final1 = np.zeros([l,b],dtype=int)
    final2 = np.zeros([l,b],dtype=int)
    final3 = np.zeros([l,b],dtype=int)

    r1 = np.zeros([l,b*3],dtype=int)
    r2 = np.zeros([l,b],dtype=int)

    secret1 = binarize_cv2(img1)
    secret2 = binarize_cv2(img2)
    secret3 = binarize_cv2(img3)

    #secret1 = np.array([[0,1,0],[1,0,1],[1,0,1],[0,0,0],[0,0,0]])
    #print("secret1\n",secret1)

    r1 = create_r1(l,b*3)
    print("r1\n",r1)

    r2 = create_r2_at_0(secret1,r1,r2,l,b)
    print("r2 with secret1 \n",r2)

    final1 = superimpose_at_0(r1,r2,final1,l,b)
    display_as_plt(final1,"Superimposed Image 1 Image")

    r2 = create_r2_at_60(secret2,r1,r2,l,b)
    print("r2 with secret2 \n",r2)

    final2 = superimpose_at_60(r1,r2,final2,l,b)
    display_as_plt(final2,"Superimposed Image 2 Image")

    r2 = create_r2_at_180(secret3,r1,r2,l,b)
    print("r2 with secret3 \n",r2)

    final3 = superimpose_at_180(r1,r2,final3,l,b)
    display_as_plt(final3,"Superimposed Image 3 Image")

    #r2 = create_r2_at_60(secret2,r1,r2,l,b)
    #print(r2)

    #r2 = create_r2_at_180(secret3,r1,r2,l,b)
    #print(r2)

    #WRITE IN FILE
    write_in_file(final1, "image1_superimp_circ.png")
    write_in_file(final2, "image2_superimp_circ.png")
    write_in_file(final3, "image3_superimp_circ.png")

    calc_contrast("image1_superimp_circ.png")
    calc_contrast("image2_superimp_circ.png")
    calc_contrast("image3_superimp_circ.png")
    

def display_as_plt(img,name):
    plt.figure(name)
    plt.imshow(img,cmap='gray')
    plt.show()
    
def binarize_cv2(img):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh1

def create_r1(l,b):
    r1 = np.random.randint(2, size=(l,b))
    return r1

def create_r2_at_0(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b):
            if secret[i,j]==0:
                r2[i,j] = r1[i,j]
            else:
                r2[i,j] = 1 - r1[i,j] 
    return r2

def create_r2_at_60(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(b,2*b):
            if secret[i,j-b]==0:
                r2[i,j-b] = r1[i,j]
            else:
                r2[i,j-b] = 1 - r1[i,j] 
    return r2

def create_r2_at_180(secret,r1,r2,l,b):
    for i in range(l):
        for j in range(2*b,3*b):
            if secret[i,j-2*b]==0:
                r2[i,j-2*b] = r1[i,j]
            else:
                r2[i,j-2*b] = 1 - r1[i,j] 
    return r2

def superimpose_at_0(r1,r2,final,l,b):
    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    
    #display_as_plt(final,"Superimposed Image at 0 degree")
    return final

def superimpose_at_60(r1,r2,final,l,b):
    for i in range(l):
        for j in range(b,2*b):
            if((r1[i,j]==1 and r2[i,j-b]==0)or(r1[i,j]==0 and r2[i,j-b]==1)):
                final[i,j-b] = 1
            else:
                final[i,j-b] = 0
    
    #display_as_plt(final,"Superimposed Image at 0 degree")
    return final

def superimpose_at_180(r1,r2,final,l,b):
    for i in range(l):
        for j in range(2*b,3*b):
            if((r1[i,j]==1 and r2[i,j-2*b]==0)or(r1[i,j]==0 and r2[i,j-2*b]==1)):
                final[i,j-2*b] = 1
            else:
                final[i,j-2*b] = 0
    
    #display_as_plt(final,"Superimposed Image at 0 degree")
    return final

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