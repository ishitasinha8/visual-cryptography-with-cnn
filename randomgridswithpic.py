#importing packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import halftone

#main function that calls other functions
def main():
    path = 'image1.png'     #path
    img = cv2.imread(path,0)    #read image
    img = np.array(img) #make np array
    l, b = img.shape #get shape
    display_as_plt(img,"Secret Image")

    #create np arrays with 0 values
    secret = np.zeros([l,b],dtype=int)
    r1 = np.zeros([l,b],dtype=int)
    r2 = np.zeros([l,b],dtype=int)
    final = np.zeros([l,b],dtype=int)

    img = hist_eq(img)
    display_as_plt(img,"1")

    #secret = binarize(img,l,b) #without default thresholding
    #display_as_plt(secret,"Binarized Image")

    secret = binarize_cv2(img)  #with default thresholding
    print(secret)
    display_as_plt(secret,"Binarized Image")

    r1 = create_r1(l,b)
    display_as_plt(r1,"First Random Grid")

    r2 = create_r2(secret,r1,r2,l,b)
    display_as_plt(r2,"Second Random Grid")

    final = superimpose(r1,r2,final,l,b)
    display_as_plt(final,"Superimposed Secret Image")

    print("PSNR: ",calc_psnr(secret, final))
    

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
    #ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)  #bad quality of superimposed image
    #ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    return thresh1


#create a random matrix r1 with arbitrary 0,1 values
def create_r1(l,b):
    #r1 = np.random.rand(l,b)
    r1 = np.random.randint(2, size=(l,b))
    return r1

#create second random matrix r2 that results from secret and r1
def create_r2(secret,r1,r2,l,b):
    for i in range(l):
      for j in range(b):
         if secret[i,j]==0:
            r2[i,j] = r1[i,j]
         else:
            r2[i,j] = 1-r1[i,j] 
    return r2

#create the superimpose (XOR) of r1 and r2
def superimpose(r1,r2,final,l,b):
    for i in range(l):
        for j in range(b):
            if((r1[i,j]==1 and r2[i,j]==0)or(r1[i,j]==0 and r2[i,j]==1)):
                final[i,j] = 1
            else:
                final[i,j] = 0
    return final

def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    
    # Calculate the maximum pixel value
    max_pixel_value = np.max(original)
    
    # Calculate the PSNR value
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    
    return psnr

def hist_eq(im):
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
   cl1 = clahe.apply(im)
    #ret, cl1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
   #blur = cv2.GaussianBlur(img,(5,5),0)
   #ret3,cl1 = cv2.threshold(blur,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   return cl1


if __name__ == "__main__":
    main()