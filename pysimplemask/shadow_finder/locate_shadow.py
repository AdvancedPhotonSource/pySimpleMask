import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os

# import program
from rigaku_reader import RigakuReader

        

class estimate_shadow():
    
    def __init__(self, file, img_2D):
        self.file = file
        self.img_2D = img_2D
        
    # save png
    def save_img(self):
        
        root, _ = os.path.split(self.file) 
        path = root + '/'+ 'img_kmeans.png'        
        
        img_2D = np.array(self.img_2D)
        fig1, ax1 = plt.subplots()
        ax1.pcolor(img_2D, norm=colors.LogNorm(vmin=1e-5, vmax=0.5))
        plt.axis('off') 
        plt.imshow(img_2D)
        plt.savefig(path ,bbox_inches='tight',pad_inches=0)
        
        return path
            
    def kmeans(self, path):
        
        img = cv2.imread(path)
        
        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = img.reshape((-1,3))
         
        # Convert to float type
        pixel_vals = np.float32(pixel_vals)
        
        # criteria for stopping: 100 iterations or 85% accuracy,
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
         
        # random centroids, k clusters
        k = 2
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
        
        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
            
        # reshape data 
        segmented_image = segmented_data.reshape(img.shape)
        
        fig2, ax2 = plt.subplots()
        
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)) 

        return centers, segmented_image
    
    def extract_factor(self, path):
        
        orig_dim = self.img_2D.shape
        
        # altered dimensions
        img = cv2.imread(path)
        red_dim = img.shape
        
        factor = orig_dim[0]/red_dim[0]
        
        return factor
    
    def extract_centers(self, centers, segmented_image, factor):

        for i in range(len(centers)):
            c = centers[i]
            indices = np.where(np.all(segmented_image == c, axis=-1))
            ty = int(sum(indices[0])/len(indices[0]))
            tx = int(sum(indices[1])/len(indices[1]))
            
            tx = tx * factor
            ty = ty * factor     
            
            print("Shadow Guess (x,y): ", tx, ty)
        
if __name__ == "__main__":
    
    # bin file
    
    file = '/Users/jeffr/Desktop/data/F0014_10nm_Glass_006C_att00_Rq0_00001/F0014_10nm_Glass_006C_att00_Rq0_00001.bin'
    # file = '/Users/jeffr/Desktop/data/E0004_D100_006C_att00_Rq0_00001/E0004_D100_006C_att00_Rq0_00001.bin'
    # file = '/Users/jeffr/Desktop/data/B0005_12IDB_Standard_006C_att00_Rq0_00034/B0005_12IDB_Standard_006C_att00_Rq0_00034.bin'
    
    # normal plot
    reader = RigakuReader(file)
    img_2D = reader.load()
    reader.plot(img_2D)
    
    test = estimate_shadow(file, img_2D)
    path = test.save_img()
    
    centers, segmented_image = test.kmeans(path)
    factor = test.extract_factor(path)
    
    test.extract_centers(centers, segmented_image, factor)
    