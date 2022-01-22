import h5py
import numpy as np
import argparse
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import program
from imm_reader_with_plot import IMMReader8ID

class calculate_DB():
        
    def __init__(self, dim_x, dim_y, x_guess, y_guess, step_size):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_guess = x_guess
        self.y_guess = y_guess
        self.step_size = step_size
        
    def extract_pix2q(self, file):
        # blemish file if the fname is specified;
        with h5py.File(file, 'r') as hf:
            X_E = np.squeeze(hf.get('/measurement/instrument/source_begin/energy')[()])
            Det_Dist = np.squeeze(hf.get('/measurement/instrument/detector/distance')[()]) / 1000
            pix_size = np.squeeze(hf.get('/measurement/instrument/detector/x_pixel_size')[()]) * 1e-3       

        lam = 12.398/X_E;
        k0 = 2*np.pi/lam;
        pix2q = (pix_size/Det_Dist)*k0
              
        
        return pix2q

    def read_imm(self, imm_file):
        reader = IMMReader8ID(imm_file)
        img_2D = reader.calc_avg_pixel()
        # check if 2d array
        # print( len(img_2D. shape))
        return img_2D
    
    def plot_fig(self, ROI_Dev):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        colormap = plt.cm.jet
        colormap.set_under(color='w')
        
        im = ax.imshow(ROI_Dev, 
                     cmap=colormap, 
                     norm=colors.LogNorm(vmin=np.min(ROI_Dev), vmax=np.max(ROI_Dev)),
                     interpolation='none')
        fig.colorbar(im, ax=ax)
        plt.rc('font', size=20)
    
        return ax    
    
    def def_area(self, img_2D, pix2q):
        
        x = np.arange(0, len(img_2D[0]))
        y = np.arange(0, len(img_2D))    
                                                    
        pix_pos_x, pix_pos_y = np.meshgrid(x,y)        
        
        # row, col
        ROI_Dev = np.zeros((self.dim_y, self.dim_x))
    
        x0 = np.zeros(self.dim_x)    
        y0 = np.zeros(self.dim_y)
    
    
        for ii in range(self.dim_x):
            for jj in range(self.dim_y):        
    
                x0[ii] = self.x_guess + (ii-np.floor(self.dim_x/2))*self.step_size
                
                y0[jj] = self.y_guess + (jj-np.floor(self.dim_y/2))*self.step_size
                
                Q_map = np.sqrt((pix_pos_x-x0[ii])**2 + (pix_pos_y-y0[jj])**2)*pix2q

                bool_arr = np.where((Q_map > 0.0067) & (Q_map < 0.0073), 1, 0)        
      
                # np.nonzero returns indices 
                nz = np.nonzero(bool_arr) # indices
                Int_ROI = img_2D[nz] # 1D array 
                
                ROI_Dev[jj][ii] = np.var(Int_ROI)/np.square(np.mean(Int_ROI)) # mean^2  # normalization
            
        self.plot_fig(ROI_Dev)
        
        return ROI_Dev

    def find_com(self, Z):
        
        Y = len(Z)
        X = len(Z[0])
        Z = 1 - Z
        
        self.plot_fig(Z)
        
        sum_mass_X, sum_mass_Y, sum_mass  = 0, 0, 0
    
        for i in range(0, X):
            for j in range(0, Y):
                sum_mass_X = sum_mass_X + (i * Z[j,i])
                sum_mass_Y = sum_mass_Y + (j * Z[j,i])
                sum_mass = sum_mass + Z[j,i]

        print(sum_mass_X, sum_mass_Y, sum_mass)
        
        x_com = sum_mass_X/sum_mass
        y_com = sum_mass_Y/sum_mass
                
        return x_com, y_com 
    
    
def main():
    
    # x_guess = 780.6 
    # y_guess = 259.4   
    # dim_x = 81 
    # dim_y = 91 
    # step_size = 0.11
    
    #-----------------------------------------------------------------------------------
    
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument("x_guess",
                        help="value for x_guess")
    parser.add_argument("y_guess",
                        help="value for y_guess")
    parser.add_argument("dim_x",
                        help="value for dim_x")
    parser.add_argument("dim_y",
                        help="value for dim_y")    
    parser.add_argument("step_size",
                        help="value for step_size")

    args = parser.parse_args()
    # load the train and test data
    
    x_guess = float(args.x_guess)
    y_guess = float(args.y_guess)
    dim_x = int(args.dim_x)
    dim_y = int(args.dim_y)
    step_size = float(args.step_size)
     
    #-----------------------------------------------------------------------------------
    
    # hdf file
    file1 = '/Users/jeffr/Desktop/data/E005_D100_Lq1_025C_att03_001/E005_D100_Lq1_025C_att03_001_0001-1000.hdf'
    # IMM file    
    file2 = '/Users/jeffr/Desktop/data/E005_D100_Lq1_025C_att03_001/E005_D100_Lq1_025C_att03_001_00001-01000.imm'    
    
    
    db = calculate_DB(dim_x, dim_y, x_guess, y_guess, step_size)
    pix2q = db.extract_pix2q(file1)
    img_2D = db.read_imm(file2)   
    
    ROI_Dev = db.def_area(img_2D, pix2q)
    pd.DataFrame(ROI_Dev).to_csv("data.csv",header=False, index=False)

    
    x_com, y_com = db.find_com(ROI_Dev)

    print(x_com, y_com)

            
if __name__ == '__main__':
    main()