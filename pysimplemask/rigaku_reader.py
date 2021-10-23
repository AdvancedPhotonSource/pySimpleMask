import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class RigakuReader():
    def __init__(self, file):
        self.file = file
    
    def load(self):
        with open(self.file, 'r') as f:
            a = np.fromfile(f, dtype=np.uint64)

            # pixel index, pixel count, pixel frame
            b = (a >> 5+11)
            pix_ind = (b & 2**21-1).astype(int)
            pix_count = (a & 2**12-1).astype(int)
            pix_frame = (a >> 64-24).astype(int)
            
            img = csr_matrix((pix_count, (pix_frame, pix_ind)), shape=(max(pix_frame)+1, 1024*512), dtype='float')
            img_2D = np.transpose(img.mean(axis=0).reshape(512,1024))
        return img_2D  

    def plot(self, img_2D):
        
        img_2D = np.array(img_2D)
        fig, ax = plt.subplots()
        ax.pcolor(img_2D, norm=colors.LogNorm(vmin=1e-5, vmax=0.5))
        plt.imshow(img_2D)
        plt.colorbar()
        
        
    def test_plot(self, img_2D, config):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        colormap = plt.cm.jet
        colormap.set_under(color='w')
        
        det_dist = config['detector_distance']
        ccd_x0 = config['beam_center_x']
        ccd_y0 = config['beam_center_y']
        pixel_size = config['pixel_size']
        x_energy = config['x_energy']
    
        pix2q = pixel_size/det_dist*(2*3.1416/(12.4/x_energy))
        y_min = ((0-ccd_x0)*pix2q).item()
        y_max = ((img_2D.shape[1]-ccd_x0)*pix2q).item()
        x_min = (0-ccd_y0)*pix2q.item()
        x_max = (img_2D.shape[0]-ccd_y0)*pix2q.item()
    
        im = ax.imshow(img_2D, 
                     cmap=colormap, 
                     norm=colors.LogNorm(vmin=1e-5, vmax=0.5),
                     interpolation='none', 
                     extent=([y_min, y_max, x_min, x_max]))
        fig.colorbar(im, ax=ax)
        plt.rc('font', size=20)
    
        
if __name__ == "__main__":
    
    # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

    # file = 'C:/Users/jeffr/Desktop/sheyfer202106/sheyfer202106/A004_D100_att0_25C_Rq0_00001/A004_D100_att0_25C_Rq0_00001.bin'
    # file = 'C:/Users/jeffr/Desktop/sheyfer202106/sheyfer202106/A004_D100_att0_25C_Rq0_00002/A004_D100_att0_25C_Rq0_00002.bin'
    # file = 'C:/Users/jeffr/Desktop/sheyfer202106/sheyfer202106/A004_D100_att0_25C_Rq0_00003/A004_D100_att0_25C_Rq0_00003.bin'
    # file = 'C:/Users/jeffr/Desktop/sheyfer202106/sheyfer202106/A004_D100_att0_25C_Rq0_00004/A004_D100_att0_25C_Rq0_00004.bin'
    file = 'C:/Users/jeffr/Desktop/data/sheyfer202106/sheyfer202106/A004_D100_att0_25C_Rq0_00005/A004_D100_att0_25C_Rq0_00005.bin'

    
    
    reader = RigakuReader(file)
    img_2D = reader.load()
    reader.plot(img_2D)

