import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class IMMReader8ID():
    def __init__(self, filename:str, no_of_frames:int=-1, skip_frames:int = 0):
        self.filename = filename
        self.no_of_frames = no_of_frames
        self.index_data = []
        self.value_data = []
        self.skip_frames = skip_frames
        self.frames_read = 0
        self.lil_mtx = None
        self.__load__()

    def __load__(self):
        with open(self.filename, "rb") as file:

            if self.skip_frames > 0:
                self.__skip__(file)
                
            header = self.__read_imm_header(file)
            self.rows, self.cols = header['rows'], header['cols']
            self.is_compressed = bool(header['compression'] == 6)
            num_pixels = header['dlen']
            frame_index = 0

            while True:
                try:
                    num_pixels = header['dlen']
                    if self.is_compressed:
                        indexes = np.fromfile(file, dtype=np.uint32, count=num_pixels)
                        values = np.fromfile(file, dtype=np.uint16, count=num_pixels)
                        self.index_data.append(indexes)
                        self.value_data.append(values)

                    else:
                        values = np.fromfile(file, dtype=np.uint16, count=num_pixels)
                        
                        #TODO
 
                    # Check for end of file.
                    if not file.peek(1):
                        break
                    header = self.__read_imm_header(file)
                    frame_index += 1
                    self.frames_read += 1
                    if self.no_of_frames != -1 and frame_index > self.no_of_frames:
                        break
                except Exception as err:
                    raise IOError("IMM file doesn't seems to be of right type") from err
        
    def __skip__(self, file):
        for _ in range(self.skip_frames):
            header = self.__read_imm_header(file)
            is_compressed = bool(header['compression'] == 6)
            num_pixels = header['dlen']
            payload_size = num_pixels * (6 if is_compressed else 2)
            file.read(payload_size)

    def data(self):
        
        return self.index_data, self.value_data

    def __read_imm_header(self, file):
        imm_headformat = "ii32s16si16siiiiiiiiiiiiiddiiIiiI40sf40sf40sf40sf40sf40sf40sf40sf40sf40sfffiiifc295s84s12s"
        imm_fieldnames = [
            'mode',
            'compression',
            'date',
            'prefix',
            'number',
            'suffix',
            'monitor',
            'shutter',
            'row_beg',
            'row_end',
            'col_beg',
            'col_end',
            'row_bin',
            'col_bin',
            'rows',
            'cols',
            'bytes',
            'kinetics',
            'kinwinsize',
            'elapsed',
            'preset',
            'topup',
            'inject',
            'dlen',
            'roi_number',
            'buffer_number',
            'systick',
            'pv1',
            'pv1VAL',
            'pv2',
            'pv2VAL',
            'pv3',
            'pv3VAL',
            'pv4',
            'pv4VAL',
            'pv5',
            'pv5VAL',
            'pv6',
            'pv6VAL',
            'pv7',
            'pv7VAL',
            'pv8',
            'pv8VAL',
            'pv9',
            'pv9VAL',
            'pv10',
            'pv10VAL',
            'imageserver',
            'CPUspeed',
            'immversion',
            'corecotick',
            'cameratype',
            'threshhold',
            'byte632',
            'empty_space',
            'ZZZZ',
            'FFFF'
        ]
        bindata = file.read(1024)
        imm_headerdat = struct.unpack(imm_headformat, bindata)
        imm_header = dict(zip(imm_fieldnames, imm_headerdat))

        return imm_header


 
    # average
    def calc_avg_pixel(self):
        pixel_sum = np.zeros(self.cols * self.rows)
        
        for idx in range(len(self.index_data)):
            pixel_sum[self.index_data[idx]] += self.value_data[idx]
        
        pixel_sum = np.reshape(pixel_sum, (self.rows, self.cols))                                     
        pixel_avg = pixel_sum/len(self.index_data)

        return pixel_avg
        
    def plot_pix_avg(self, pixel_avg, config):
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
        y_max = ((pixel_avg.shape[1]-ccd_x0)*pix2q).item()
        x_min = (0-ccd_y0)*pix2q.item()
        x_max = (pixel_avg.shape[0]-ccd_y0)*pix2q.item()
    
        im = ax.imshow(pixel_avg, 
                     cmap=colormap, 
                     norm=colors.LogNorm(vmin=1e-6, vmax=3e-1),
                     interpolation='none', 
                     extent=([y_min, y_max, x_min, x_max]))
        fig.colorbar(im, ax=ax)
        plt.rc('font', size=20)
    
        return ax

if __name__ == "__main__":
    # IMM_FILE = "C:/Users/jeffr/Desktop/comm202106/comm202106/E004_100nm_Lq0_000C_att00_001/E004_100nm_Lq0_000C_att00_001_00001-02000.imm"
    
    IMM_FILE = "/Users/jeffr/Desktop/data/H432_OH_100_025C_att05_001/H432_OH_100_025C_att05_001_00001-01000.imm"
    # reader = IMMReader8ID(IMM_FILE)
    reader.__load__()
    reader.calc_avg_pixel()

    # reader.plot_frame()
    # reader.plot_pixel_sum()


