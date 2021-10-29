import os
import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# import matplotlib.pyplot as plt


# import other programs
from imm_reader_with_plot import IMMReader8ID
from rigaku_reader import RigakuReader
from hdf2sax import hdf2saxs

pg.setConfigOptions(imageAxisOrder='row-major')



def normalize(arr):
    # normalize arr so it's range is between 0 and 1
    vmin = np.min(arr)
    vmax = np.max(arr)

    if vmax - vmin > 1E-10:
        return (arr - vmin) / (vmax - vmin)
    else:
        return np.copy(arr)


def add_dimension(x, num):
    assert x.ndim == 1
    new_x = np.zeros((num, x.size), dtype=x.dtype)
    new_x[:] = x
    return new_x


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.data_raw = None
        self.shape = None
        self.qmap = None
        self.mask = None
        self.is_rotate = False
        self.saxs_min = 1

        self.new_partition = None
        self.meta = None

        self.hdl = pg_hdl
        self.infobar = infobar
        self.extent = None
        self.hdl.scene.sigMouseMoved.connect(self.show_location)

        self.idx_map = {
            0: "scattering",
            # 1: "scattering * (1 - mask)",
            1: "scattering * mask",
            2: "mask",
            # 4: "qr",
            # 5: "qx",
            # 6: "qy",
            3: "dqmap_partition",
            4: "sqmap_partition",
            5: "preview"
        }

    def load_meta(self, fname):
        keys = {
            'ccdx': '/measurement/instrument/acquisition/stage_x',
            'ccdx0': '/measurement/instrument/acquisition/stage_zero_x',
            'ccdz': '/measurement/instrument/acquisition/stage_z',
            'ccdz0': '/measurement/instrument/acquisition/stage_zero_z',
            'datetime': '/measurement/instrument/source_begin/datetime',
            'energy': '/measurement/instrument/source_begin/energy',
            'det_dist': '/measurement/instrument/detector/distance',
            'pix_dim': '/measurement/instrument/detector/x_pixel_size',
            'bcx': '/measurement/instrument/acquisition/beam_center_x',
            'bcy': '/measurement/instrument/acquisition/beam_center_y',
        }
        meta = {}
        with h5py.File(fname, 'r') as f:
            for key, val in keys.items():
                meta[key] = np.squeeze(f[val][()])
        return meta

    def save_partition(self, save_fname):
        # if no partition is computed yet
        if self.new_partition is None:
            return

        with h5py.File(save_fname, 'w') as hf:
            if '/data' in hf:
                del hf['/data']

            data = hf.create_group('data')
            data.create_dataset('mask', data=self.mask)
            for key, val in self.new_partition.items():
                data.create_dataset(key, data=val)

            # directories that remain the same
            dt = h5py.vlen_dtype(np.dtype('int32'))
            version = data.create_dataset('Version', (1,), dtype=dt)
            version[0] = [5]
            xspec = data.create_dataset('xspec', (1,), dtype=dt)
            xspec[0] = [-1]
            yspec = data.create_dataset('yspec', (1,), dtype=dt)
            yspec[0] = [-1]

            maps = data.create_group("Maps")
            dt = h5py.special_dtype(vlen=str)
            map1name = maps.create_dataset('map1name', (1,), dtype=dt)
            map1name[0] = 'q'

            map2name = maps.create_dataset('map2name', (1,), dtype=dt)
            map2name[0] = 'phi'

            empty_arr = np.array([])
            maps.create_dataset('q', data=self.qmap['q'])
            maps.create_dataset('phi', data=self.qmap['phi'])
            maps.create_dataset('x', data=empty_arr)
            maps.create_dataset('y', data=empty_arr)

            for key, val in self.meta.items():
                if key == 'saxs':
                    continue
                data.create_dataset(key, data=val)
        print('partition map is saved')

    def ver_hdf(self, file):
        with h5py.File(file, 'r') as hf:
            if file.endswith('.hdf'):
                if '/measurement/instrument/acquisition' in hf:
                    return True
                else:
                    return False
            else:
                return False

    def file_search(self, fname):
        # seeks directory of existing hdf program
        root = os.path.dirname(os.path.abspath(fname))
        dir_path = os.path.dirname(os.path.realpath(fname))
        files = [f for f in os.listdir(
            dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for root, dirs, files in os.walk(dir_path):
            for fname in files:

                if fname.endswith('.bin'):
                    print("-----------.bin found.-----------")
                    bin_file = root+'/'+str(fname)
                    # print(bin_file)
                    reader = RigakuReader(bin_file)
                    img_2D = reader.load()
                    # check if 2d array
                    # print( len(img_2D. shape))
                    return img_2D

                # seeks .imm file
                elif fname.endswith('.imm'):
                    print("-----------.imm found.-----------")
                    imm_file = root+'/'+str(fname)
                    # print(imm_file)
                    reader = IMMReader8ID(imm_file)
                    img_2D = reader.calc_avg_pixel()
                    # check if 2d array
                    # print( len(img_2D. shape))
                    return img_2D

                # seeks .h5 file
                elif fname.endswith('.h5'):
                    print("-----------.h5 found.-----------")
                    h5_file = root+'/'+str(fname)
                    # print(h5_file)
                    with h5py.File(h5_file, 'r') as hf:
                        e = '/entry/data/data' in hf
                        if e == True:
                            # correct h5 file, contains raw data
                            print("h5 raw data fname found.")
                            y = hdf2saxs(h5_file, num_frames=100)
                            assert y.ndim == 2
                            # check if 2d array
                            # print( len(y. shape))
                            return y
                        else:
                            print(
                                "You appeared to have inputted an incorrect or incorrectly formatted h5 file.")
                            return None

        return None
    
    
    
    
    
    
    
#-------------------------------------------------------------------------------------------------
    # generate 2d saxs
    def read_data(self, fname=None, blemish_fname=None, text_fname=None):
        # blemish file if the fname is specified;
        if blemish_fname is not None:
            
            x_list, y_list = self.read_txt(text_fname)        
            with h5py.File(blemish_fname, 'r') as hf:
                blemish = np.squeeze(hf.get('/lambda_pre_mask')[()])
                blemish = np.rot90(blemish, 3)
                blemish = np.flip(blemish, 1)
                
            for i in range(len(x_list)):
                blemish[x_list[i]][y_list[i]] = 0              
        else:
            blemish = None

        if not self.ver_hdf(fname):
            raise ValueError('check raw file')

        with h5py.File(fname, 'r') as f:
            saxs = self.file_search(fname)

        self.meta = self.load_meta(fname)

        # keep same
        self.data_raw = np.zeros(shape=(5, *saxs.shape))
        if blemish is None:
            self.mask = np.ones(saxs.shape, dtype=np.bool)
        else:
            if saxs.shape != blemish.shape:
                blemish = blemish.swapaxis(0, 1)
            self.mask = blemish

        self.shape = self.data_raw[0].shape
        self.qmap = self.compute_qmap()
        self.extent = self.compute_extent()

        self.min_val = np.min(saxs[saxs > 0])
        saxs = np.log10(saxs + self.min_val)
        self.meta['saxs'] = saxs
        self.data_raw[0] = saxs

        saxs_mask = saxs * self.mask
        min_val = np.min(saxs_mask[self.mask > 0.5])
        saxs_mask[self.mask < 0.5] = min_val
        self.data_raw[1] =saxs_mask 
        self.data_raw[2] = self.mask

    def compute_qmap(self):
        k0 = 2 * np.pi / (12.398 / self.meta['energy'])
        v = np.arange(self.shape[0], dtype=np.uint32) - self.meta['bcy'] 
        h = np.arange(self.shape[1], dtype=np.uint32) - self.meta['bcx']
        vg, hg = np.meshgrid(v, h, indexing='ij')

        r = np.sqrt(vg * vg + hg * hg) * self.meta['pix_dim']
        phi = np.arctan2(vg, hg)
        phi[phi < 0] = phi[phi < 0] + np.pi * 2.0

        alpha = np.arctan(r / self.meta['det_dist'])
        qr = np.sin(alpha) * k0
        qr = 2 * np.sin(alpha / 2) * k0
        qx = qr * np.cos(phi)
        qy = qr * np.sin(phi)

        qmap = {
            'phi': phi.astype(np.float32),
            'alpha': alpha.astype(np.float32),
            'q': qr.astype(np.float32),
            'qx': qx.astype(np.float32),
            'qy': qy.astype(np.float32)
        }

        return qmap

    def compute_extent(self):
        k0 = 2 * np.pi / (12.3980 / self.meta['energy'])
        x_range = np.array([0, self.shape[1]]) - self.meta['bcx']
        y_range = np.array([-self.shape[0], 0]) + self.meta['bcy']
        x_range = x_range * self.meta['pix_dim'] / self.meta['det_dist'] * k0
        y_range = y_range * self.meta['pix_dim'] / self.meta['det_dist'] * k0
        # the extent for matplotlib imshow is:
        # self._extent = xmin, xmax, ymin, ymax = extent
        # convert to a tuple of 4 elements;

        return (*x_range, *y_range)

    def show_location(self, pos):

        if not self.hdl.scene.itemsBoundingRect().contains(pos) or \
           self.shape is None:
            return

        shape = self.shape
        mouse_point = self.hdl.getView().mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        if col < 0 or col >= shape[1]:
            return
        if row < 0 or row >= shape[0]:
            return

        qx = self.qmap['qx'][row, col]
        qy = self.qmap['qy'][row, col]
        phi = self.qmap['phi'][row, col] * 180 / np.pi
        val = self.data[self.hdl.currentIndex][row, col]

        # msg = f'{self.idx_map[self.hdl.currentIndex]}: ' + \
        msg = f'[x={col:4d}, y={row:4d}, ' + \
              f'qx={qx:.04f}Å⁻¹, qy={qy:.06f}Å⁻¹, phi={phi:.1f}deg], ' + \
              f'val={val}'

        self.infobar.clear()
        self.infobar.setText(msg)

        return None

    def show_saxs(self, cmap='jet', log=True, invert=False, rotate=False,
                  plot_center=True, plot_index=0, **kwargs):
        if self.meta is None or self.data_raw is None:
            return
        # self.hdl.reset_limits()
        self.hdl.clear()
        self.data = np.copy(self.data_raw)

        center = (self.meta['bcx'], self.meta['bcy'])
        if rotate:
            self.data = np.swapaxes(self.data, 1, 2)
            center = [center[1], center[0]]
        self.is_rotate = rotate

        if not log:
            self.data[0] = 10 ** self.data[0]

        if invert:
            temp = np.max(self.data[0]) - self.data[0]
            self.data[0] = temp

        # <----------------------------------------------
        self.hdl.setImage(self.data)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[0]], y=[center[1]], symbol='+', size=15)
            self.hdl.add_item(t)

        self.hdl.setCurrentIndex(plot_index)

        return
    
    def select_mask(self, mask_preload, mask_directory):
        
        # with h5py.File(mask_preload, 'r') as hf:
        #     mask = np.squeeze(hf.get(mask_directory)[()])
        #     mask = np.rot90(mask, 3)
        #     mask = np.flip(mask, 1)

        with h5py.File(mask_preload, 'r') as hf:
            mask = np.squeeze(hf.get('/mask_triangular')[()])
            mask = np.rot90(mask, 3)
            mask = np.flip(mask, 1)

        print("mask selected")
        return mask


    def preview_mask(self, file):
        print("test")

    def apply_mask(self, mask):
        self.mask = self.mask * mask
        self.data[1:] *= self.mask

    def apply_roi(self):
        
        if self.meta is None or self.data_raw is None:
            return
        if len(self.hdl.roi) <= 0:
            return

        ones = np.ones(self.data[0].shape, dtype=np.bool)
        mask_n = np.zeros_like(ones, dtype=np.bool)
        mask_e = np.zeros_like(mask_n)
        mask_i = np.zeros_like(mask_n)

        for x in self.hdl.roi:
            # get ride of the center plot if it's there
            if isinstance(x, pg.ScatterPlotItem):
                continue
            # else
            mask_temp = np.zeros_like(ones, dtype=np.bool)
            # return slice and transfrom
            sl, _ = x.getArraySlice(self.data[1], self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            # sometimes the roi size returned from getArraySlice and
            # getArrayRegion are different;
            nz_idx = np.nonzero(y)

            h_beg = np.min(nz_idx[1])
            h_end = np.max(nz_idx[1])

            v_beg = np.min(nz_idx[0])
            v_end = np.max(nz_idx[0])

            sl_v = slice(sl[0].start, sl[0].start + v_end - v_beg + 1)
            sl_h = slice(sl[1].start, sl[1].start + h_end - h_beg + 1)
            mask_temp[sl_v, sl_h] = y[v_beg:v_end + 1, h_beg: h_end + 1]

            if x.sl_mode == 'exclusive':
                mask_e[mask_temp] = 1
            elif x.sl_mode == 'inclusive':
                mask_i[mask_temp] = 1

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i

        self.mask = mask_p
        # self.data[1] = self.data[0] * (1 - mask_p)
        self.data[1] = self.data[0] * mask_p
        self.data[2] = self.mask
        self.hdl.setImage(self.data)
        self.hdl.setCurrentIndex(2)

        return self.mask

    def add_roi(self, num_edges=None, radius=60, color='r', sl_type='Polygon',
                width=3, sl_mode='exclusive'):
        
        # self.add_roi_test() 
        # return

        shape = self.data.shape
        cen = (shape[1] // 2, shape[2] // 2)
        if sl_mode == 'inclusive':
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        if sl_type == 'Ellipse':
            new_roi = pg.EllipseROI([cen[1], cen[0]], [60, 80], pen=pen,
                                    removable=True, hoverPen=pen)
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == 'Circle':
            new_roi = pg.CircleROI([cen[1], cen[0]], [60, 80], pen=pen,
                                   removable=True, hoverPen=pen)

        elif sl_type == 'Polygon':
            if num_edges is None:
                num_edges = np.random.random_integers(6, 10)

            # add angle offset so that the new rois don't overlap with each other
            offset = np.random.random_integers(0, 359)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = radius * np.cos(theta) + cen[1]
            y = radius * np.sin(theta) + cen[0]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, pen=pen,
                                     removable=True, hoverPen=pen)

        elif sl_type == 'Rectangle':
            new_roi = pg.RectROI([cen[1], cen[0]], [30, 150], pen=pen,
                                 removable=True, hoverPen=pen)
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addRotateHandle([0,1], [0.5, 0.5])

        else:
            raise TypeError('type not implemented. %s' % sl_type)

        new_roi.sl_mode = sl_mode
        self.hdl.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(new_roi))
        return

    def remove_roi(self, roi):
        self.hdl.remove_item(roi)

    def compute_partition(self, dq_num=10, sq_num=100, style='linear',
                          dp_num=36, sp_num=360):
        if self.meta is None or self.data_raw is None:
            return

        if sq_num % dq_num != 0:
            raise ValueError('sq_num must be multiple of dq_num')

        if sp_num % dp_num != 0:
            raise ValueError('sq_num must be multiple of dq_num')

        qmap = self.qmap['q'] * self.mask

        qmap_valid = qmap[self.mask == True]
        qmin = np.min(qmap_valid)
        qmax = np.max(qmap_valid)

        if style == 'linear':
            dqspan = np.linspace(qmin, qmax, dq_num + 1)
            sqspan = np.linspace(qmin, qmax, sq_num + 1)
            dphi = np.linspace(0, np.pi * 2.0, dp_num + 1)
            sphi = np.linspace(0, np.pi * 2.0, sp_num + 1)

            dqval_list = (dqspan[1:] + dqspan[:-1]) / 2.0
            sqval_list = (sqspan[1:] + sqspan[:-1]) / 2.0

        elif style == 'logarithmic':
            qmin = np.log10(qmin)
            qmax = np.log10(qmax)
            dqspan = np.logspace(qmin, qmax, dq_num + 1)
            sqspan = np.logspace(qmin, qmax, sq_num + 1)

            dphi = np.linspace(0, np.pi * 2.0, dp_num + 1)
            sphi = np.linspace(0, np.pi * 2.0, sp_num + 1)

            dqval_list = np.sqrt(dqspan[1:] * dqspan[:-1])
            sqval_list = np.sqrt(sqspan[1:] * sqspan[:-1])
        else:
            raise ValueError("style not supported")

        dqmap_partition = np.zeros_like(qmap, dtype=np.uint32)
        sqmap_partition = np.zeros_like(qmap, dtype=np.uint32)

        # dqval
        for n in range(dq_num):
            qval = dqspan[n]
            dqmap_partition[qmap >= qval] = n + 1

        # sqval
        for n in range(sq_num):
            qval = sqspan[n]
            sqmap_partition[qmap >= qval] = n + 1

        dphi_partition = np.zeros_like(qmap, dtype=np.uint32)
        sphi_partition = np.zeros_like(qmap, dtype=np.uint32)

        # phi partition starts from 0; not 1
        for n in range(dp_num):
            dphi_partition[self.qmap['phi'] >= dphi[n]] = n
        dphival = np.unique(dphi_partition)
        dphival.sort()
        dphispan = np.linspace(0, 2 * np.pi, dp_num + 1)

        # phi partition starts from 0; not 1
        for n in range(sp_num):
            sphi_partition[self.qmap['phi'] >= sphi[n]] = n
        sphival = np.unique(sphi_partition)
        sphival.sort()
        sphispan = np.linspace(0, 2 * np.pi, sp_num + 1)

        dyn_combined = np.zeros_like(dqmap_partition, dtype=np.uint32)
        sta_combined = np.zeros_like(dqmap_partition, dtype=np.uint32)

        # dqmap, dqlist
        for n in range(dp_num):
            idx = dphi_partition == n
            dyn_combined[idx] = dqmap_partition[idx] + n * dq_num

        # sqmap, sqlist
        for n in range(sp_num):
            idx = sphi_partition == n
            sta_combined[idx] = sqmap_partition[idx] + n * sq_num

        self.data[3] = dyn_combined * self.mask
        self.data[4] = sta_combined * self.mask
        self.hdl.setCurrentIndex(3)

        partition = {
            'dqval': dqval_list,
            'sqval': sqval_list,
            'dynamicMap': dqmap_partition,
            # 'dynamicQList': dqlist,
            # 'staticQList': sqlist,
            'staticMap': sqmap_partition,
            'dphival': dphival,
            'sphival': sphival,
            'dqspan': dqspan,
            'sqspan': sqspan,
            'dphispan': dphispan,
            'sphispan': sphispan,
            'dnophi': dp_num,
            'snophi': sp_num,
            'dnoq': dq_num,
            'snoq': sq_num
            }

        for key in ['dphival', 'dqspan', 'dqval']:
            partition[key] = add_dimension(partition[key], dp_num)
        for key in ['sphival', 'sqspan', 'sqval']:
            partition[key] = add_dimension(partition[key], sp_num)

        self.new_partition = partition 

        return partition 

    def update_parameters(self, val):
        assert(len(val) == 5)
        for idx, key in enumerate(
            ['bcx', 'bcy', 'energy', 'pix_dim', 'det_dist']):
            self.meta[key] = val[idx]
        self.qmap = self.compute_qmap()

    def get_parameters(self):
        val = []
        for key in ['bcx', 'bcy', 'energy', 'pix_dim', 'det_dist']:
            val.append(self.meta[key])
        return val


    def read_txt(self, file):
        x_list = []
        y_list = []
        with open(file) as f:
            for line in f:
                x, y = line.strip().split(",")
                x_list.append(int(x) - 1)
                y_list.append(int(y) - 1)
        return x_list, y_list



def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'

    # fname = '\Desktop\sheyfer202106\sheyfer202106\A004_D100_att0_25C_Rq0_00001\A004_D100_att0_25C_Rq0_00001_0001-100000.hdf'

    sm = SimpleMask()
    sm.read_data(fname)
    # sm.show_saxs()
    # sm.compute_qmap()


if __name__ == '__main__':
    test01()