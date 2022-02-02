import os
import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from area_mask import MaskAssemble

# import other programs
from imm_reader_with_plot import IMMReader8ID
from rigaku_reader import RigakuReader
from hdf2sax import hdf2saxs

pg.setConfigOptions(imageAxisOrder='row-major')


def combine_qp(q_num, p_num, qval_1d, pval_1d, qmap, pmap):
    combined_map = np.zeros_like(qmap, dtype=np.uint32)

    for n in range(q_num):
        idx = qmap == (n + 1)
        combined_map[idx] = pmap[idx] + n * p_num
    combined_map = combined_map

    # total number of q's, including 0;
    total_num = q_num * p_num + 1
    num_pts = np.bincount(combined_map.ravel(), minlength=total_num)[1:]

    cqval = np.tile(qval_1d, p_num).reshape(p_num, q_num)
    cqval = np.swapaxes(cqval, 0, 1).flatten()
    cpval = np.tile(pval_1d, q_num)
    # the 0 axis is the q direction and 1st axis is phi direction

    # the roi that has zero points in it;
    invalid_roi = (num_pts == 0)
    cqval[invalid_roi] = np.nan
    cpval[invalid_roi] = np.nan

    cqval = np.expand_dims(cqval, axis=0)
    cpval = np.expand_dims(cpval, axis=0)

    return combined_map, cqval, cpval


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.data_raw = None
        self.shape = None
        self.qmap = None
        self.mask = None
        self.mask_kernel = None
        self.saxs_lin = None
        self.saxs_log = None
        self.saxs_log_min = None
        self.plot_log = True

        self.new_partition = None
        self.meta = None

        self.hdl = pg_hdl
        self.infobar = infobar
        self.extent = None
        self.hdl.scene.sigMouseMoved.connect(self.show_location)
        self.bad_pixel_set = set()
        self.qrings = []

        self.idx_map = {
            0: "scattering",
            1: "scattering * mask",
            2: "mask",
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
            meta['data_name'] = os.path.basename(fname).encode("ascii")
        return meta

    def is_ready(self):
        if self.meta is None or self.data_raw is None:
            return False
        else:
            return True

    def mask_evaluate(self, target, **kwargs):
        msg = self.mask_kernel.evaluate(target, **kwargs)
        # preview the mask
        mask = self.mask_kernel.get_one_mask(target)
        self.data[5][:, :] = mask
        return msg

    def mask_apply(self, target):
        mask = self.mask_kernel.get_one_mask(target)
        self.mask_kernel.enable(target)
        self.mask = np.logical_and(self.mask, mask)
        if target == "mask_qring":
            self.qrings = self.mask_kernel.workers[target].get_qrings()
        self.data[1:] *= self.mask
        if self.plot_log:
            self.data[1][np.logical_not(self.mask)] = self.saxs_log_min
        else:
            self.data[1][np.logical_not(self.mask)] = self.saxs_lin_min

    def get_pts_with_similar_intensity(self, cen=None, radius=50,
                                       variation=50):
        vmin = max(0, int(cen[0] - radius))
        vmax = min(self.shape[0], int(cen[0] + radius))

        hmin = max(0, int(cen[1] - radius))
        hmax = min(self.shape[1], int(cen[1] + radius))
        crop = self.saxs_lin[vmin:vmax, hmin:hmax]
        val = self.saxs_lin[cen]
        idx = np.abs(crop - val) <= variation / 100.0 * val
        pos = np.array(np.nonzero(idx))
        pos[0] += vmin
        pos[1] += hmin
        pos = np.roll(pos, shift=1, axis=0)
        return pos.T

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
                if key in ['datetime', 'energy', 'det_dist', 'pix_dim', 'bcx',
                           'bcy', 'saxs']:
                    continue
                val = np.array(val)
                if val.size == 1:
                    val = val.reshape(1, 1)
                data.create_dataset(key, data=val)
        print('partition map is saved')

    def verify_metadata_hdf(self, file):
        try:
            with h5py.File(file, 'r') as hf:
                if '/measurement/instrument/acquisition' in hf:
                    return True
                else:
                    return False
        except Exception:
            return False

    def get_scattering(self, fname, num_frames=-1, beg_idx=0, **kwargs):
        # seeks directory of existing hdf program
        dirname = os.path.dirname(os.path.realpath(fname))
        files = os.listdir(os.path.dirname(os.path.realpath(fname)))

        for fname in files:
            if fname.endswith('.bin'):
                print("-----------.bin found.-----------")
                bin_file = os.path.join(dirname, fname)
                reader = RigakuReader(bin_file)
                saxs = reader.load()
                return saxs

            # seeks .imm file
            elif fname.endswith('.imm'):
                print("-----------.imm found.-----------")
                imm_file = os.path.join(dirname, fname)
                reader = IMMReader8ID(imm_file)
                saxs = reader.calc_avg_pixel()
                return saxs

            # seeks .h5 file
            elif fname.endswith('.h5') or fname.endswith('.hdf'):
                hdf_file = os.path.join(dirname, fname)
                with h5py.File(hdf_file, 'r') as hf:
                    if '/entry/data/data' in hf:
                        # correct h5 file, contains raw data
                        print("-----------.hdf/.h5 found.-----------")
                        saxs = hdf2saxs(hdf_file, beg_idx=beg_idx,
                                        num_frames=num_frames)
                        return saxs
        return None

    # generate 2d saxs
    def read_data(self, fname=None, **kwargs):
        if not self.verify_metadata_hdf(fname):
            # raise ValueError('check raw file')
            print('the selected file is not a valid metadata hdf file.')
            return

        saxs = self.get_scattering(fname, **kwargs)
        if saxs is None:
            print('cannot read the scattering data from raw data file.')
            return

        self.meta = self.load_meta(fname)

        # keep same
        self.data_raw = np.zeros(shape=(6, *saxs.shape))
        self.mask = np.ones(saxs.shape, dtype=np.bool)

        saxs_nonzero = saxs[saxs > 0]
        # use percentile instead of min to be robust
        self.saxs_lin_min = np.percentile(saxs_nonzero, 1)
        self.saxs_log_min = np.log10(self.saxs_lin_min)

        self.saxs_lin = saxs.astype(np.float32)
        self.min_val = np.min(saxs[saxs > 0])
        self.saxs_log = np.log10(saxs + self.min_val)

        self.shape = self.data_raw[0].shape

        # reset the qrings after data loading
        self.qrings = []
        self.qmap = self.compute_qmap()
        self.mask_kernel = MaskAssemble(self.shape, self.saxs_log,
                                        self.qmap['q'])
        self.extent = self.compute_extent()

        # self.meta['saxs'] = saxs
        self.data_raw[0] = self.saxs_log
        self.data_raw[1] = self.saxs_log * self.mask
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

        phi = np.rad2deg(phi)
        qmap = {
            'phi': phi.astype(np.float32),
            'alpha': alpha.astype(np.float32),
            'q': qr.astype(np.float32),
            'qx': qx.astype(np.float32),
            'qy': qy.astype(np.float32)
        }

        return qmap

    def get_q_value(self, x, y):
        shape = self.qmap['q'].shape
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            return self.qmap['q'][y, x]
        else:
            return None

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
        phi = self.qmap['phi'][row, col]
        val = self.data[self.hdl.currentIndex][row, col]

        # msg = f'{self.idx_map[self.hdl.currentIndex]}: ' + \
        msg = f'[x={col:4d}, y={row:4d}, ' + \
              f'qx={qx:.3e}Å⁻¹, qy={qy:.3e}Å⁻¹, phi={phi:.1f}deg], ' + \
              f'val={val:.03e}'

        self.infobar.clear()
        self.infobar.setText(msg)

        return None

    def show_saxs(self, cmap='jet', log=True, invert=False,
                  plot_center=True, plot_index=0, **kwargs):
        if self.meta is None or self.data_raw is None:
            return
        # self.hdl.reset_limits()
        self.hdl.clear()
        self.data = np.copy(self.data_raw)

        center = (self.meta['bcx'], self.meta['bcy'])

        self.plot_log = log
        if not log:
            self.data[0] = 10 ** self.data[0]

        if invert:
            temp = np.max(self.data[0]) - self.data[0]
            self.data[0] = temp

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

    def apply_drawing(self):
        if self.meta is None or self.data_raw is None:
            return
        if len(self.hdl.roi) <= 0:
            return

        ones = np.ones(self.data[0].shape, dtype=np.bool)
        mask_n = np.zeros_like(ones, dtype=np.bool)
        mask_e = np.zeros_like(mask_n)
        mask_i = np.zeros_like(mask_n)

        remove_roi = []
        for x in self.hdl.roi:
            # get ride of the center plot if it's there
            if isinstance(x, pg.ScatterPlotItem):
                continue
            remove_roi.append(x)

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

        # remove roi; using for loop directly in self.hdl.roi only works if
        # there is only one ROI.
        for x in remove_roi:
            self.remove_roi(x)
        del remove_roi

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i

        return mask_p

    def add_drawing(self, num_edges=None, radius=60, color='r',
                    sl_type='Polygon', width=3, sl_mode='exclusive'):

        shape = self.data.shape
        # cen = (shape[1] // 2, shape[2] // 2)
        cen = (self.meta['bcy'], self.meta['bcx'])
        if sl_mode == 'inclusive':
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        kwargs = {
            'pen': pen,
            'removable': True,
            'hoverPen': pen,
            'handlePen': pen
        }
        if sl_type == 'Ellipse':
            new_roi = pg.EllipseROI([cen[1], cen[0]], [60, 80], **kwargs)
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 1], )
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == 'Circle':
            new_roi = pg.CircleROI([cen[1], cen[0]], [80, 80], **kwargs)

        elif sl_type == 'Polygon':
            if num_edges is None:
                num_edges = np.random.random_integers(6, 10)

            # add angle offset so that the new rois don't overlap with each
            # other
            offset = np.random.random_integers(0, 359)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = radius * np.cos(theta) + cen[1]
            y = radius * np.sin(theta) + cen[0]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, **kwargs)

        elif sl_type == 'Rectangle':
            new_roi = pg.RectROI([cen[1], cen[0]], [30, 150], **kwargs)
            new_roi.addScaleHandle([0, 0], [1, 1])
            # new_roi.addRotateHandle([0, 1], [0.5, 0.5])

        else:
            raise TypeError('type not implemented. %s' % sl_type)

        new_roi.sl_mode = sl_mode
        self.hdl.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(new_roi))
        return

    def remove_roi(self, roi):
        self.hdl.remove_item(roi)

    def get_partition(self, mask=None, num=400, style='linear', mode='q',
                      qrings=None):
        """
        get_partion computes the partition for either qmap or phi-map
        Args:
            mask: 2d array to use as the mask. if none is given, then use the
                default mask
            num: integer, number of points
            style = ['linear', 'logarithmic']. logarithmic only works for q map
            mode = ['q', 'phi']
            qrings: list of tuples (qmin, qmax) or empty list or None, describe
                the start and end q value
        """
        assert mode in ['q', 'phi']
        if mask is None:
            mask = self.mask

        vmap = self.qmap[mode] * mask
        vmap_valid = vmap[self.mask == 1]
        if qrings is None or len(qrings) == 0:
            vmin = np.min(vmap_valid)
            vmax = np.max(vmap_valid)
            qrings = [(vmin, vmax)]

        qspan = []
        qlist = []

        num_rings = len(qrings)
        pnum = num // num_rings     # partial number in each ring 
        if mode == 'phi' or style == 'linear':
            for m in range(num_rings):
                vmin, vmax = qrings[m]
                tmp = np.linspace(vmin, vmax, pnum + 1)
                qspan.append(tmp)
                qlist.append((tmp[1:] + tmp[:-1]) / 2.0)
        elif style == 'logarithmic':
            for m in range(num_rings):
                vmin, vmax = qrings[m]
                qmin = np.log10(vmin)
                qmax = np.log10(vmax)
                tmp = np.logspace(qmin, qmax, pnum + 1)
                qspan.append(tmp)
                qlist.append(np.sqrt(tmp[1:] * tmp[:-1]))

        partition = np.zeros_like(vmap, dtype=np.uint32)
        for m in range(num_rings):
            for n in range(pnum):
                val = qspan[m][n]
                partition[vmap >= val] = n + 1 + (pnum * m)  # overall offset
        qlist = np.hstack(qlist)
        # remove one element starting from the second array so that the 
        # dimension matches
        for n in range(1, num_rings):
            qspan[n] = qspan[n][1:]
        qspan = np.hstack(qspan)
 
        qspan = qspan.reshape(1, -1)
        return qspan, qlist, partition

    def compute_saxs1d(self, cutoff=3.0, episilon=1e-16, **kwargs):
        _, qlist, partition = self.get_partition(**kwargs)
        self.data[5] = partition

        num_q = qlist.size
        saxs1d = np.zeros((5, num_q), dtype=np.float64)

        rows = []
        cols = []

        for n in range(num_q):
            roi = (partition == n + 1)
            if np.sum(roi) == 0:
                continue
            idx = np.nonzero(roi)
            values = self.saxs_lin[idx]

            x0 = np.percentile(values, 5)
            x1 = np.percentile(values, 95)

            val_min, val_max = np.min(values), np.max(values)
            # make sure the edge case works when x0 == x1;
            if x0 == x1:
                x0 = val_min - episilon
                x1 = val_max + episilon

            val_roi = values[(values >= x0) * (values <= x1)]
            avg = np.mean(val_roi)
            std = np.std(val_roi)
            avg_raw = np.mean(values)
            # the median value cannot be used for xpcs at high angles; because
            # the number is likely to be zero

            saxs1d[:, n] = np.array([qlist[n],
                                     avg,                   # avg reference
                                     avg + cutoff * std,    # threshold
                                     val_max,               # max value in roi
                                     avg_raw])              # avg raw

            bad_pixel = np.abs(values - avg) >= cutoff * std
            rows.append(idx[0][bad_pixel])
            cols.append(idx[1][bad_pixel])

        saxs1d = saxs1d[:, saxs1d[1] > 0]

        rows = np.hstack(rows)
        cols = np.hstack(cols)
        zero_loc = np.vstack([rows, cols])

        return saxs1d, zero_loc

    def compute_partition(self, dq_num=10, sq_num=100, style='linear',
                          dp_num=36, sp_num=360):
        if self.meta is None or self.data_raw is None:
            return

        # make the dq number equal to the multiples of num-rings
        num_rings = len(self.qrings)
        if num_rings > 0:
            dq_num = (dq_num + num_rings - 1) // num_rings * num_rings

        # make the static values multiples of the dynamic value
        sq_num = (sq_num + dq_num - 1) // dq_num * dq_num
        sp_num = (sp_num + dp_num - 1) // dp_num * dp_num

        dqspan, dqval_1d, dqmap_partition = \
            self.get_partition(num=dq_num, style=style, mode='q',
                               qrings=self.qrings)

        sqspan, sqval_1d, sqmap_partition = \
            self.get_partition(num=sq_num, style=style, mode='q',
                               qrings=self.qrings)

        dphispan, dphival_1d, dphi_partition = \
            self.get_partition(num=dp_num, style=style, mode='phi')
        sphispan, sphival_1d, sphi_partition = \
            self.get_partition(num=sp_num, style=style, mode='phi')

        dyn_map, dqval, dphival = combine_qp(dq_num, dp_num, dqval_1d,
                                             dphival_1d, dqmap_partition,
                                             dphi_partition)

        sta_map, sqval, sphival = combine_qp(sq_num, sp_num, sqval_1d,
                                             sphival_1d, sqmap_partition,
                                             sphi_partition)

        self.data[3] = dyn_map
        self.data[4] = sta_map
        self.hdl.setCurrentIndex(3)

        partition = {
            'dqval': dqval,
            'sqval': sqval,
            'dynamicMap': dyn_map,
            'staticMap': sta_map,
            'dphival': dphival,
            'sphival': sphival,
            'dynamicQList': np.arange(0, dq_num * dp_num + 1).reshape(1, -1),
            'staticQList': np.arange(0, sq_num * sp_num + 1).reshape(1, -1),
            'dqspan': dqspan,
            'sqspan': sqspan,
            'dphispan': dphispan,
            'sphispan': sphispan,
            'dnophi': dp_num,
            'snophi': sp_num,
            'dnoq': dq_num,
            'snoq': sq_num,
            'x0': np.array([self.meta['bcx']]).reshape(1, 1),
            'y0': np.array([self.meta['bcy']]).reshape(1, 1),
            'xspec': np.array(-1.0).reshape(1, 1),
            'yspec': np.array(-1.0).reshape(1, 1),
        }
        for key in ['snophi', 'snoq', 'dnophi', 'dnoq']:
            partition[key] = np.array(partition[key]).reshape(1, 1)

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


def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'
    sm = SimpleMask()
    sm.read_data(fname)


if __name__ == '__main__':
    test01()
