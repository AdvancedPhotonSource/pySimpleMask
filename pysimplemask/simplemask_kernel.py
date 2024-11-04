import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from .area_mask import MaskAssemble
from .find_center import find_center
from .pyqtgraph_mod import LineROI
from .file_reader import read_raw_file
import skimage.io as skio
import logging

pg.setConfigOptions(imageAxisOrder='row-major')


logger = logging.getLogger(__name__)


def create_xy_mesh(mask, qmap, x_num, y_num):
    mask_nz = np.nonzero(mask)
    vrange = (np.min(mask_nz[0]), np.max(mask_nz[0]) + 1)
    hrange = (np.min(mask_nz[1]), np.max(mask_nz[1]) + 1)

    vg, hg = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]),
                         indexing='ij')

    dy = (vrange[1] - vrange[0]) * 1.0 / y_num
    dx = (hrange[1] - hrange[0]) * 1.0 / x_num

    idx_map = np.zeros_like(mask, dtype=np.uint32)

    # -- x --
    for n in range(x_num):
        roi = hg >= hrange[0] + dx * n
        roi = roi * (hg < hrange[0] + dx * (n + 1))
        idx_map[roi] = n + 1

    # -- y --
    for n in range(y_num):
        roi = vg >= vrange[0] + (dy * n)
        roi = roi * (vg < vrange[0] + dy * (n + 1))
        idx_map[roi] += x_num * n

    idx_map = idx_map * mask

    xcorr = np.linspace(hrange[0], hrange[1], x_num + 1) + 0.5
    # qxspan = qmap['qx'][0][xcorr.astype(np.int64)]
    qxspan = xcorr

    xcorr2 = (xcorr[:-1] + xcorr[1:]) / 2.0
    # qxlist = qmap['qx'][0][xcorr2.astype(np.int64)]
    qxlist = xcorr2  # qmap['qx'][0][xcorr2.astype(np.int64)]

    qxlist = np.tile(qxlist, y_num).reshape(y_num, x_num)
    qxlist = np.swapaxes(qxlist, 0, 1)
    qxlist = qxlist.reshape(1, -1)

    ycorr = np.linspace(vrange[0], vrange[1], y_num + 1) + 0.5
    # qyspan = qmap['qy'][:, 0][xcorr.astype(np.int64)]
    qyspan = ycorr

    ycorr2 = (ycorr[:-1] + ycorr[1:]) / 2.0
    # qylist = qmap['qx'][:, 0][ycorr2.astype(np.int64)]
    qylist = ycorr2

    qylist = np.tile(qylist, x_num).reshape(x_num, y_num)
    qylist = qylist.reshape(1, -1)

    return idx_map, qxspan, qyspan, qxlist, qylist


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
        self.corr_roi = None

        self.idx_map = {
            0: "scattering",
            1: "scattering * mask",
            2: "mask",
            3: "dqmap_partition",
            4: "sqmap_partition",
            5: "preview"
        }

    def is_ready(self):
        if self.meta is None or self.data_raw is None:
            return False
        else:
            return True

    def find_center(self):
        if self.saxs_lin is None:
            return
        mask = self.mask
        # center = (self.meta['bcy'], self.meta['bcx'])
        center = find_center(self.saxs_lin, mask=mask, center_guess=None,
                             scale='log')
        return center

    def mask_evaluate(self, target, **kwargs):
        msg = self.mask_kernel.evaluate(target, **kwargs)
        # preview the mask
        mask = self.mask_kernel.get_one_mask(target)
        self.data_raw[5][:, :] = mask
        return msg

    def mask_action(self, action='undo'):
        self.mask_kernel.redo_undo(action=action)
        self.mask_apply()

    def mask_apply(self, target=None):
        # if target is None, apply will return the current mask
        self.mask = self.mask_kernel.apply(target)

        self.data_raw[1] = self.saxs_log * self.mask
        self.data_raw[2] = self.mask

        if target == "mask_qring":
            self.qrings = self.mask_kernel.workers[target].get_qrings()

        if self.plot_log:
            log_min = np.min(self.saxs_log[self.mask > 0])
            self.data_raw[1][np.logical_not(self.mask)] = log_min
        else:
            lin_min = np.min(self.saxs_lin[self.mask > 0])
            self.data_raw[1][np.logical_not(self.mask)] = lin_min

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

            # data.create_dataset('datetime', data=self.meta['datetime'])
            for key, val in self.meta.items():
                if key in ['datetime', 'energy', 'det_dist', 'pix_dim', 'bcx',
                           'bcy', 'saxs']:
                    continue
                val = np.array(val)
                if val.size == 1:
                    val = val.reshape(1, 1)
                data.create_dataset(key, data=val)
        print('partition map is saved')

    # generate 2d saxs
    def read_data(self, fname=None, **kwargs):
        reader = read_raw_file(fname)
        if reader is None:
            logger.error(f'failed to create a dataset handler for {fname}')
            return False

        saxs = reader.get_scattering(**kwargs)
        if saxs is None:
            logger.error('failed to read scattering signal from the dataset.')
            return False

        self.reader = reader
        self.meta = reader.load_meta()

        # keep same
        self.data_raw = np.zeros(shape=(6, *saxs.shape))
        self.mask = np.ones(saxs.shape, dtype=bool)

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
        self.mask_kernel = MaskAssemble(self.shape, self.saxs_log)
        self.mask_kernel.update_qmap(self.qmap)
        self.extent = self.compute_extent()

        # self.meta['saxs'] = saxs
        self.data_raw[0] = self.saxs_log
        self.data_raw[1] = self.saxs_log * self.mask
        self.data_raw[2] = self.mask

        return True 

    def compute_qmap(self):
        k0 = 2 * np.pi / (12.398 / self.meta['energy'])
        v = np.arange(self.shape[0], dtype=np.uint32) - self.meta['bcy']
        h = np.arange(self.shape[1], dtype=np.uint32) - self.meta['bcx']
        vg, hg = np.meshgrid(v, h, indexing='ij')

        r = np.sqrt(vg * vg + hg * hg) * self.meta['pix_dim']
        # phi = np.arctan2(vg, hg)
        # to be compatible with matlab xpcs-gui; phi = 0 starts at 6 clock
        # and it goes clockwise;
        phi = np.arctan2(hg, vg)
        phi[phi < 0] = phi[phi < 0] + np.pi * 2.0
        phi = np.max(phi) - phi     # make it clockwise

        alpha = np.arctan(r / self.meta['det_dist'])
        qr = np.sin(alpha) * k0
        qr = 2 * np.sin(alpha / 2) * k0
        qx = qr * np.cos(phi)
        qy = qr * np.sin(phi)

        phi = np.rad2deg(phi)

        # keep phi and q as np.float64 to keep the precision.
        qmap = {
            'phi': phi,
            'alpha': alpha.astype(np.float32),
            'q': qr,
            'qx': qx.astype(np.float32),
            'qy': qy.astype(np.float32)
        }

        return qmap

    def get_qp_value(self, x, y):
        x = int(x)
        y = int(y)
        shape = self.qmap['q'].shape
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            return self.qmap['q'][y, x], self.qmap['phi'][y, x]
        else:
            return None, None

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
        val = self.data_raw[self.hdl.currentIndex][row, col]

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
        # self.data = np.copy(self.data_raw)
        # print('show_saxs', np.min(self.data[1]))

        center = (self.meta['bcx'], self.meta['bcy'])

        self.plot_log = log
        if not log:
            self.data_raw[0] = self.saxs_lin
        else:
            self.data_raw[0] = self.saxs_log

        # if invert:
        #     temp = np.max(self.data[0]) - self.data[0]
        #     self.data[0] = temp

        self.hdl.setImage(self.data_raw)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[0]], y=[center[1]], symbol='+', size=15)
            self.hdl.add_item(t, label='center')

        self.hdl.setCurrentIndex(plot_index)

        return

    def apply_drawing(self):
        if self.meta is None or self.data_raw is None:
            return

        # ones = np.ones(self.data_raw[0].shape, dtype=np.bool)
        shape = self.data_raw[0].shape
        ones = np.ones((shape[0] + 1, shape[1] + 1), dtype=bool)
        mask_n = np.zeros_like(ones, dtype=bool)
        mask_e = np.zeros_like(mask_n)
        mask_i = np.zeros_like(mask_n)

        for k, x in self.hdl.roi.items():
            if not k.startswith('roi_'):
                continue

            mask_temp = np.zeros_like(ones, dtype=bool)
            # return slice and transfrom
            sl, _ = x.getArraySlice(self.data_raw[1], self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            # sometimes the roi size returned from getArraySlice and
            # getArrayRegion are different;
            nz_idx = np.nonzero(y)

            h_beg = np.min(nz_idx[1])
            h_end = np.max(nz_idx[1]) + 1

            v_beg = np.min(nz_idx[0])
            v_end = np.max(nz_idx[0]) + 1

            sl_v = slice(sl[0].start, sl[0].start + v_end - v_beg)
            sl_h = slice(sl[1].start, sl[1].start + h_end - h_beg)
            mask_temp[sl_v, sl_h] = y[v_beg: v_end, h_beg: h_end]

            if x.sl_mode == 'exclusive':
                mask_e[mask_temp] = 1
            elif x.sl_mode == 'inclusive':
                mask_i[mask_temp] = 1

        self.hdl.remove_rois(filter_str='roi_')

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i
        mask_p = mask_p[:-1, :-1]

        return mask_p

    def add_drawing(self, num_edges=None, radius=60, color='r',
                    sl_type='Polygon', width=3, sl_mode='exclusive',
                    second_point=None, label=None, movable=True):
        # label: label of roi; default is None, which is for roi-draw

        if label is not None and label in self.hdl.roi:
            self.hdl.remove_item(label)

        # cen = (shape[1] // 2, shape[2] // 2)
        cen = (self.meta['bcx'], self.meta['bcy'])
        if sl_mode == 'inclusive':
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        handle_pen = pg.mkPen(color=color, width=width)

        kwargs = {
            'pen': pen,
            'removable': True,
            'hoverPen': pen,
            'handlePen': handle_pen,
            'movable': movable
        }
        if sl_type == 'Ellipse':
            new_roi = pg.EllipseROI(cen, [60, 80], **kwargs)
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 1], )
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == 'Circle':
            if second_point is not None:
                radius = np.sqrt((second_point[1] - cen[1]) ** 2 +
                                 (second_point[0] - cen[0]) ** 2)
            new_roi = pg.CircleROI(pos=[cen[0] - radius, cen[1] - radius],
                                   radius=radius,
                                   **kwargs)

        elif sl_type == 'Polygon':
            if num_edges is None:
                num_edges = np.random.random_integers(6, 10)

            # add angle offset so that the new rois don't overlap with each
            # other
            offset = np.random.random_integers(0, 359)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = radius * np.cos(theta) + cen[0]
            y = radius * np.sin(theta) + cen[1]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, **kwargs)

        elif sl_type == 'Rectangle':
            new_roi = pg.RectROI(cen, [30, 150], **kwargs)
            new_roi.addScaleHandle([0, 0], [1, 1])
            # new_roi.addRotateHandle([0, 1], [0.5, 0.5])

        elif sl_type == 'Line':
            if second_point is None:
                return
            width = kwargs.pop('width', 1)
            new_roi = LineROI(cen, second_point, width,
                              **kwargs)
        else:
            raise TypeError('type not implemented. %s' % sl_type)

        new_roi.sl_mode = sl_mode
        roi_key = self.hdl.add_item(new_roi, label)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(roi_key))
        return new_roi

    def get_qring_values(self):
        result = {}
        cen = (self.meta['bcx'], self.meta['bcy'])

        for key in ['qring_qmin', 'qring_qmax']:
            if key in self.hdl.roi:
                x = tuple(self.hdl.roi[key].state['size'])[0] / 2.0 + cen[0]
                value, _ = self.get_qp_value(x, cen[1])
            else:
                value = None
            result[key] = value

        for key in ['qring_pmin', 'qring_pmax']:
            if key in self.hdl.roi:
                value = self.hdl.roi[key].state['angle']
                value = value - 90
                value = value - np.floor(value / 360.0) * 360.0
            else:
                value = None
            result[key] = value
        return result

    def remove_roi(self, roi_key):
        self.hdl.remove_item(roi_key)

    def get_partition(self, qnum, pnum, qmin=None, qmax=None, pmin=None,
                      pmax=None, style='linear'):

        mask = self.mask
        qmap = self.qmap['q'] * mask
        pmap_org = self.qmap['phi'] * mask

        if qmin is None or qmax is None:
            qmin = np.min(self.qmap['q'][mask > 0])
            qmax = np.max(self.qmap['q'][mask > 0]) + 1e-9  # exclusive
        if pmin is None or pmax is None:
            pmin = 0
            pmax = 360.0 + 1e-9

        assert style in ('linear', 'logarithmic')
        # q
        if style == 'linear':
            qspan = np.linspace(qmin, qmax, qnum + 1)
            qlist = (qspan[1:] + qspan[:-1]) / 2.0
        elif style == 'logarithmic':
            qspan = np.logspace(np.log10(qmin), np.log10(qmax), qnum + 1)
            qlist = np.sqrt(qspan[1:] * qspan[:-1])

        # phi
        pmap = pmap_org.copy()
        # deal with edge case, eg (pmin, pmax) = (350, 10)
        if pmax < pmin:
            pmax += 360.0
            pmap[pmap < pmin] += 360.0

        pspan = np.linspace(pmin, pmax, pnum + 1)
        plist = (pspan[1:] + pspan[:-1]) / 2.0

        qroi = np.logical_and(qmap >= qmin, qmap < qmax)
        proi = np.logical_and(pmap >= pmin, pmap < pmax)
        roi = np.logical_and(qroi, proi)

        qmap = qmap * roi
        pmap = pmap * roi

        partition = np.zeros_like(roi, dtype=np.uint32)

        cqlist = np.tile(qlist, pnum).reshape(pnum, qnum)
        cqlist = np.swapaxes(cqlist, 0, 1)
        cplist = np.tile(plist, qnum).reshape(qnum, pnum)

        idx = 1
        for m in range(qnum):
            qroi = (qmap >= qspan[m]) * (qmap < qspan[m + 1])
            for n in range(pnum):
                proi = (pmap >= pspan[n]) * (pmap < pspan[n + 1])
                comb_roi = qroi * proi
                if np.sum(comb_roi) == 0:
                    cqlist[m, n] = np.nan
                    cplist[m, n] = np.nan
                else:
                    partition[comb_roi] = idx
                    idx += 1

        return (qspan, cqlist, pspan, cplist), partition

    def compute_saxs1d(self, cutoff=3.0, episilon=1e-16, num=180):
        t_dq_span_val, partition = self.get_partition(num, 1,
                                                      None, None, 0, 360, 'linear')
        qlist = t_dq_span_val[1].flatten()

        self.data_raw[5] = partition
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

    def compute_partition(self, mode='q-phi', **kwargs):
        if mode == 'q-phi':
            return self.compute_partition_qphi(**kwargs)
        elif mode == 'xy-mesh':
            return self.compute_partition_xymesh(**kwargs)

    def compute_partition_qphi(self,
                               dq_num=10, sq_num=100, style='linear',
                               dp_num=36, sp_num=360):
        if self.meta is None or self.data_raw is None:
            return

        qrings = self.qrings

        if qrings is None or len(qrings) == 0:
            qrings = [[None, None, 0, 360]]

        # dqspan = []
        # , dqval, dphispan, dphival, dyn_map
        dyn_map = np.zeros_like(self.mask, dtype=np.uint32)
        sta_map = np.zeros_like(self.mask, dtype=np.uint32)

        def combine_span_val(record, new_item):
            for n in range(4):
                record[n].append(new_item[n])
            return record

        dq_record = [[] for n in range(4)]
        sq_record = [[] for n in range(4)]

        for segment in qrings:
            # qmin, qmax, pmin, pmax = segment, 0, 60
            qmin, qmax, pmin, pmax = segment
            t_dq_span_val, tdyn_map = self.get_partition(
                dq_num, dp_num, qmin, qmax, pmin, pmax, style)
            dq_record = combine_span_val(dq_record, t_dq_span_val)

            # offset the nonzero dynamic qmap
            tdyn_map[tdyn_map > 0] += np.max(dyn_map)
            tdyn_idx = np.nonzero(tdyn_map)
            dyn_map[tdyn_idx] = tdyn_map[tdyn_idx]

            # offset the nonzero static qmap
            t_sq_span_val, tsta_map = self.get_partition(
                sq_num, sp_num, qmin, qmax, pmin, pmax, style)
            sq_record = combine_span_val(sq_record, t_sq_span_val)

            tsta_map[tsta_map > 0] += np.max(sta_map)
            sta_map += tsta_map
            tsta_idx = np.nonzero(tsta_map)
            sta_map[tsta_idx] = tsta_map[tsta_idx]

        # (qspan, cqlist, pspan, cplist)
        dqspan = np.hstack(dq_record[0])
        sqspan = np.hstack(sq_record[0])

        dqval = np.hstack(dq_record[1]).reshape(1, -1)
        sqval = np.hstack(sq_record[1]).reshape(1, -1)

        dphispan = np.hstack(dq_record[2])
        sphispan = np.hstack(sq_record[2])

        dphival = np.hstack(dq_record[3]).reshape(1, -1)
        sphival = np.hstack(sq_record[3]).reshape(1, -1)

        # dump result to file;
        self.data_raw[3] = dyn_map
        self.data_raw[4] = sta_map
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

    def compute_partition_xymesh(self, sx_num=8, sy_num=8, dx_num=1, dy_num=1):
        if self.meta is None or self.data_raw is None:
            return

        # make the static numbers multiples of the dynamics numbers
        sy_num = (sy_num + dy_num - 1) // dy_num * dy_num
        sx_num = (sx_num + dx_num - 1) // dx_num * dx_num

        # return idx_map, qxspan, qyspan, qxlist, qylist
        dyn_map, dxspan, dyspan, dxlist, dylist = create_xy_mesh(
            self.mask, self.qmap, dx_num, dy_num)
        sta_map, sxspan, syspan, sxlist, sylist = create_xy_mesh(
            self.mask, self.qmap, sx_num, sy_num)

        self.data_raw[3] = dyn_map
        self.data_raw[4] = sta_map
        self.hdl.setCurrentIndex(3)

        partition = {
            'dqval': dxlist,
            'sqval': sxlist,
            'dynamicMap': dyn_map,
            'staticMap': sta_map,
            'dphival': dylist,
            'sphival': sylist,
            'dynamicQList': np.arange(0, dx_num * dy_num + 1).reshape(1, -1),
            'staticQList': np.arange(0, sx_num * sx_num + 1).reshape(1, -1),
            'dqspan': dxspan,
            'sqspan': sxspan,
            'dphispan': dyspan,
            'sphispan': syspan,
            'dnophi': dy_num,
            'snophi': sy_num,
            'dnoq': dx_num,
            'snoq': sx_num,
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
        self.mask_kernel.update_qmap(self.qmap)

    def get_parameters(self):
        val = []
        for key in ['bcx', 'bcy', 'energy', 'pix_dim', 'det_dist']:
            val.append(self.meta[key])
        return val

    def set_corr_roi(self, roi):
        self.corr_roi = roi

    def perform_correlation(self, angle_list):
        pass


def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'
    sm = SimpleMask()
    sm.read_data(fname)


if __name__ == '__main__':
    test01()
