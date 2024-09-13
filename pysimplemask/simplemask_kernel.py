import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from .area_mask import MaskAssemble
from .find_center import find_center
from .pyqtgraph_mod import LineROI
from .reader import APS9IDCReader, APS8IDIReader 
import logging
from .scattering_geometry import get_scattering_geometry
from .qpartition_utilis import create_partitions, create_single_partition
from .qmc_saver import save_qmc
pg.setConfigOptions(imageAxisOrder='row-major')

logger = logging.getLogger(__name__)


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.reader = None
        self.data_raw = None
        self.shape = None
        self.qmap = None
        self.mask = None
        self.mask_kernel = None
        self.plot_log = True

        self.new_partition = None

        self.hdl = pg_hdl
        self.infobar = infobar
        self.extent = None
        self.hdl.scene.sigMouseMoved.connect(self.show_location)
        self.bad_pixel_set = set()
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
        return self.read_data is not None

    def find_center(self):
        if not self.is_ready():
            return

        mask = self.mask
        center = find_center(self.reader.saxs_lin, mask=mask, center_guess=None,
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

    def mask_apply(self):
        self.mask = self.mask_kernel.apply()
        saxs_with_mask = self.reader.get_scattering_with_mask(self.mask,
                                                              self.plot_log)
        self.data_raw[1] = saxs_with_mask 
        self.data_raw[2] = self.mask

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

    def save_partition(self, save_fname, method='nexus'):
        # if no partition is computed yet
        if self.new_partition is None:
            return
        
        partiton_info = self.new_partition.copy()
        partiton_info['mask'] = self.mask.astype(bool)
        save_qmc(save_fname, partiton_info, method=method)
        logger.info('partition map is saved')

    # generate 2d saxs
    def read_data(self, fname=None, beamline='APS-8ID-I', **kwargs):
        if beamline == 'APS-8ID-I':
            self.reader = APS8IDIReader(fname, **kwargs)
        elif beamline == 'APS-9ID-C':
            self.reader = APS9IDCReader(fname, **kwargs)
        else:
            logger.error(f'failed to create a dataset handler for {fname}')
            return False

        shape = self.reader.shape
        self.shape = shape
        self.data_raw = np.zeros(shape=(16, *shape))
        self.mask = np.ones(shape, dtype=bool)

        self.qmap, self.qmap_unit = self.compute_qmap()

        self.mask_kernel = MaskAssemble(shape, self.reader.saxs_log)
        self.mask_kernel.update_qmap(self.qmap)

        self.data_raw[0] = self.reader.saxs_log
        self.data_raw[1] = self.reader.saxs_log * self.mask
        self.data_raw[2] = self.mask
        return True 

    def compute_qmap(self):
        sg_type = self.reader.meta['sg_type']
        qmap, qmap_unit = get_scattering_geometry(sg_type, self.reader.meta)
        for n, (_, val) in enumerate(qmap.items()):
            self.data_raw[n + 6] = val
        return qmap, qmap_unit
    
    def get_qmap_vrange(self, target='q'):
        if self.qmap is None or target not in self.qmap.keys():
            return (0, 1), 'unit'
        else:
            xmap = self.qmap[target][self.mask == 1]
            return (np.nanmin(xmap), np.nanmax(xmap)), self.qmap_unit[target]

    def set_partition_range(self, x, y, axis, vtarget, pkwargs_list):
        map_name = pkwargs_list[axis]['map_name']
        val = self.qmap[map_name][int(y), int(x)]
        new_val_dict = {vtarget: val}
        pkwargs_list[axis].update(new_val_dict)

        mask = np.copy(self.mask)
        for kwargs in pkwargs_list:
            mask_t = self.get_mask_with_partition(**kwargs)
            mask *= mask_t
        data = self.reader.get_scattering_with_mask(mask, log_style=True)
        self.data_raw[5] = data
        return val

    def get_mask_with_partition(self, map_name='q', vbeg=None, vend=None,
                                **kwargs):
        if map_name == 'none':
            return 1
        xmap = self.qmap[map_name]
        vbeg = np.nanmin(xmap[self.mask == 1]) if vbeg is None else vbeg
        vend = np.nanmax(xmap[self.mask == 1]) if vend is None else vend
        mask = (xmap >= vbeg) * (xmap <= vend)
        return mask

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
        if not self.is_ready():
            return

        self.hdl.clear()

        center = self.reader.get_center()
        self.plot_log = log
        if not log:
            self.data_raw[0] = self.reader.saxs_lin
        else:
            self.data_raw[0] = self.reader.saxs_log

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
        if not self.is_ready():
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
        if label is not None and label in self.hdl.roi:
            self.hdl.remove_item(label)

        cen = self.reader.get_center()
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

    def remove_roi(self, roi_key):
        self.hdl.remove_item(roi_key)

    def compute_saxs1d(self, cutoff=3.0, episilon=1e-16, num=180):
        p_dict = create_single_partition(self.qmap['q'], self.mask, None, None,
                                         n_bins=num, style='linear')
        # t_dq_span_val, partition = self.get_partition(num, 1,
        #                                    None, None, 0, 360, 'linear')
        qlist = p_dict['vlist']
        self.data_raw[5] = p_dict['partition']
        num_q = qlist.size
        saxs1d = np.zeros((5, num_q), dtype=np.float64)

        rows = []
        cols = []

        for n in range(num_q):
            if p_dict['counts'][n] == 0:
                continue
            roi = (p_dict['partition'] == n + 1)
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

    def compute_partition(self, kwargs0, kwargs1):
        map_name = kwargs0['map_name']
        kwargs0['xmap'] = self.qmap[map_name]
        kwargs0['mask'] = self.mask

        map_name = kwargs1['map_name']
        if map_name == 'none':
            kwargs1 = None
        else:
            kwargs1['xmap'] = self.qmap[map_name]
            kwargs1['mask'] = self.mask

        static_p, dynamic_p = create_partitions(kwargs0, kwargs1)

        self.data_raw[3] = dynamic_p['partition']
        self.data_raw[4] = static_p['partition'] 
        self.hdl.setCurrentIndex(3)

        partition = {
            'static_q_list': static_p['vlist'],
            'static_roi_map': static_p['partition'],
            'static_counts': static_p['counts'],
            'dynamic_q_list': dynamic_p['vlist'],
            'dynamic_roi_map': dynamic_p['partition'],
            'dynamic_counts': dynamic_p['counts'],
            'map_name': static_p['map_name'],
        }
        self.new_partition = partition
        return partition

    def update_parameters(self, val_dict):
        self.reader.meta.update(val_dict)
        self.qmap, self.qmap_unit = self.compute_qmap()
        self.mask_kernel.update_qmap(self.qmap)



