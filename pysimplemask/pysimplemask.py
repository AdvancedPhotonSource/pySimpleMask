import os
import sys
import json
import logging
import traceback
import numpy as np
import pyqtgraph as pg

from .simplemask_ui import Ui_SimpleMask as Ui
from .simplemask_kernel import SimpleMask
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5 import QtWidgets


home_dir = os.path.join(os.path.expanduser('~'), '.simple-mask')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
log_filename = os.path.join(home_dir, 'simple-mask.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-24s: %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode='a'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


def exception_hook(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook


def text_to_array(pts, dtype=np.int64):
    for symbol in '[](),':
        pts = pts.replace(symbol, ' ')
    pts = pts.split(' ')

    if dtype == np.int64:
        pts = [int(x) for x in pts if x != '']
    elif dtype == np.float64:
        pts = [float(x) for x in pts if x != '']

    pts = np.array(pts).astype(dtype)

    return pts


def get_widget_value(widget, index=False):
    if isinstance(widget, QtWidgets.QComboBox):
        if index:
            return widget.currentIndex()
        else:
            return widget.currentText()
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return widget.value()
    elif isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QLabel)):
        return widget.text()
    elif isinstance(widget, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
        return widget.isChecked()
    else:
        raise TypeError(str(type(widget)) + ' not supported')
    

def put_widget_value(widget, value):
    if isinstance(widget, QtWidgets.QComboBox):
        if isinstance(value, int):
            widget.setCurrentIndex(value)
        else:
            if not isinstance(value, list):
                value = [value]
            widget.addItems(value)
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QLabel)):
        widget.setText(str(value))
    elif isinstance(widget, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
        widget.setEnabled(value)
    else:
        raise TypeError(str(type(widget)) + ' not supported')



class SimpleMaskGUI(QMainWindow, Ui):
    def __init__(self, path=None):

        super(SimpleMaskGUI, self).__init__()

        self.setupUi(self)
        self.btn_load.clicked.connect(self.load)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_compute_qpartition.clicked.connect(self.compute_partition)
        self.btn_select_raw.clicked.connect(self.select_raw)
        # self.btn_select_txt.clicked.connect(self.select_txt)
        self.btn_update_parameters.clicked.connect(self.update_metadata)
        self.btn_swapxy.clicked.connect(
            lambda: self.update_metadata(swapxy=True))

        self.btn_find_center.clicked.connect(self.find_center)

        # need a function for save button -- simple_mask_ui
        self.pushButton.clicked.connect(self.save_mask)

        self.plot_index.currentIndexChanged.connect(self.mp1.setCurrentIndex)

        # reset, redo, undo botton for mask
        self.btn_mask_reset.clicked.connect(lambda: self.mask_action('reset'))
        self.btn_mask_redo.clicked.connect(lambda: self.mask_action('redo'))
        self.btn_mask_undo.clicked.connect(lambda: self.mask_action('undo'))

        # simple mask kernep
        self.sm = SimpleMask(self.mp1, self.infobar)
        self.mp1.sigTimeChanged.connect(self.update_index)
        self.state = 'lock'

        # mask_list
        self.btn_mask_list_load.clicked.connect(self.mask_list_load)
        self.btn_mask_list_clear.clicked.connect(self.mask_list_clear)
        self.btn_mask_list_add.clicked.connect(self.mask_list_add)

        self.btn_mask_list_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_list'))
        self.btn_mask_apply.clicked.connect(self.mask_apply)

        # blemish
        self.btn_select_blemish.clicked.connect(self.select_blemish)
        self.btn_apply_blemish.clicked.connect(
            lambda: self.mask_evaluate('mask_blemish'))

        # mask_file
        self.btn_select_maskfile.clicked.connect(self.select_maskfile)
        self.btn_apply_maskfile.clicked.connect(
            lambda: self.mask_evaluate('mask_file'))

        # draw method / array
        self.btn_mask_draw_add.clicked.connect(self.add_drawing)
        self.btn_mask_draw_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_draw'))

        # binary threshold
        self.btn_mask_threshold_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_threshold'))

        # btn_mask_outlier_evaluate
        self.btn_mask_outlier_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_outlier'))

        # btn_mask_qring
        # self.btn_mask_qring_evaluate.clicked.connect(
        #     lambda: self.mask_evaluate('mask_qring'))
        # self.btn_mask_qring_add1.clicked.connect(
        #     lambda: self.mask_qring_list_add('mouse_click'))
        # self.btn_mask_qring_add2.clicked.connect(
        #     lambda: self.mask_qring_list_add('manual'))
        # self.btn_mask_qring_add3.clicked.connect(
        #     lambda: self.mask_qring_list_add('file'))
        # self.btn_mask_qring_clear.clicked.connect(self.clear_qring_list)
        self.cb_qmap_axis0.currentTextChanged.connect(
            lambda: self.update_axis_vrange(0))
        self.cb_qmap_axis1.currentTextChanged.connect(
            lambda: self.update_axis_vrange(1))

        # tab correlation
        # self.btn_mask_draw_add_corr.clicked.connect(self.add_drawing)
        # self.btn_corr.clicked.connect(self.perform_correlation)
        # self.btn_mask_draw_apply_corr.clicked.connect(self.corr_add_roi)
        # self.angle_n_corr.valueChanged.connect(self.update_corr_angle)

        self.mask_outlier_hdl.setBackground((255, 255, 255))
        self.mp1.scene.sigMouseClicked.connect(self.mouse_clicked)

        self.work_dir = None
        if path is not None:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                self.fname.setText(str(path))
                self.work_dir = os.path.dirname(path)
            elif os.path.isdir(path):
                self.work_dir = path
        else:
            self.work_dir = os.path.expanduser('~')

        self.MaskWidget.setCurrentIndex(0)
        # self.qring_model = QringTableModel(data=[[]])
        # self.tableView.setModel(self.qring_model)
        # header = self.tableView.horizontalHeader()
        # header.setSectionResizeMode(QHeaderView.Stretch)  
        self.setting_fname = os.path.join(home_dir, 'default_setting.json')
        self.lastconfig_fname = os.path.join(home_dir, 'last_config.json')

        # self.tabWidget.setCurrentIndex(0)
        self.load_default_settings()
        self.load_last_config()
        self.show()
        # self.plot_index.addItem('helloworld')

    def load_default_settings(self):
        # copy the default values
        if not os.path.isfile(self.setting_fname):
            config = {
                "window_size_w": 1400,
                "window_size_h": 740
            }
            with open(self.setting_fname, 'w') as f:
                json.dump(config, f, indent=4)

        # the display size might too big for some laptops
        with open(self.setting_fname, 'r') as f:
            config = json.load(f)
            if "window_size_h" in config:
                new_size = (config["window_size_w"], config["window_size_h"])
                logger.info('set mainwindow to size %s', new_size)
                self.resize(*new_size)

        return

    def mouse_clicked(self, event):
        if not event.double():
            return

        # make sure the maskwidget is at manual mode or qring mode;
        current_idx = self.MaskWidget.currentIndex()
        if current_idx not in [3, 5]:
            return

        if not self.mp1.scene.itemsBoundingRect().contains(event.pos()):
            return

        mouse_point = self.mp1.getView().mapSceneToView(event.pos())
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if current_idx == 3:
            # manual mode; select the dead pixel with mouse click
            if not self.mask_list_include.isChecked():
                self.mask_list_add_pts([np.array([col, row])])
            else:
                kwargs = {
                    'radius': self.mask_list_radius.value(),
                    'variation': self.mask_list_variation.value(),
                    'cen': (row, col)
                }
                pos = self.sm.get_pts_with_similar_intensity(**kwargs)
                self.mask_list_add_pts(pos)
        else:
            # qring mode, select qbegin and qend with mouse
            q, p = self.sm.get_qp_value(col, row)
            if q is None or p is None:
                return
            if self.box_qring_qmin.isChecked():
                self.mask_qring_qmin.setValue(q)
                label = 'qring_qmin'
                color = 'k'
            elif self.box_qring_qmax.isChecked():
                self.mask_qring_qmax.setValue(q)
                label = 'qring_qmax'
                color = 'r'
            elif self.box_qring_pmin.isChecked():
                self.mask_qring_pmin.setValue(p)
                label = 'qring_pmin'
                color = 'k'
            elif self.box_qring_pmax.isChecked():
                self.mask_qring_pmax.setValue(p)
                label = 'qring_pmax'
                color = 'r'
            if label.startswith('qring_q'):
                new_roi = self.sm.add_drawing(sl_type='Circle',
                                          second_point=(col, row),
                                          width=1.0,
                                          color=color,
                                          label=label,
                                          movable=False)
            else:
                new_roi = self.sm.add_drawing(sl_type='Line',
                                          second_point=(col, row),
                                          width=0.5,
                                          color=color,
                                          label=label)
            new_roi.sigRegionChanged.connect(self.update_qring_values)
    
    def update_qring_values(self):
        new_state = self.sm.get_qring_values()
        for k, v in new_state.items():
            if v is not None:
                self.__dict__['mask_' + k].setValue(v)
    
    def mask_action(self, action):
        if not self.is_ready():
            return
        self.sm.mask_action(action)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(1)

    def find_center(self):
        if not self.is_ready():
            return
        try:
            self.btn_find_center.setText('Finding Center ...')
            self.centralwidget.repaint()
            center = self.sm.find_center()
        except Exception:
            traceback.print_exc()
            self.statusbar.showMessage('Failed to find center. Abort', 2000)
        else:
            cen_old = (
                self.db_bcx.value(), self.db_bcy.value()
            )
            self.db_bcx.setValue(center[1])
            self.db_bcy.setValue(center[0])
            self.update_metadata(direction='gui->file')
            cen_new = (round(center[1], 4), round(center[0], 4))
            logger.info(f'found center: {cen_old} --> {cen_new}')
        finally:
            self.btn_find_center.setText('Find Center')


    def mask_evaluate(self, target=None):
        if target is None or not self.is_ready():
            return

        if target == 'mask_blemish':
            kwargs = {
                'fname': self.blemish_fname.text(),
                'key': self.blemish_path.text()
            }
        elif target == 'mask_file':
            kwargs = {
                'fname': self.maskfile_fname.text(),
                'key': self.maskfile_path.text()
            }
        elif target == 'mask_list':
            num_row = self.mask_list_xylist.count()
            val = [str(self.mask_list_xylist.item(i).text())
                   for i in range(num_row)]
            val = ' '.join(val)
            xy = text_to_array(val)
            xy = xy[0: xy.size // 2 * 2].reshape(-1, 2).T
            xy = np.roll(xy, shift=1, axis=0)
            kwargs = {
                'zero_loc': xy
            }
        elif target == 'mask_draw':
            kwargs = {
                'arr': np.logical_not(self.sm.apply_drawing())
            }
        elif target == 'mask_threshold':
            kwargs = {
                'low': self.binary_threshold_low.value(),
                'high': self.binary_threshold_high.value(),
                'scale': ['linear', 'log'][self.binary_scale.currentIndex()]
            }
        elif target == 'mask_outlier':
            num = self.outlier_num_roi.value()
            cutoff = self.outlier_cutoff.value()
            # iterations = self.outlier_iterations.value()
            saxs1d, zero_loc = self.sm.compute_saxs1d(num=num, cutoff=cutoff)

            self.mask_outlier_hdl.clear()
            p = self.mask_outlier_hdl
            p.addLegend()
            p.plot(saxs1d[0], saxs1d[1], name='average_ref',
                   pen=pg.mkPen(color='g', width=2))
            p.plot(saxs1d[0], saxs1d[4], name='average_raw',
                   pen=pg.mkPen(color='m', width=2))
            p.plot(saxs1d[0], saxs1d[2], name='cutoff',
                   pen=pg.mkPen(color='b', width=2))
            p.plot(saxs1d[0], saxs1d[3], name='maximum value',
                   pen=pg.mkPen(color='r', width=2))
            p.setLabel('bottom', 'q (Å⁻¹)')
            p.setLabel('left', 'Intensity (a.u.)')
            p.setLogMode(y=True)
            kwargs = {'zero_loc': zero_loc}
        elif target == 'mask_qring':
            if self.qring_model.data == [[]]:
                return
            else:
                data = self.qring_model.data.copy()
                # self.qring_model.data = [[]]
            kwargs = {'qrings': data}

        msg = self.sm.mask_evaluate(target, **kwargs)
        self.statusbar.showMessage(msg, 10000)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(5)
        return

    def mask_apply(self):
        idx = self.MaskWidget.currentIndex()
        target = str(self.MaskWidget.tabText(idx))
        if not self.is_ready():
            return

        self.sm.mask_apply(target)
        # perform evaluate again so the saxs1d shows the new results;
        if target == 'Threshold':
            self.mask_evaluate(target=target)
        elif target == 'Manual':
            self.mask_list_clear()
        elif target == 'qring':
            self.clear_qring_list()

        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(1)

    # def mask_qring_list_add(self, method='manual'):
    #     if method == 'mouse_click':
    #         tmp_kwargs = {
    #             "qmin": self.mask_qring_qmin.value(),
    #             "qmax": self.mask_qring_qmax.value(),
    #             "pmin": self.mask_qring_pmin.value(),
    #             "pmax": self.mask_qring_pmax.value(),
    #             "qnum": self.mask_qring_num.value(),
    #             "flag_const_width": self.mask_qring_constwidth.isChecked(),
    #         }
    #         qrings = create_qring(**tmp_kwargs)
    #     elif method == 'manual':
    #         pts = self.mask_qring_input.text()
    #         self.mask_qring_input.clear()
    #         if len(pts) < 1:
    #             self.statusbar.showMessage('Input list is invalid.', 500)
    #             return
    #         try:
    #             xy = text_to_array(pts, dtype=np.float64)
    #             # to a 2d list
    #             qrings = xy[0: xy.size // 4 * 4].reshape(-1, 4).tolist()
    #         except Exception:
    #             self.statusbar.showMessage('Input list is invalid.', 500)
    #             return

    #     elif method == 'file':
    #         fname = QFileDialog.getOpenFileName(self, 'Select qring file',
    #                 filter='Text/Json (*.txt *.csv *.json);;All files(*.*)')[0]
    #         if fname in ['', None]:
    #             return
    #     
    #         if fname.endswith('.json'):
    #             with open(fname, 'r') as f:
    #                 x = json.load(f)
    #             xy = []
    #             for _, v in x.items():
    #                 xy.append(v)
    #             xy = np.array(xy)

    #         elif fname.endswith('.txt') or fname.endswith('.csv'):
    #             try:
    #                 xy = np.loadtxt(fname, delimiter=',')
    #             except ValueError:
    #                 xy = np.loadtxt(fname)
    #             except Exception:
    #                 self.statusbar.showMessage(
    #                     'only support csv and space separated file', 500)
    #                 return
    #         qrings = xy[0: xy.size // 4 * 4].reshape(-1, 4).tolist()

    #     if self.qring_model.data == [[]]:
    #         self.qring_model.data = qrings
    #     else:
    #         self.qring_model.data.extend(qrings)
    #     # update tableview
    #     self.tableView.setModel(None)
    #     self.tableView.setModel(self.qring_model)
    #     return
    
    def clear_qring_list(self):
        self.tableView.setModel(None)
        self.qring_model.data = [[]]
        self.tableView.setModel(self.qring_model)
        self.sm.hdl.remove_rois(filter_str='qring_')

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)
        # make the mask and preview binary
        if idx in [2, 5]:
            self.mp1.setLevels(0, 1)

    def is_ready(self):
        if not self.sm.is_ready():
            self.statusbar.showMessage('No scattering image is loaded.', 500)
            return False
        return True

    def update_metadata(self, swapxy=False, direction='gui->file'):
        if not self.is_ready():
            return
        
        sg_idx = self.metaTab.currentIndex()

        if sg_idx == 0:
            pv = {
                'energy': self.db_energy, 
                'det_dist': self.db_det_dist, 
                'pix_dim': self.db_pix_dim,
                'bcx': self.db_bcx, 
                'bcy': self.db_bcy,
                'shape': self.le_shape,
                }
        elif sg_idx == 1:
            pv = {
                'energy': self.db_energy_1, 
                'det_dist': self.db_det_dist_1, 
                'pix_dim': self.db_pix_dim_1,
                'bcx': self.db_bcx_1, 
                'bcy': self.db_bcy_1,
                'shape': self.le_shape_1,
                'alpha_i': self.alpha_i_1,
                }
        else:
            logger.error(f'{sg_idx=} not implemented')
            return

        if direction == 'gui->file':
            values = {} 
            pv.pop('shape', None)
            for k, v in pv.items():
                values[k] = v.value()
            if swapxy:
                x, y = values['bcx'], values['bcy']
                values['bcx'], values['bcy'] = y, x 
                pv['bcx'].setValue(y)
                pv['bcy'].setValue(x)
            self.sm.update_parameters(values)

        elif direction == 'file->gui':
            for k, v in self.sm.meta.items():
                if k not in pv.keys():
                    continue
                if k == 'shape':
                    pv[k].setText(str(v))
                else:
                    pv[k].setValue(v)

        self.update_qmap_info()
        self.groupBox.repaint()
        self.plot()

    def select_raw(self):
        fname = QFileDialog.getOpenFileName(self,
                    caption='Select raw file hdf',
                    filter='Supported Formats(*.hdf *.h5 *.hdf5 *.imm *.bin *.tif *.tiff *.fits *.raw, *.bin.*)',
                    directory=self.work_dir)[0]

        if fname not in [None, '']:
            self.fname.setText(fname)
        self.work_dir = os.path.dirname(fname)

        return

    def select_blemish(self):
        fname = QFileDialog.getOpenFileName(self, 'Select blemish file',
                    filter='Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)')[0]
        if fname not in [None, '']:
            self.blemish_fname.setText(fname)

        if fname.endswith('.tif') or fname.endswith('.tiff'):
            self.blemish_path.setDisabled(True)
        else:
            self.blemish_path.setEnabled(True)

        return

    def select_maskfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Select mask file',
                    filter='Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)')[0]
        # fname = "../tests/data/triangle_mask/mask_lambda_test.h5"
        if fname not in [None, '']:
            self.maskfile_fname.setText(fname)
        if fname.endswith('.tif') or fname.endswith('.tiff'):
            self.maskfile_path.setDisabled(True)
        else:
            self.maskfile_path.setEnabled(True)
        return

    def load(self):
        # self.fname.setText('/mnt/c/Users/mqichu/Documents/local_dev/pysimplemask/tests/data/E0135_La0p65_L2_013C_att04_Rq0_00001/E0135_La0p65_L2_013C_att04_Rq0_00001_0001-100000.hdf')
        if not os.path.isfile(self.fname.text()):
            self.select_raw()
        if not os.path.isfile(self.fname.text()):
            self.statusbar.showMessage('select a valid file')
            return

        self.btn_load.setText('loading...')
        self.statusbar.showMessage('loading data...', 120000)
        self.centralwidget.repaint()

        fname = self.fname.text()
        kwargs = {
            'begin_idx': self.spinBox_3.value(),
            'num_frames': self.spinBox_4.value(),
            'beamline': str(self.cb_beamline.currentText())
        }
        default_sg_idx = {'APS-8ID-I': 0,
                          'APS-9ID-C': 1,
                          'APS-12ID-B': 2}[kwargs['beamline']]

        self.metaTab.setCurrentIndex(default_sg_idx)

        if not self.sm.read_data(fname, **kwargs):
            return

        self.update_metadata(direction='file->gui')
        self.statusbar.showMessage('data is loaded', 500)
        self.btn_load.setText('load data')
        self.btn_load.repaint()

    def plot(self):
        kwargs = {
            'cmap': self.plot_cmap.currentText(),
            'log': self.plot_log.isChecked(),
            'invert': self.plot_invert.isChecked(),
            # 'rotate': self.plot_rotate.isChecked(),
            'plot_center': self.plot_center.isChecked(),
        }
        self.sm.show_saxs(**kwargs)
        self.plot_index.setCurrentIndex(1)

    def add_drawing(self):
        if not self.is_ready():
            return
        if self.MaskWidget.currentIndex() == 1:
            color = self.cb_selector_color.currentText()
            kwargs = {
                'color': color,
                'sl_type': self.cb_selector_type.currentText(),
                'sl_mode': self.cb_selector_mode.currentText(),
                'width': self.plot_width.value()
            }
        # elif self.MaskWidget.currentIndex() == 6:
        #     color = ('g', 'y', 'b', 'r', 'c', 'm', 'k', 'w')[
        #         self.cb_selector_color_corr.currentIndex()]
        #     kwargs = {
        #         'color': color,
        #         'sl_type': self.cb_selector_type_corr.currentText(),
        #         'sl_mode': 'inclusive',
        #         'width': self.plot_width_corr.value()
        #     }
        # else:
        #     return
        self.sm.add_drawing(**kwargs)
        return
    
    # self.btn_mask_draw_apply_corr.clicked.connect(self.corr_add_roi)
    def corr_add_roi(self):
        roi = self.sm.apply_drawing()
        self.sm.set_corr_roi(roi)
        return
    
    def update_corr_angle(self):
        angle_deg = 360.0 / self.angle_n_corr.value()
        self.angle_corr_text.setText(f"{angle_deg:.2f} (deg)")
    
    def perform_correlation(self):
        angle = 2 * np.pi / self.angle_n_corr.value()
        self.sm.perform_correlation(angle)

    def update_axis_vrange(self, index=0):
        if index == 0:
            target = self.cb_qmap_axis0.currentText()
            display = [self.vbeg_axis0, self.vend_axis0, self.unit_axis0,
                       self.partition_style_axis0]
        elif index == 1:
            target = self.cb_qmap_axis1.currentText()
            display = [self.vbeg_axis1, self.vend_axis1, self.unit_axis1,
                       self.partition_style_axis1]

        display[-1].clear()
        vrange, unit = self.sm.get_qmap_vrange(target)
        options = ['linear']
        if vrange[0] > 0:
            options.append('logarithmic')
        for widget, value in zip(display, (*vrange, unit, options)):
            print(value)
            put_widget_value(widget, value)
    
    def update_qmap_info(self, default_qmap=('q', 'phi')):
        axis_list = (self.cb_qmap_axis0, self.cb_qmap_axis1)
        for hdl in axis_list:
            hdl.clear()
            hdl.addItem('none')

        while self.plot_index.count() > 6:
            self.plot_index.removeItem(6)

        for n, (key, val) in enumerate(self.sm.qmap.items()):
            self.sm.data_raw[n + 6] = val
            self.plot_index.addItem(f'qmap: {key}')
            for hdl in axis_list:
                hdl.addItem(f'{key}')
        for axis, name in zip(axis_list, default_qmap):
            axis.setCurrentText(name)

    def compute_partition(self):
        if not self.is_ready():
            return

        keys = ('xmap', 'vbeg', 'vend', 'sn', 'dn', 'style')
        axis0 = (self.cb_qmap_axis0, self.vbeg_axis0, self.vend_axis0,
                 self.sn_axis0, self.dn_axis0, self.partition_style_axis0)

        axis1 = (self.cb_qmap_axis1, self.vbeg_axis1, self.vend_axis1,
                 self.sn_axis1, self.dn_axis1, self.partition_style_axis1)

        values0 = [get_widget_value(w) for w in axis0]
        values1 = [get_widget_value(w) for w in axis1]
        kwargs0 = {k:v for k, v in zip(keys, values0)}
        kwargs1 = {k:v for k, v in zip(keys, values1)}

        self.btn_compute_qpartition.setDisabled(True)
        self.statusbar.showMessage('Computing partition ... ', 10000)
        self.centralwidget.repaint()

        try:
            self.sm.compute_partition(kwargs0, kwargs1)
        except Exception:
            traceback.print_exception()

        self.statusbar.showMessage('New partition is generated.', 1000)
        self.btn_compute_qpartition.setEnabled(True)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(3)
        # self.btn_compute_qpartition.setStyleSheet("background-color: green")

    def save_mask(self):
        if not self.is_ready():
            return

        if self.sm.new_partition is None:
            self.compute_partition()
        save_fname = QFileDialog.getSaveFileName(
            self, caption='Save mask/qmap as', filter='HDF (*.h5)')[0]
        self.sm.save_partition(save_fname)

    def mask_list_load(self):
        if not self.is_ready():
            return

        fname = QFileDialog.getOpenFileName(self, 'Select mask file',
                    filter='Text/Json (*.txt *.csv *.json);;All files(*.*)')[0]
        if fname in ['', None]:
            return
        
        if fname.endswith('.json'):
            with open(fname, 'r') as f:
                x = json.load(f)['Bad pixels']
            xy = []
            for t in x:
                xy.append(t['Pixel'])
            xy = np.array(xy)
        elif fname.endswith('.txt') or fname.endswith('.csv'):
            try:
                xy = np.loadtxt(fname, delimiter=',')
            except ValueError:
                xy = np.loadtxt(fname)
            except Exception:
                self.statusbar.showMessage(
                    'only support csv and space separated file', 500)
                return

        if self.mask_list_rowcol.isChecked():
            xy = np.roll(xy, shift=1, axis=1)
        if self.mask_list_1based.isChecked():
            xy = xy - 1

        xy = xy.astype(np.int64)
        self.mask_list_add_pts(xy)

    def mask_list_add(self):
        pts = self.mask_list_input.text()
        self.mask_list_input.clear()
        if len(pts) < 1:
            self.statusbar.showMessage('Input list is almost empty.', 500)
            return

        xy = text_to_array(pts)
        xy = xy[0: xy.size // 2 * 2].reshape(-1, 2)
        self.mask_list_add_pts(xy)

    def mask_list_add_pts(self, pts):
        for xy in pts:
            xy_str = str(xy)
            if xy_str not in self.sm.bad_pixel_set:
                self.mask_list_xylist.addItem(xy_str)
                self.sm.bad_pixel_set.add(xy_str)
        self.groupBox_11.setTitle('xy list: %d' % len(self.sm.bad_pixel_set))

    def mask_list_clear(self):
        self.sm.bad_pixel_set.clear()
        self.mask_list_xylist.clear()
        self.groupBox_11.setTitle('xy list')

    def load_last_config(self, ):
        if not os.path.isfile(self.lastconfig_fname):
            logger.info('no configure file found. skip')
            return

        try:
            with open(self.lastconfig_fname, 'r') as fhdl:
                logger.info('load the last configure.')
                config = json.load(fhdl)
                for key, val in config.items():
                    put_widget_value(self.__dict__[key], val)
        except Exception:
            os.remove(self.lastconfig_fname)
            logger.info('configuration file damaged. delete it now')
        return

    def closeEvent(self, e) -> None:
        keys = ['blemish_fname', 'blemish_path', 'maskfile_fname',
                'maskfile_path', 'cb_beamline']
        config = {}
        for key in keys:
            config[key] = get_widget_value(self.__dict__[key], index=True)

        with open(self.lastconfig_fname, 'w') as fhdl:
            json.dump(config, fhdl)


def run(path=None):
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = SimpleMaskGUI(path)
    app.exec_()


if __name__ == '__main__':
    run()
