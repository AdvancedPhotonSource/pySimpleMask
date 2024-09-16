import os
import sys
import json
import logging
import traceback
import numpy as np
import pyqtgraph as pg

from .simplemask_ui import Ui_SimpleMask as Ui
from .simplemask_kernel import SimpleMask
from PySide6.QtWidgets import QFileDialog, QApplication, QMainWindow
from PySide6 import QtWidgets


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
            lambda: self.mask_evaluate('Manual'))
        self.btn_mask_apply.clicked.connect(self.mask_apply)

        # blemish
        self.btn_select_blemish.clicked.connect(self.select_blemish)
        self.btn_apply_blemish.clicked.connect(
            lambda: self.mask_evaluate('Blemish'))

        # File
        self.btn_select_maskfile.clicked.connect(self.select_maskfile)
        self.btn_apply_maskfile.clicked.connect(
            lambda: self.mask_evaluate('File'))

        # draw method / array
        self.btn_mask_draw_add.clicked.connect(self.add_drawing)
        self.btn_mask_draw_evaluate.clicked.connect(
            lambda: self.mask_evaluate('Draw'))

        # binary threshold
        self.btn_mask_threshold_evaluate.clicked.connect(
            lambda: self.mask_evaluate('GlobalThreshold'))

        # btn_mask_outlier_evaluate
        self.btn_mask_outlier_evaluate.clicked.connect(
            lambda: self.mask_evaluate('CircularThreshold'))

        self.cb_qmap_axis0.currentTextChanged.connect(
            lambda: self.update_axis_vrange(0))
        self.cb_qmap_axis1.currentTextChanged.connect(
            lambda: self.update_axis_vrange(1))

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
    
    def handle_manual_mask_with_mouse_click(self, col, row):
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
    
    def handle_manual_partition_with_mouse_click(self, col, row):
        widgets = (self.rb_beg_axis0, self.rb_end_axis0,
                   self.rb_beg_axis1, self.rb_end_axis1) 
        selected_map_idx = 0
        for w in widgets:
            if get_widget_value(w) == True:
                break
            selected_map_idx += 1
        if selected_map_idx == len(widgets):   # none is selected
            return

        vtarget, axis = (('vbeg', 0), ('vend', 0),
                         ('vbeg', 1), ('vend', 1),
                        )[selected_map_idx]

        kwargs0, kwargs1 = self.compute_partition(kwargs_only=True)
        val = self.sm.set_partition_range(col, row, axis, vtarget,
                                          [kwargs0, kwargs1])
        if val is None:
            return

        widget_disp = (self.vbeg_axis0, self.vend_axis0,
                        self.vbeg_axis1, self.vend_axis1)[selected_map_idx]

        put_widget_value(widget_disp, val)
        self.plot_index.setCurrentIndex(0)
        # self.mp1.setLevels(None, None)
        self.plot_index.setCurrentIndex(5)

    def mouse_clicked(self, event):
        if not event.double():
            return

        current_mask_index = self.MaskWidget.currentIndex()
        mouse_point = self.mp1.getView().mapSceneToView(event.pos())
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if not self.mp1.scene.itemsBoundingRect().contains(event.pos()):
            return 

        if current_mask_index == 3:
            self.handle_manual_mask_with_mouse_click(col, row)
            return
        else:
            self.handle_manual_partition_with_mouse_click(col, row)
            return
       
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
            cen_old = (self.db_bcx.value(), self.db_bcy.value())
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

        if target == 'Blemish':
            kwargs = {
                'fname': self.blemish_fname.text(),
                'key': self.blemish_path.text()
            }
        elif target == 'File':
            kwargs = {
                'fname': self.maskfile_fname.text(),
                'key': self.maskfile_path.text()
            }
        elif target == 'Manual':
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
        elif target == 'Draw':
            kwargs = {
                'arr': np.logical_not(self.sm.apply_drawing())
            }
        elif target == 'GlobalThreshold':
            kwargs = {
                'low': self.binary_threshold_low.value(),
                'high': self.binary_threshold_high.value(),
                'scale': ['linear', 'log'][self.binary_scale.currentIndex()]
            }
        elif target == 'CircularThreshold':
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

        self.sm.mask_apply()
        # perform evaluate again so the saxs1d shows the new results;
        if target == 'CircularThreshold':
            self.mask_evaluate(target=target)
        elif target == 'Manual':
            self.mask_list_clear()

        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(1)

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)
        # make the mask and preview binary
        if idx == 2:
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
            for k, v in self.sm.reader.meta.items():
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
        fname, _ = QFileDialog.getOpenFileName(self,
                    'Select raw file hdf',
                    self.work_dir,
                    filter='Supported Formats(*.hdf *.h5 *.hdf5 *.imm *.bin *.tif *.tiff *.fits *.raw, *.bin.*)',
        )

        if fname not in [None, '']:
            self.fname.setText(fname)
        self.work_dir = os.path.dirname(fname)

        return

    def select_blemish(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select blemish file',
                    filter='Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)')
        if fname not in [None, '']:
            self.blemish_fname.setText(fname)

        if fname.endswith('.tif') or fname.endswith('.tiff'):
            self.blemish_path.setDisabled(True)
        else:
            self.blemish_path.setEnabled(True)

        return

    def select_maskfile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select mask file',
                    filter='Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)')
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
            'begin_idx': self.begin_idx.value(),
            'num_frames': self.num_frames.value(),
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

        self.sm.add_drawing(**kwargs)
        return
    
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

    def compute_partition(self, kwargs_only=False):
        if not self.is_ready():
            return

        keys = ('map_name', 'vbeg', 'vend', 'sn', 'dn', 'style')
        axis0 = (self.cb_qmap_axis0, self.vbeg_axis0, self.vend_axis0,
                 self.sn_axis0, self.dn_axis0, self.partition_style_axis0)

        axis1 = (self.cb_qmap_axis1, self.vbeg_axis1, self.vend_axis1,
                 self.sn_axis1, self.dn_axis1, self.partition_style_axis1)

        values0 = [get_widget_value(w) for w in axis0]
        values1 = [get_widget_value(w) for w in axis1]
        kwargs0 = {k:v for k, v in zip(keys, values0)}
        kwargs1 = {k:v for k, v in zip(keys, values1)}

        # make sn a multiple of dn
        for ax, kw in zip([axis0, axis1], [kwargs0, kwargs1]):
            dn_val, sn_val = kw['dn'], kw['sn']
            if sn_val % dn_val != 0:
                kw['sn'] = dn_val * ((sn_val + dn_val - 1) // dn_val)
                ax[keys.index('sn')].setValue(kw['sn'])

        if kwargs_only:
            return kwargs0, kwargs1

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
        method = self.output_method.currentText()
        save_fname, _ = QFileDialog.getSaveFileName(self,
                        caption='Save mask/qmap as')
        self.sm.save_partition(save_fname, method=method)

    def mask_list_load(self):
        if not self.is_ready():
            return

        fname, _ = QFileDialog.getOpenFileName(self, 'Select mask file',
                    filter='Text/Json (*.txt *.csv *.json);;All files(*.*)')
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
                'maskfile_path', 'cb_beamline', 'num_frames', 'begin_idx']
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
