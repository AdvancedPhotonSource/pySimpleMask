# from PyQt5 import QtCore
from simple_mask_ui import Ui_SimpleMask as Ui
from simple_mask_kernel import SimpleMask
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pyqtgraph as pg

import os
import sys
import logging


home_dir = os.path.join(os.path.expanduser('~'), '.simple-mask')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
log_filename = os.path.join(home_dir, 'viewer.log')
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


def text_to_array(pts):
    for symbol in '[](),':
        pts = pts.replace(symbol, ' ')
    pts = pts.split(' ')
    pts = [int(x) for x in pts if x != '']
    pts = np.array(pts).astype(np.int64)

    return pts


class SimpleMaskGUI(QtWidgets.QMainWindow, Ui):
    def __init__(self, path=None):

        super(SimpleMaskGUI, self).__init__()

        self.setupUi(self)
        self.btn_load.clicked.connect(self.load)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_compute_qpartition.clicked.connect(self.compute_partition)
        self.btn_select_raw.clicked.connect(self.select_raw)
        # self.btn_select_txt.clicked.connect(self.select_txt)
        self.btn_update_parameters.clicked.connect(self.update_parameters)

        # need a function for save button -- simple_mask_ui
        self.pushButton.clicked.connect(self.save_mask)

        self.plot_index.currentIndexChanged.connect(self.mp1.setCurrentIndex)

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
        self.btn_mask_list_apply.clicked.connect(
            lambda: self.mask_apply('mask_list'))
    
        # blemish
        self.btn_select_blemish.clicked.connect(self.select_blemish)
        self.btn_apply_blemish.clicked.connect(
            lambda: self.mask_evaluate('mask_blemish'))
        self.btn_mask_blemish_apply.clicked.connect(
            lambda: self.mask_apply('mask_blemish'))

        # mask_file
        self.btn_select_maskfile.clicked.connect(self.select_maskfile)
        self.btn_apply_maskfile.clicked.connect(
            lambda: self.mask_evaluate('mask_file'))
        self.btn_mask_file_apply.clicked.connect(
            lambda: self.mask_apply('mask_file'))

        # draw method / array
        self.btn_mask_draw_add.clicked.connect(self.add_drawing)
        self.btn_mask_draw_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_draw'))
        self.btn_mask_draw_apply.clicked.connect(
            lambda: self.mask_apply('mask_draw'))

        # binary threshold
        self.btn_mask_threshold_evaluate.clicked.connect(
            lambda: self.mask_evaluate('mask_threshold'))
        self.btn_mask_threshold_apply.clicked.connect(
            lambda: self.mask_apply('mask_threshold'))

        self.show()

    def mask_evaluate(self, target=None):
        if target is None:
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
        self.sm.mask_evaluate(target, **kwargs)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(5)
        return
 
    def mask_apply(self, target):
        self.sm.mask_apply(target)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(2)

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)

    def update_parameters(self):
        pvs = (self.db_cenx, self.db_ceny, self.db_energy, self.db_pix_dim,
               self.db_det_dist)
        values = []
        for pv in pvs:
            values.append(pv.value())
        self.sm.update_parameters(values)
        self.groupBox.repaint()
        self.plot()

    def select_raw(self):
        # fname = QFileDialog.getOpenFileName(self, 'Select raw file hdf')[0]
        fname = "../tests/data/H432_OH_100_025C_att05_001/H432_OH_100_025C_att05_001_0001-1000.hdf"
        if fname not in [None, '']:
            self.fname.setText(fname)
        return

    def select_blemish(self):
        fname = QFileDialog.getOpenFileName(self, 'Select blemish file')[0]
        if fname not in [None, '']:
            self.blemish_fname.setText(fname)

        if fname.endswith('.tif') or fname.endswith('.tiff'):
            self.blemish_path.setDisabled(True)
        else:
            self.blemish_path.setEnabled(True)

        return

    def select_maskfile(self):
        # fname = QFileDialog.getOpenFileName(self, 'Select mask file')[0]
        fname = "../tests/data/triangle_mask/mask_lambda_test.h5"
        if fname not in [None, '']:
            self.maskfile_fname.setText(fname)
        if fname.endswith('.tif') or fname.endswith('.tiff'):
            self.maskfile_path.setDisabled(True)
        else:
            self.maskfile_path.setEnabled(True)
        return

    def load(self):
        while not os.path.isfile(self.fname.text()):
            self.select_raw()
        fname = self.fname.text()
        self.sm.read_data(fname)

        self.db_cenx.setValue(self.sm.meta['bcx'])
        self.db_ceny.setValue(self.sm.meta['bcy'])
        self.db_energy.setValue(self.sm.meta['energy'])
        self.db_pix_dim.setValue(self.sm.meta['pix_dim'])
        self.db_det_dist.setValue(self.sm.meta['det_dist'])
        self.le_shape.setText(str(self.sm.shape))
        self.groupBox.repaint()
        self.plot()

    def plot(self):
        kwargs = {
            'cmap': self.plot_cmap.currentText(),
            'log': self.plot_log.isChecked(),
            'invert': self.plot_invert.isChecked(),
            # 'rotate': self.plot_rotate.isChecked(),
            'plot_center': self.plot_center.isChecked(),
        }
        self.sm.show_saxs(**kwargs)
        self.plot_index.setCurrentIndex(0)

    def add_drawing(self):
        color = ('g', 'y', 'b', 'r', 'c', 'm', 'k', 'w')[
            self.cb_selector_color.currentIndex()]
        kwargs = {
            'color': color,
            'sl_type': self.cb_selector_type.currentText(),
            'sl_mode': self.cb_selector_mode.currentText(),
            'width': self.plot_width.value()
        }
        self.sm.add_drawing(**kwargs)
        return

    def compute_partition(self):
        # mask = self.apply_roi()
        # self.test_output("test", mask)
        kwargs = {
            'sq_num': self.sb_sqnum.value(),
            'dq_num': self.sb_dqnum.value(),
            'sp_num': self.sb_spnum.value(),
            'dp_num': self.sb_dpnum.value(),
            'style': self.partition_style.currentText(),
        }
        self.sm.compute_partition(**kwargs)
        self.plot_index.setCurrentIndex(3)

    def save_mask(self):
        if self.sm.new_partition is None:
            self.compute_partition()
        save_fname = QFileDialog.getSaveFileName(
            self, caption='Save mask/qmap as')[0]
        self.sm.save_partition(save_fname)

    def mask_list_load(self):
        # fname = QFileDialog.getOpenFileName(self, 'Select mask file')[0]
        fname = 'mask_list.txt'
        if fname in ['', None]:
            return

        try:
            xy = np.loadtxt(fname, delimiter=',')
        except ValueError:
            xy = np.loadtxt(fname)
        except Exception:
            print('only support csv and space separated file')
            return

        if self.mask_list_rowcol.isChecked():
            xy = np.roll(xy, shift=1, axis=1)
        if self.mask_list_1based.isChecked():
            xy = xy - 1

        xy = xy.astype(np.int64)
        xy_str = [str(t) for t in xy]
        self.mask_list_xylist.addItems(xy_str)

    def mask_list_add(self):
        pts = self.mask_list_input.text()
        self.mask_list_input.clear()
        if len(pts) < 3:
            return

        xy = text_to_array(pts)
        xy = xy[0: xy.size // 2 * 2].reshape(-1, 2)
        xy_str = [str(t) for t in xy]
        self.mask_list_xylist.addItems(xy_str)

    def mask_list_clear(self):
        self.mask_list_xylist.clear()


def run():
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleMaskGUI()
    app.exec_()


if __name__ == '__main__':
    run()
