# from PyQt5 import QtCore
from simple_mask_ui import Ui_SimpleMask as Ui
from simple_mask_kernel import SimpleMask
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
# import pyqtgraph as pg

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


class SimpleMaskGUI(QtWidgets.QMainWindow, Ui):
    def __init__(self, path=None):

        super(SimpleMaskGUI, self).__init__()

        self.setupUi(self)
        self.btn_load.clicked.connect(self.load)
        self.btn_add_roi.clicked.connect(self.add_roi)
        self.btn_apply_roi.clicked.connect(self.apply_roi)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_editlock.clicked.connect(self.editlock)
        self.btn_compute_qpartition.clicked.connect(self.compute_partition)
        self.btn_select_raw.clicked.connect(self.select_raw) # ... 
        self.btn_select_blemish.clicked.connect(self.select_blemish) # ... 
        self.btn_select_txt.clicked.connect(self.select_txt) # ... 
        

        # need a function for save button -- simple_mask_ui
        self.pushButton.clicked.connect(self.save_mask)

        self.plot_index.currentIndexChanged.connect(self.mp1.setCurrentIndex)

        # debug;
        # C:/Users/jeffr/Desktop/data/H432_OH_100_025C_att05_001/H432_OH_100_025C_att05_001_0001-1000.hdf
        self.blemish_fname.setText('')
        self.save_fname.setText('')     
        
        # self.save_fname.setText('../tests/data/qmap_output_simplemask.hdf')
        self.fname.setText('')

        # simple mask kernel
        self.sm = SimpleMask(self.mp1, self.infobar)
        self.mp1.sigTimeChanged.connect(self.update_index)
        self.state = 'lock'
        
        
        # ---------------------------------------------------------------------------------
        # preloaded mask - in progress
        
        self.mask_fname.clicked.connect(self.select_mask_file) # ...   
        # buttons for select, preview, apply mask
        self.pushButton_2.clicked.connect(self.select_mask) #select --> select_mask
        self.pushButton_4.clicked.connect(self.preview_mask) #preview --> preview_mask
        self.pushButton_3.clicked.connect(self.apply_mask) # apply--> apply_mask
        
        # ---------------------------------------------------------------------------------
        self.show()

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)

    def editlock(self):
        pvs = (self.db_cenx, self.db_ceny, self.db_energy, self.db_pix_dim,
               self.db_det_dist)

        if self.state == 'lock':
            self.state = 'edit'
            for pv in pvs:
                pv.setEnabled(True)
        elif self.state == 'edit':
            self.state = 'lock'
            values = []
            for pv in pvs:
                pv.setDisabled(True)
                values.append(pv.value())

            self.sm.update_parameters(values)

        self.groupBox.repaint()
        self.plot()
    
    def select_raw(self):
        fname = QFileDialog.getOpenFileName(self, 'Select raw file hdf')[0]
        if fname not in [None, '']:
            self.fname.setText(fname)
        return

    def select_blemish(self):
        fname = QFileDialog.getOpenFileName(self, 'Select blemish file')[0]
        if fname not in [None, '']:
            self.blemish_fname.setText(fname)
        return
    

    def load(self):
        fname = self.fname.text()
        blemish_fname = self.blemish_fname.text()
        text_fname = self.text_fname.text() 


        if not os.path.isfile(blemish_fname):
            blemish_fname = None
        if not os.path.isfile(fname):
            return

        self.sm.read_data(fname, blemish_fname, text_fname)

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
            'rotate': self.plot_rotate.isChecked(),
            'plot_center': self.plot_center.isChecked(),
        }
        self.sm.show_saxs(**kwargs)
        self.plot_index.setCurrentIndex(0)

    def add_roi(self):
        color = ('g', 'y', 'b', 'r', 'c', 'm', 'k', 'w')[
                self.cb_selector_color.currentIndex()]
        kwargs = {
            'color': color,
            'sl_type': self.cb_selector_type.currentText(),
            'sl_mode': self.cb_selector_mode.currentText(),
            'width': self.plot_width.value()
        }
        self.sm.add_roi(**kwargs)
        return

    def apply_roi(self):
        self.sm.apply_roi()
        self.plot_index.setCurrentIndex(2)
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
        save_fname = self.save_fname.text()
        self.sm.save_partition(save_fname)


    # ---------------------------------------------------------------------------------
    # preloaded mask - in progress
    
    def select_mask_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Select mask')[0]
        if fname not in [None, '']:
            self.mask_preload.setText(fname)
            
        return 
    
    def select_mask(self):
        mask_preload = self.mask_preload.text() # file for mask
        mask_directory = self.mask_directory.text() # directory in hdf

        self.mask = self.sm.select_mask(mask_preload, mask_directory)
        
    # ask miaoqi for help        
    def preview_mask(self):
        self.sm.preview_mask(self.mask)
        # self.plot_index.setCurrentIndex(5)
        
    def apply_mask(self):     
        self.sm.apply_mask(self.mask)
        
    # ---------------------------------------------------------------------------------        
    # modify blemish    
        
    def select_txt(self):
        fname = QFileDialog.getOpenFileName(self, 'Select text file')[0]
        if fname not in [None, '']:
            self.text_fname.setText(fname)
        return        
        

        
    def test(self):     
        print("test")
    # ---------------------------------------------------------------------------------


def run():
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleMaskGUI()
    app.exec_()


if __name__ == '__main__':
    run()