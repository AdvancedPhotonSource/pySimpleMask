# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mask.ui'
##
## Created by: Qt User Interface Compiler version 6.7.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QSpinBox, QSplitter, QStatusBar, QTabWidget,
    QToolButton, QVBoxLayout, QWidget)

from pyqtgraph import PlotWidget
from .pyqtgraph_mod import ImageViewROI

class Ui_SimpleMask(object):
    def setupUi(self, SimpleMask):
        if not SimpleMask.objectName():
            SimpleMask.setObjectName(u"SimpleMask")
        SimpleMask.resize(1678, 1017)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SimpleMask.sizePolicy().hasHeightForWidth())
        SimpleMask.setSizePolicy(sizePolicy)
        SimpleMask.setMinimumSize(QSize(800, 600))
        SimpleMask.setMaximumSize(QSize(16777215, 16777215))
        self.centralwidget = QWidget(SimpleMask)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_16 = QGridLayout(self.centralwidget)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.groupBox = QGroupBox(self.layoutWidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy1)
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cb_beamline = QComboBox(self.groupBox)
        self.cb_beamline.addItem("")
        self.cb_beamline.addItem("")
        self.cb_beamline.addItem("")
        self.cb_beamline.addItem("")
        self.cb_beamline.setObjectName(u"cb_beamline")

        self.horizontalLayout.addWidget(self.cb_beamline)

        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout.addWidget(self.label_9)

        self.begin_idx = QSpinBox(self.groupBox)
        self.begin_idx.setObjectName(u"begin_idx")
        self.begin_idx.setMaximum(99999)

        self.horizontalLayout.addWidget(self.begin_idx)

        self.label_28 = QLabel(self.groupBox)
        self.label_28.setObjectName(u"label_28")

        self.horizontalLayout.addWidget(self.label_28)

        self.num_frames = QSpinBox(self.groupBox)
        self.num_frames.setObjectName(u"num_frames")
        self.num_frames.setMinimum(-1)
        self.num_frames.setMaximum(1000000)
        self.num_frames.setSingleStep(100)
        self.num_frames.setValue(-1)

        self.horizontalLayout.addWidget(self.num_frames)

        self.btn_load = QPushButton(self.groupBox)
        self.btn_load.setObjectName(u"btn_load")

        self.horizontalLayout.addWidget(self.btn_load)


        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_17 = QLabel(self.groupBox)
        self.label_17.setObjectName(u"label_17")

        self.horizontalLayout_2.addWidget(self.label_17)

        self.fname = QLineEdit(self.groupBox)
        self.fname.setObjectName(u"fname")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.fname.sizePolicy().hasHeightForWidth())
        self.fname.setSizePolicy(sizePolicy2)
        self.fname.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.fname)

        self.btn_select_raw = QToolButton(self.groupBox)
        self.btn_select_raw.setObjectName(u"btn_select_raw")

        self.horizontalLayout_2.addWidget(self.btn_select_raw)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.metaTab = QTabWidget(self.groupBox)
        self.metaTab.setObjectName(u"metaTab")
        self.metaTab.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.metaTab.sizePolicy().hasHeightForWidth())
        self.metaTab.setSizePolicy(sizePolicy2)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_10 = QGridLayout(self.tab)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.db_pix_dim = QDoubleSpinBox(self.tab)
        self.db_pix_dim.setObjectName(u"db_pix_dim")
        self.db_pix_dim.setEnabled(True)
        self.db_pix_dim.setDecimals(4)

        self.gridLayout_10.addWidget(self.db_pix_dim, 2, 1, 1, 1)

        self.le_shape = QLineEdit(self.tab)
        self.le_shape.setObjectName(u"le_shape")
        self.le_shape.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.le_shape.sizePolicy().hasHeightForWidth())
        self.le_shape.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.le_shape, 2, 4, 1, 1)

        self.label_4 = QLabel(self.tab)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_4, 1, 0, 1, 1)

        self.db_energy = QDoubleSpinBox(self.tab)
        self.db_energy.setObjectName(u"db_energy")
        self.db_energy.setEnabled(True)
        self.db_energy.setDecimals(4)

        self.gridLayout_10.addWidget(self.db_energy, 1, 1, 1, 1)

        self.label_7 = QLabel(self.tab)
        self.label_7.setObjectName(u"label_7")
        sizePolicy1.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_7, 2, 3, 1, 1)

        self.db_bcx = QDoubleSpinBox(self.tab)
        self.db_bcx.setObjectName(u"db_bcx")
        self.db_bcx.setEnabled(True)
        self.db_bcx.setDecimals(4)
        self.db_bcx.setMinimum(-9999.000000000000000)
        self.db_bcx.setMaximum(9999.000000000000000)

        self.gridLayout_10.addWidget(self.db_bcx, 0, 1, 1, 1)

        self.db_bcy = QDoubleSpinBox(self.tab)
        self.db_bcy.setObjectName(u"db_bcy")
        self.db_bcy.setEnabled(True)
        self.db_bcy.setDecimals(4)
        self.db_bcy.setMinimum(-9999.000000000000000)
        self.db_bcy.setMaximum(9999.000000000000000)

        self.gridLayout_10.addWidget(self.db_bcy, 0, 4, 1, 1)

        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")
        sizePolicy1.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_6, 2, 0, 1, 1)

        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_5 = QLabel(self.tab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy1.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_5, 1, 3, 1, 1)

        self.label_21 = QLabel(self.tab)
        self.label_21.setObjectName(u"label_21")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy3)

        self.gridLayout_10.addWidget(self.label_21, 3, 0, 1, 1)

        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")
        sizePolicy1.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.label_3, 0, 3, 1, 1)

        self.db_det_dist = QDoubleSpinBox(self.tab)
        self.db_det_dist.setObjectName(u"db_det_dist")
        self.db_det_dist.setEnabled(True)
        self.db_det_dist.setDecimals(4)
        self.db_det_dist.setMaximum(99999.000000000000000)

        self.gridLayout_10.addWidget(self.db_det_dist, 1, 4, 1, 1)

        self.det_yaw = QDoubleSpinBox(self.tab)
        self.det_yaw.setObjectName(u"det_yaw")
        self.det_yaw.setEnabled(True)
        self.det_yaw.setDecimals(4)

        self.gridLayout_10.addWidget(self.det_yaw, 3, 1, 1, 1)

        self.metaTab.addTab(self.tab, "")
        self.tab_8 = QWidget()
        self.tab_8.setObjectName(u"tab_8")
        self.gridLayout_39 = QGridLayout(self.tab_8)
        self.gridLayout_39.setObjectName(u"gridLayout_39")
        self.label_51 = QLabel(self.tab_8)
        self.label_51.setObjectName(u"label_51")
        sizePolicy1.setHeightForWidth(self.label_51.sizePolicy().hasHeightForWidth())
        self.label_51.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_51, 0, 0, 1, 1)

        self.db_bcx_1 = QDoubleSpinBox(self.tab_8)
        self.db_bcx_1.setObjectName(u"db_bcx_1")
        self.db_bcx_1.setEnabled(True)
        self.db_bcx_1.setDecimals(4)
        self.db_bcx_1.setMinimum(-9999.000000000000000)
        self.db_bcx_1.setMaximum(9999.000000000000000)

        self.gridLayout_39.addWidget(self.db_bcx_1, 0, 1, 1, 1)

        self.btn_swapxy_3 = QPushButton(self.tab_8)
        self.btn_swapxy_3.setObjectName(u"btn_swapxy_3")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.btn_swapxy_3.sizePolicy().hasHeightForWidth())
        self.btn_swapxy_3.setSizePolicy(sizePolicy4)

        self.gridLayout_39.addWidget(self.btn_swapxy_3, 0, 2, 1, 1)

        self.label_53 = QLabel(self.tab_8)
        self.label_53.setObjectName(u"label_53")
        sizePolicy1.setHeightForWidth(self.label_53.sizePolicy().hasHeightForWidth())
        self.label_53.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_53, 0, 3, 1, 1)

        self.db_bcy_1 = QDoubleSpinBox(self.tab_8)
        self.db_bcy_1.setObjectName(u"db_bcy_1")
        self.db_bcy_1.setEnabled(True)
        self.db_bcy_1.setDecimals(4)
        self.db_bcy_1.setMinimum(-9999.000000000000000)
        self.db_bcy_1.setMaximum(9999.000000000000000)

        self.gridLayout_39.addWidget(self.db_bcy_1, 0, 4, 1, 1)

        self.label_52 = QLabel(self.tab_8)
        self.label_52.setObjectName(u"label_52")
        sizePolicy1.setHeightForWidth(self.label_52.sizePolicy().hasHeightForWidth())
        self.label_52.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_52, 1, 0, 1, 1)

        self.db_energy_1 = QDoubleSpinBox(self.tab_8)
        self.db_energy_1.setObjectName(u"db_energy_1")
        self.db_energy_1.setEnabled(True)
        self.db_energy_1.setDecimals(4)

        self.gridLayout_39.addWidget(self.db_energy_1, 1, 1, 1, 1)

        self.label_55 = QLabel(self.tab_8)
        self.label_55.setObjectName(u"label_55")
        sizePolicy1.setHeightForWidth(self.label_55.sizePolicy().hasHeightForWidth())
        self.label_55.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_55, 1, 3, 1, 1)

        self.db_det_dist_1 = QDoubleSpinBox(self.tab_8)
        self.db_det_dist_1.setObjectName(u"db_det_dist_1")
        self.db_det_dist_1.setEnabled(True)
        self.db_det_dist_1.setDecimals(4)
        self.db_det_dist_1.setMaximum(99999.000000000000000)

        self.gridLayout_39.addWidget(self.db_det_dist_1, 1, 4, 1, 1)

        self.label_49 = QLabel(self.tab_8)
        self.label_49.setObjectName(u"label_49")
        sizePolicy1.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_49, 2, 0, 1, 1)

        self.db_pix_dim_1 = QDoubleSpinBox(self.tab_8)
        self.db_pix_dim_1.setObjectName(u"db_pix_dim_1")
        self.db_pix_dim_1.setEnabled(True)
        self.db_pix_dim_1.setDecimals(4)

        self.gridLayout_39.addWidget(self.db_pix_dim_1, 2, 1, 1, 1)

        self.label_50 = QLabel(self.tab_8)
        self.label_50.setObjectName(u"label_50")
        sizePolicy1.setHeightForWidth(self.label_50.sizePolicy().hasHeightForWidth())
        self.label_50.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.label_50, 2, 3, 1, 1)

        self.le_shape_1 = QLineEdit(self.tab_8)
        self.le_shape_1.setObjectName(u"le_shape_1")
        self.le_shape_1.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.le_shape_1.sizePolicy().hasHeightForWidth())
        self.le_shape_1.setSizePolicy(sizePolicy1)

        self.gridLayout_39.addWidget(self.le_shape_1, 2, 4, 1, 1)

        self.label_54 = QLabel(self.tab_8)
        self.label_54.setObjectName(u"label_54")
        sizePolicy3.setHeightForWidth(self.label_54.sizePolicy().hasHeightForWidth())
        self.label_54.setSizePolicy(sizePolicy3)

        self.gridLayout_39.addWidget(self.label_54, 3, 0, 1, 1)

        self.alpha_i_1 = QDoubleSpinBox(self.tab_8)
        self.alpha_i_1.setObjectName(u"alpha_i_1")
        self.alpha_i_1.setEnabled(True)
        self.alpha_i_1.setDecimals(4)

        self.gridLayout_39.addWidget(self.alpha_i_1, 3, 1, 1, 1)

        self.btn_find_center_3 = QPushButton(self.tab_8)
        self.btn_find_center_3.setObjectName(u"btn_find_center_3")

        self.gridLayout_39.addWidget(self.btn_find_center_3, 3, 3, 1, 1)

        self.btn_update_parameters_3 = QPushButton(self.tab_8)
        self.btn_update_parameters_3.setObjectName(u"btn_update_parameters_3")

        self.gridLayout_39.addWidget(self.btn_update_parameters_3, 3, 4, 1, 1)

        self.metaTab.addTab(self.tab_8, "")

        self.gridLayout_2.addWidget(self.metaTab, 2, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.btn_swapxy = QPushButton(self.groupBox)
        self.btn_swapxy.setObjectName(u"btn_swapxy")
        sizePolicy4.setHeightForWidth(self.btn_swapxy.sizePolicy().hasHeightForWidth())
        self.btn_swapxy.setSizePolicy(sizePolicy4)

        self.horizontalLayout_3.addWidget(self.btn_swapxy)

        self.btn_find_center = QPushButton(self.groupBox)
        self.btn_find_center.setObjectName(u"btn_find_center")

        self.horizontalLayout_3.addWidget(self.btn_find_center)

        self.btn_update_parameters = QPushButton(self.groupBox)
        self.btn_update_parameters.setObjectName(u"btn_update_parameters")

        self.horizontalLayout_3.addWidget(self.btn_update_parameters)


        self.gridLayout_2.addLayout(self.horizontalLayout_3, 3, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_4 = QGroupBox(self.layoutWidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy1.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy1)
        self.gridLayout_3 = QGridLayout(self.groupBox_4)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.plot_center = QCheckBox(self.groupBox_4)
        self.plot_center.setObjectName(u"plot_center")
        self.plot_center.setChecked(True)

        self.gridLayout_3.addWidget(self.plot_center, 0, 2, 1, 1)

        self.plot_log = QCheckBox(self.groupBox_4)
        self.plot_log.setObjectName(u"plot_log")
        self.plot_log.setChecked(True)

        self.gridLayout_3.addWidget(self.plot_log, 0, 3, 1, 1)

        self.plot_cmap = QComboBox(self.groupBox_4)
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.addItem("")
        self.plot_cmap.setObjectName(u"plot_cmap")
        sizePolicy1.setHeightForWidth(self.plot_cmap.sizePolicy().hasHeightForWidth())
        self.plot_cmap.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.plot_cmap, 0, 4, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(59, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_6, 0, 6, 1, 1)

        self.btn_plot = QPushButton(self.groupBox_4)
        self.btn_plot.setObjectName(u"btn_plot")

        self.gridLayout_3.addWidget(self.btn_plot, 0, 5, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(59, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_7, 0, 0, 1, 1)

        self.plot_invert = QCheckBox(self.groupBox_4)
        self.plot_invert.setObjectName(u"plot_invert")

        self.gridLayout_3.addWidget(self.plot_invert, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.groupBox_2 = QGroupBox(self.layoutWidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy5)
        self.gridLayout_4 = QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.btn_mask_reset = QPushButton(self.groupBox_2)
        self.btn_mask_reset.setObjectName(u"btn_mask_reset")
        self.btn_mask_reset.setMinimumSize(QSize(80, 0))

        self.gridLayout_4.addWidget(self.btn_mask_reset, 2, 0, 1, 1)

        self.btn_mask_undo = QPushButton(self.groupBox_2)
        self.btn_mask_undo.setObjectName(u"btn_mask_undo")
        self.btn_mask_undo.setMinimumSize(QSize(80, 0))

        self.gridLayout_4.addWidget(self.btn_mask_undo, 2, 2, 1, 1)

        self.btn_mask_apply = QPushButton(self.groupBox_2)
        self.btn_mask_apply.setObjectName(u"btn_mask_apply")
        sizePolicy4.setHeightForWidth(self.btn_mask_apply.sizePolicy().hasHeightForWidth())
        self.btn_mask_apply.setSizePolicy(sizePolicy4)
        self.btn_mask_apply.setMinimumSize(QSize(80, 0))
        self.btn_mask_apply.setMaximumSize(QSize(80, 16777215))

        self.gridLayout_4.addWidget(self.btn_mask_apply, 2, 4, 1, 1)

        self.btn_mask_redo = QPushButton(self.groupBox_2)
        self.btn_mask_redo.setObjectName(u"btn_mask_redo")
        self.btn_mask_redo.setMinimumSize(QSize(80, 0))

        self.gridLayout_4.addWidget(self.btn_mask_redo, 2, 1, 1, 1)

        self.MaskWidget = QTabWidget(self.groupBox_2)
        self.MaskWidget.setObjectName(u"MaskWidget")
        sizePolicy5.setHeightForWidth(self.MaskWidget.sizePolicy().hasHeightForWidth())
        self.MaskWidget.setSizePolicy(sizePolicy5)
        self.tab_6 = QWidget()
        self.tab_6.setObjectName(u"tab_6")
        self.gridLayout_25 = QGridLayout(self.tab_6)
        self.gridLayout_25.setObjectName(u"gridLayout_25")
        self.groupBox_7 = QGroupBox(self.tab_6)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.gridLayout_9 = QGridLayout(self.groupBox_7)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.label_30 = QLabel(self.groupBox_7)
        self.label_30.setObjectName(u"label_30")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy6)

        self.gridLayout_9.addWidget(self.label_30, 0, 0, 1, 1)

        self.blemish_fname = QLineEdit(self.groupBox_7)
        self.blemish_fname.setObjectName(u"blemish_fname")
        sizePolicy3.setHeightForWidth(self.blemish_fname.sizePolicy().hasHeightForWidth())
        self.blemish_fname.setSizePolicy(sizePolicy3)
        self.blemish_fname.setMinimumSize(QSize(300, 20))

        self.gridLayout_9.addWidget(self.blemish_fname, 0, 1, 1, 1)

        self.btn_select_blemish = QPushButton(self.groupBox_7)
        self.btn_select_blemish.setObjectName(u"btn_select_blemish")
        sizePolicy1.setHeightForWidth(self.btn_select_blemish.sizePolicy().hasHeightForWidth())
        self.btn_select_blemish.setSizePolicy(sizePolicy1)

        self.gridLayout_9.addWidget(self.btn_select_blemish, 0, 2, 1, 1)

        self.label_31 = QLabel(self.groupBox_7)
        self.label_31.setObjectName(u"label_31")
        sizePolicy6.setHeightForWidth(self.label_31.sizePolicy().hasHeightForWidth())
        self.label_31.setSizePolicy(sizePolicy6)

        self.gridLayout_9.addWidget(self.label_31, 1, 0, 1, 1)

        self.blemish_path = QLineEdit(self.groupBox_7)
        self.blemish_path.setObjectName(u"blemish_path")
        sizePolicy3.setHeightForWidth(self.blemish_path.sizePolicy().hasHeightForWidth())
        self.blemish_path.setSizePolicy(sizePolicy3)
        self.blemish_path.setMinimumSize(QSize(300, 20))

        self.gridLayout_9.addWidget(self.blemish_path, 1, 1, 1, 1)

        self.btn_apply_blemish = QPushButton(self.groupBox_7)
        self.btn_apply_blemish.setObjectName(u"btn_apply_blemish")
        sizePolicy4.setHeightForWidth(self.btn_apply_blemish.sizePolicy().hasHeightForWidth())
        self.btn_apply_blemish.setSizePolicy(sizePolicy4)
        self.btn_apply_blemish.setMinimumSize(QSize(60, 0))

        self.gridLayout_9.addWidget(self.btn_apply_blemish, 1, 2, 1, 1)


        self.gridLayout_25.addWidget(self.groupBox_7, 0, 0, 1, 1)

        self.groupBox_8 = QGroupBox(self.tab_6)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.gridLayout_12 = QGridLayout(self.groupBox_8)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.label_18 = QLabel(self.groupBox_8)
        self.label_18.setObjectName(u"label_18")
        sizePolicy6.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy6)

        self.gridLayout_12.addWidget(self.label_18, 0, 0, 1, 1)

        self.maskfile_fname = QLineEdit(self.groupBox_8)
        self.maskfile_fname.setObjectName(u"maskfile_fname")
        sizePolicy3.setHeightForWidth(self.maskfile_fname.sizePolicy().hasHeightForWidth())
        self.maskfile_fname.setSizePolicy(sizePolicy3)
        self.maskfile_fname.setMinimumSize(QSize(300, 20))

        self.gridLayout_12.addWidget(self.maskfile_fname, 0, 1, 1, 1)

        self.btn_select_maskfile = QPushButton(self.groupBox_8)
        self.btn_select_maskfile.setObjectName(u"btn_select_maskfile")
        sizePolicy1.setHeightForWidth(self.btn_select_maskfile.sizePolicy().hasHeightForWidth())
        self.btn_select_maskfile.setSizePolicy(sizePolicy1)

        self.gridLayout_12.addWidget(self.btn_select_maskfile, 0, 2, 1, 1)

        self.label_32 = QLabel(self.groupBox_8)
        self.label_32.setObjectName(u"label_32")
        sizePolicy5.setHeightForWidth(self.label_32.sizePolicy().hasHeightForWidth())
        self.label_32.setSizePolicy(sizePolicy5)

        self.gridLayout_12.addWidget(self.label_32, 1, 0, 1, 1)

        self.maskfile_path = QLineEdit(self.groupBox_8)
        self.maskfile_path.setObjectName(u"maskfile_path")
        sizePolicy3.setHeightForWidth(self.maskfile_path.sizePolicy().hasHeightForWidth())
        self.maskfile_path.setSizePolicy(sizePolicy3)
        self.maskfile_path.setMinimumSize(QSize(300, 20))

        self.gridLayout_12.addWidget(self.maskfile_path, 1, 1, 1, 1)

        self.btn_apply_maskfile = QPushButton(self.groupBox_8)
        self.btn_apply_maskfile.setObjectName(u"btn_apply_maskfile")
        sizePolicy4.setHeightForWidth(self.btn_apply_maskfile.sizePolicy().hasHeightForWidth())
        self.btn_apply_maskfile.setSizePolicy(sizePolicy4)
        self.btn_apply_maskfile.setMinimumSize(QSize(60, 0))

        self.gridLayout_12.addWidget(self.btn_apply_maskfile, 1, 2, 1, 1)


        self.gridLayout_25.addWidget(self.groupBox_8, 1, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_25.addItem(self.verticalSpacer_2, 2, 0, 1, 1)

        self.MaskWidget.addTab(self.tab_6, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_13 = QGridLayout(self.tab_2)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.groupBox_21 = QGroupBox(self.tab_2)
        self.groupBox_21.setObjectName(u"groupBox_21")
        self.gridLayout_8 = QGridLayout(self.groupBox_21)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.cb_selector_mode = QComboBox(self.groupBox_21)
        self.cb_selector_mode.addItem("")
        self.cb_selector_mode.addItem("")
        self.cb_selector_mode.setObjectName(u"cb_selector_mode")

        self.gridLayout_8.addWidget(self.cb_selector_mode, 0, 4, 1, 1)

        self.btn_mask_draw_add = QPushButton(self.groupBox_21)
        self.btn_mask_draw_add.setObjectName(u"btn_mask_draw_add")

        self.gridLayout_8.addWidget(self.btn_mask_draw_add, 0, 5, 1, 1)

        self.btn_mask_draw_evaluate = QPushButton(self.groupBox_21)
        self.btn_mask_draw_evaluate.setObjectName(u"btn_mask_draw_evaluate")

        self.gridLayout_8.addWidget(self.btn_mask_draw_evaluate, 1, 5, 1, 1)

        self.label_22 = QLabel(self.groupBox_21)
        self.label_22.setObjectName(u"label_22")

        self.gridLayout_8.addWidget(self.label_22, 0, 3, 1, 1)

        self.plot_width = QSpinBox(self.groupBox_21)
        self.plot_width.setObjectName(u"plot_width")
        self.plot_width.setMinimum(1)
        self.plot_width.setValue(1)

        self.gridLayout_8.addWidget(self.plot_width, 1, 4, 1, 1)

        self.label_14 = QLabel(self.groupBox_21)
        self.label_14.setObjectName(u"label_14")
        sizePolicy5.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy5)

        self.gridLayout_8.addWidget(self.label_14, 1, 0, 1, 1)

        self.label_23 = QLabel(self.groupBox_21)
        self.label_23.setObjectName(u"label_23")

        self.gridLayout_8.addWidget(self.label_23, 0, 0, 1, 1)

        self.label_8 = QLabel(self.groupBox_21)
        self.label_8.setObjectName(u"label_8")
        sizePolicy5.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy5)

        self.gridLayout_8.addWidget(self.label_8, 1, 3, 1, 1)

        self.cb_selector_type = QComboBox(self.groupBox_21)
        self.cb_selector_type.addItem("")
        self.cb_selector_type.addItem("")
        self.cb_selector_type.addItem("")
        self.cb_selector_type.addItem("")
        self.cb_selector_type.setObjectName(u"cb_selector_type")

        self.gridLayout_8.addWidget(self.cb_selector_type, 0, 1, 1, 1)

        self.cb_selector_color = QComboBox(self.groupBox_21)
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.addItem("")
        self.cb_selector_color.setObjectName(u"cb_selector_color")

        self.gridLayout_8.addWidget(self.cb_selector_color, 1, 1, 1, 1)


        self.gridLayout_13.addWidget(self.groupBox_21, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 193, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_13.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.MaskWidget.addTab(self.tab_2, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.gridLayout_14 = QGridLayout(self.tab_5)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.mask_outlier_hdl = PlotWidget(self.tab_5)
        self.mask_outlier_hdl.setObjectName(u"mask_outlier_hdl")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.mask_outlier_hdl.sizePolicy().hasHeightForWidth())
        self.mask_outlier_hdl.setSizePolicy(sizePolicy7)
        self.mask_outlier_hdl.setMinimumSize(QSize(0, 0))

        self.gridLayout_14.addWidget(self.mask_outlier_hdl, 3, 0, 1, 1)

        self.groupBox_9 = QGroupBox(self.tab_5)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.gridLayout_15 = QGridLayout(self.groupBox_9)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.binary_threshold_low = QDoubleSpinBox(self.groupBox_9)
        self.binary_threshold_low.setObjectName(u"binary_threshold_low")
        self.binary_threshold_low.setDecimals(3)
        self.binary_threshold_low.setMinimum(-10000.000000000000000)
        self.binary_threshold_low.setMaximum(10000.000000000000000)

        self.gridLayout_15.addWidget(self.binary_threshold_low, 0, 1, 1, 1)

        self.btn_mask_threshold_evaluate = QPushButton(self.groupBox_9)
        self.btn_mask_threshold_evaluate.setObjectName(u"btn_mask_threshold_evaluate")

        self.gridLayout_15.addWidget(self.btn_mask_threshold_evaluate, 0, 5, 1, 1)

        self.binary_threshold_high = QDoubleSpinBox(self.groupBox_9)
        self.binary_threshold_high.setObjectName(u"binary_threshold_high")
        self.binary_threshold_high.setDecimals(3)
        self.binary_threshold_high.setMaximum(9999.000000000000000)
        self.binary_threshold_high.setSingleStep(100.000000000000000)
        self.binary_threshold_high.setValue(9999.000000000000000)

        self.gridLayout_15.addWidget(self.binary_threshold_high, 0, 3, 1, 1)

        self.binary_scale = QComboBox(self.groupBox_9)
        self.binary_scale.addItem("")
        self.binary_scale.addItem("")
        self.binary_scale.setObjectName(u"binary_scale")

        self.gridLayout_15.addWidget(self.binary_scale, 0, 4, 1, 1)

        self.label_26 = QLabel(self.groupBox_9)
        self.label_26.setObjectName(u"label_26")

        self.gridLayout_15.addWidget(self.label_26, 0, 0, 1, 1)

        self.label_27 = QLabel(self.groupBox_9)
        self.label_27.setObjectName(u"label_27")
        sizePolicy1.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy1)

        self.gridLayout_15.addWidget(self.label_27, 0, 2, 1, 1)


        self.gridLayout_14.addWidget(self.groupBox_9, 0, 0, 1, 1)

        self.groupBox_19 = QGroupBox(self.tab_5)
        self.groupBox_19.setObjectName(u"groupBox_19")
        self.groupBox_19.setMinimumSize(QSize(0, 60))
        self.gridLayout_17 = QGridLayout(self.groupBox_19)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.label_16 = QLabel(self.groupBox_19)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_17.addWidget(self.label_16, 0, 0, 1, 1)

        self.outlier_num_roi = QSpinBox(self.groupBox_19)
        self.outlier_num_roi.setObjectName(u"outlier_num_roi")
        self.outlier_num_roi.setMaximum(2000)
        self.outlier_num_roi.setValue(400)

        self.gridLayout_17.addWidget(self.outlier_num_roi, 0, 1, 1, 1)

        self.label_15 = QLabel(self.groupBox_19)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_17.addWidget(self.label_15, 0, 2, 1, 1)

        self.outlier_cutoff = QDoubleSpinBox(self.groupBox_19)
        self.outlier_cutoff.setObjectName(u"outlier_cutoff")
        self.outlier_cutoff.setMaximum(999.000000000000000)
        self.outlier_cutoff.setSingleStep(0.100000000000000)
        self.outlier_cutoff.setValue(3.000000000000000)

        self.gridLayout_17.addWidget(self.outlier_cutoff, 0, 3, 1, 1)

        self.btn_mask_outlier_evaluate = QPushButton(self.groupBox_19)
        self.btn_mask_outlier_evaluate.setObjectName(u"btn_mask_outlier_evaluate")

        self.gridLayout_17.addWidget(self.btn_mask_outlier_evaluate, 0, 4, 1, 1)


        self.gridLayout_14.addWidget(self.groupBox_19, 1, 0, 1, 1)

        self.groupBox_20 = QGroupBox(self.tab_5)
        self.groupBox_20.setObjectName(u"groupBox_20")
        self.groupBox_20.setMinimumSize(QSize(0, 60))
        self.gridLayout_21 = QGridLayout(self.groupBox_20)
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.label_35 = QLabel(self.groupBox_20)
        self.label_35.setObjectName(u"label_35")

        self.gridLayout_21.addWidget(self.label_35, 0, 0, 1, 1)

        self.outlier_num_roi_2 = QSpinBox(self.groupBox_20)
        self.outlier_num_roi_2.setObjectName(u"outlier_num_roi_2")
        self.outlier_num_roi_2.setMaximum(2000)
        self.outlier_num_roi_2.setValue(400)

        self.gridLayout_21.addWidget(self.outlier_num_roi_2, 0, 1, 1, 1)

        self.label_40 = QLabel(self.groupBox_20)
        self.label_40.setObjectName(u"label_40")

        self.gridLayout_21.addWidget(self.label_40, 0, 2, 1, 1)

        self.outlier_cutoff_2 = QDoubleSpinBox(self.groupBox_20)
        self.outlier_cutoff_2.setObjectName(u"outlier_cutoff_2")
        self.outlier_cutoff_2.setMaximum(999.000000000000000)
        self.outlier_cutoff_2.setSingleStep(0.100000000000000)
        self.outlier_cutoff_2.setValue(3.000000000000000)

        self.gridLayout_21.addWidget(self.outlier_cutoff_2, 0, 3, 1, 1)

        self.btn_mask_outlier_evaluate_2 = QPushButton(self.groupBox_20)
        self.btn_mask_outlier_evaluate_2.setObjectName(u"btn_mask_outlier_evaluate_2")

        self.gridLayout_21.addWidget(self.btn_mask_outlier_evaluate_2, 0, 4, 1, 1)


        self.gridLayout_14.addWidget(self.groupBox_20, 2, 0, 1, 1)

        self.MaskWidget.addTab(self.tab_5, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_5 = QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.groupBox_12 = QGroupBox(self.tab_3)
        self.groupBox_12.setObjectName(u"groupBox_12")
        sizePolicy3.setHeightForWidth(self.groupBox_12.sizePolicy().hasHeightForWidth())
        self.groupBox_12.setSizePolicy(sizePolicy3)
        self.gridLayout_18 = QGridLayout(self.groupBox_12)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.mask_list_1based = QCheckBox(self.groupBox_12)
        self.mask_list_1based.setObjectName(u"mask_list_1based")

        self.gridLayout_18.addWidget(self.mask_list_1based, 0, 0, 1, 1)

        self.mask_list_rowcol = QCheckBox(self.groupBox_12)
        self.mask_list_rowcol.setObjectName(u"mask_list_rowcol")

        self.gridLayout_18.addWidget(self.mask_list_rowcol, 0, 1, 1, 1)

        self.btn_mask_list_load = QPushButton(self.groupBox_12)
        self.btn_mask_list_load.setObjectName(u"btn_mask_list_load")

        self.gridLayout_18.addWidget(self.btn_mask_list_load, 0, 2, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox_12, 0, 0, 1, 1)

        self.groupBox_11 = QGroupBox(self.tab_3)
        self.groupBox_11.setObjectName(u"groupBox_11")
        sizePolicy.setHeightForWidth(self.groupBox_11.sizePolicy().hasHeightForWidth())
        self.groupBox_11.setSizePolicy(sizePolicy)
        self.gridLayout_22 = QGridLayout(self.groupBox_11)
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.mask_list_xylist = QListWidget(self.groupBox_11)
        self.mask_list_xylist.setObjectName(u"mask_list_xylist")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.mask_list_xylist.sizePolicy().hasHeightForWidth())
        self.mask_list_xylist.setSizePolicy(sizePolicy8)
        self.mask_list_xylist.setMaximumSize(QSize(500, 16777215))

        self.gridLayout_22.addWidget(self.mask_list_xylist, 0, 0, 1, 3)

        self.btn_mask_list_clear = QPushButton(self.groupBox_11)
        self.btn_mask_list_clear.setObjectName(u"btn_mask_list_clear")
        sizePolicy4.setHeightForWidth(self.btn_mask_list_clear.sizePolicy().hasHeightForWidth())
        self.btn_mask_list_clear.setSizePolicy(sizePolicy4)

        self.gridLayout_22.addWidget(self.btn_mask_list_clear, 1, 0, 1, 1)

        self.btn_mask_list_evaluate = QPushButton(self.groupBox_11)
        self.btn_mask_list_evaluate.setObjectName(u"btn_mask_list_evaluate")
        sizePolicy4.setHeightForWidth(self.btn_mask_list_evaluate.sizePolicy().hasHeightForWidth())
        self.btn_mask_list_evaluate.setSizePolicy(sizePolicy4)

        self.gridLayout_22.addWidget(self.btn_mask_list_evaluate, 1, 1, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox_11, 0, 1, 3, 1)

        self.groupBox_13 = QGroupBox(self.tab_3)
        self.groupBox_13.setObjectName(u"groupBox_13")
        sizePolicy3.setHeightForWidth(self.groupBox_13.sizePolicy().hasHeightForWidth())
        self.groupBox_13.setSizePolicy(sizePolicy3)
        self.gridLayout_19 = QGridLayout(self.groupBox_13)
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.btn_mask_list_add = QPushButton(self.groupBox_13)
        self.btn_mask_list_add.setObjectName(u"btn_mask_list_add")
        sizePolicy4.setHeightForWidth(self.btn_mask_list_add.sizePolicy().hasHeightForWidth())
        self.btn_mask_list_add.setSizePolicy(sizePolicy4)

        self.gridLayout_19.addWidget(self.btn_mask_list_add, 0, 1, 1, 1)

        self.mask_list_input = QLineEdit(self.groupBox_13)
        self.mask_list_input.setObjectName(u"mask_list_input")
        sizePolicy3.setHeightForWidth(self.mask_list_input.sizePolicy().hasHeightForWidth())
        self.mask_list_input.setSizePolicy(sizePolicy3)
        self.mask_list_input.setMinimumSize(QSize(200, 0))

        self.gridLayout_19.addWidget(self.mask_list_input, 0, 0, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox_13, 1, 0, 1, 1)

        self.groupBox_14 = QGroupBox(self.tab_3)
        self.groupBox_14.setObjectName(u"groupBox_14")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.groupBox_14.sizePolicy().hasHeightForWidth())
        self.groupBox_14.setSizePolicy(sizePolicy9)
        self.gridLayout_27 = QGridLayout(self.groupBox_14)
        self.gridLayout_27.setObjectName(u"gridLayout_27")
        self.mask_list_include = QCheckBox(self.groupBox_14)
        self.mask_list_include.setObjectName(u"mask_list_include")
        self.mask_list_include.setChecked(True)

        self.gridLayout_27.addWidget(self.mask_list_include, 0, 1, 1, 2)

        self.label_19 = QLabel(self.groupBox_14)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_27.addWidget(self.label_19, 1, 0, 1, 1)

        self.mask_list_radius = QDoubleSpinBox(self.groupBox_14)
        self.mask_list_radius.setObjectName(u"mask_list_radius")
        self.mask_list_radius.setMinimum(10.000000000000000)
        self.mask_list_radius.setMaximum(1000.000000000000000)
        self.mask_list_radius.setValue(50.000000000000000)

        self.gridLayout_27.addWidget(self.mask_list_radius, 1, 1, 1, 1)

        self.label_24 = QLabel(self.groupBox_14)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_27.addWidget(self.label_24, 1, 2, 1, 1)

        self.mask_list_variation = QDoubleSpinBox(self.groupBox_14)
        self.mask_list_variation.setObjectName(u"mask_list_variation")
        self.mask_list_variation.setMinimum(1.000000000000000)
        self.mask_list_variation.setValue(80.000000000000000)

        self.gridLayout_27.addWidget(self.mask_list_variation, 1, 3, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox_14, 2, 0, 1, 1)

        self.MaskWidget.addTab(self.tab_3, "")
        self.tab_7 = QWidget()
        self.tab_7.setObjectName(u"tab_7")
        self.gridLayout_23 = QGridLayout(self.tab_7)
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.groupBox_10 = QGroupBox(self.tab_7)
        self.groupBox_10.setObjectName(u"groupBox_10")
        self.groupBox_10.setEnabled(True)
        self.gridLayout_20 = QGridLayout(self.groupBox_10)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.pushButton_5 = QPushButton(self.groupBox_10)
        self.pushButton_5.setObjectName(u"pushButton_5")

        self.gridLayout_20.addWidget(self.pushButton_5, 0, 0, 1, 1)

        self.pushButton_9 = QPushButton(self.groupBox_10)
        self.pushButton_9.setObjectName(u"pushButton_9")

        self.gridLayout_20.addWidget(self.pushButton_9, 0, 1, 1, 1)

        self.pushButton_16 = QPushButton(self.groupBox_10)
        self.pushButton_16.setObjectName(u"pushButton_16")

        self.gridLayout_20.addWidget(self.pushButton_16, 0, 2, 1, 1)

        self.pushButton_15 = QPushButton(self.groupBox_10)
        self.pushButton_15.setObjectName(u"pushButton_15")

        self.gridLayout_20.addWidget(self.pushButton_15, 0, 3, 1, 1)

        self.pushButton_2 = QPushButton(self.groupBox_10)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.gridLayout_20.addWidget(self.pushButton_2, 0, 4, 1, 1)


        self.gridLayout_23.addWidget(self.groupBox_10, 0, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_23.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.MaskWidget.addTab(self.tab_7, "")

        self.gridLayout_4.addWidget(self.MaskWidget, 1, 0, 1, 5)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget2 = QWidget(self.splitter)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.groupBox_5 = QGroupBox(self.layoutWidget2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy10)
        self.groupBox_5.setMinimumSize(QSize(800, 0))
        self.gridLayout = QGridLayout(self.groupBox_5)
        self.gridLayout.setObjectName(u"gridLayout")
        self.infobar = QLineEdit(self.groupBox_5)
        self.infobar.setObjectName(u"infobar")
        font = QFont()
        font.setFamilies([u"Courier New"])
        self.infobar.setFont(font)

        self.gridLayout.addWidget(self.infobar, 0, 2, 1, 1)

        self.label_11 = QLabel(self.groupBox_5)
        self.label_11.setObjectName(u"label_11")
        sizePolicy6.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy6)

        self.gridLayout.addWidget(self.label_11, 0, 1, 1, 1)

        self.mp1 = ImageViewROI(self.groupBox_5)
        self.mp1.setObjectName(u"mp1")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy11.setHorizontalStretch(4)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.mp1.sizePolicy().hasHeightForWidth())
        self.mp1.setSizePolicy(sizePolicy11)
        self.mp1.setMinimumSize(QSize(400, 0))
        self.mp1.setFocusPolicy(Qt.ClickFocus)

        self.gridLayout.addWidget(self.mp1, 2, 0, 1, 3)

        self.plot_index = QComboBox(self.groupBox_5)
        self.plot_index.addItem("")
        self.plot_index.addItem("")
        self.plot_index.addItem("")
        self.plot_index.addItem("")
        self.plot_index.addItem("")
        self.plot_index.addItem("")
        self.plot_index.setObjectName(u"plot_index")
        sizePolicy4.setHeightForWidth(self.plot_index.sizePolicy().hasHeightForWidth())
        self.plot_index.setSizePolicy(sizePolicy4)

        self.gridLayout.addWidget(self.plot_index, 0, 0, 1, 1)


        self.verticalLayout_2.addWidget(self.groupBox_5)

        self.groupBox_3 = QGroupBox(self.layoutWidget2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        sizePolicy1.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy1)
        self.gridLayout_6 = QGridLayout(self.groupBox_3)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.cb_qmap_axis0 = QComboBox(self.groupBox_3)
        self.cb_qmap_axis0.setObjectName(u"cb_qmap_axis0")

        self.gridLayout_6.addWidget(self.cb_qmap_axis0, 1, 0, 1, 1)

        self.label_13 = QLabel(self.groupBox_3)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_6.addWidget(self.label_13, 0, 5, 1, 1)

        self.rb_beg_axis0 = QRadioButton(self.groupBox_3)
        self.rb_beg_axis0.setObjectName(u"rb_beg_axis0")
        self.rb_beg_axis0.setMaximumSize(QSize(20, 16777215))
        self.rb_beg_axis0.setCheckable(True)

        self.gridLayout_6.addWidget(self.rb_beg_axis0, 1, 2, 1, 1)

        self.label_20 = QLabel(self.groupBox_3)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_6.addWidget(self.label_20, 0, 8, 1, 1)

        self.label_33 = QLabel(self.groupBox_3)
        self.label_33.setObjectName(u"label_33")

        self.gridLayout_6.addWidget(self.label_33, 0, 1, 1, 1)

        self.unit_axis0 = QLabel(self.groupBox_3)
        self.unit_axis0.setObjectName(u"unit_axis0")

        self.gridLayout_6.addWidget(self.unit_axis0, 1, 1, 1, 1)

        self.sn_axis1 = QSpinBox(self.groupBox_3)
        self.sn_axis1.setObjectName(u"sn_axis1")
        self.sn_axis1.setMinimum(1)
        self.sn_axis1.setMaximum(9999)
        self.sn_axis1.setValue(1)

        self.gridLayout_6.addWidget(self.sn_axis1, 2, 6, 1, 1)

        self.partition_style_axis1 = QComboBox(self.groupBox_3)
        self.partition_style_axis1.addItem("")
        self.partition_style_axis1.addItem("")
        self.partition_style_axis1.setObjectName(u"partition_style_axis1")

        self.gridLayout_6.addWidget(self.partition_style_axis1, 2, 8, 1, 1)

        self.unit_axis1 = QLabel(self.groupBox_3)
        self.unit_axis1.setObjectName(u"unit_axis1")

        self.gridLayout_6.addWidget(self.unit_axis1, 2, 1, 1, 1)

        self.vbeg_axis1 = QDoubleSpinBox(self.groupBox_3)
        self.vbeg_axis1.setObjectName(u"vbeg_axis1")
        self.vbeg_axis1.setEnabled(False)
        self.vbeg_axis1.setMinimumSize(QSize(0, 20))
        self.vbeg_axis1.setDecimals(6)
        self.vbeg_axis1.setMinimum(-10000.000000000000000)
        self.vbeg_axis1.setMaximum(10000.000000000000000)
        self.vbeg_axis1.setValue(0.000000000000000)

        self.gridLayout_6.addWidget(self.vbeg_axis1, 2, 3, 1, 1)

        self.label_39 = QLabel(self.groupBox_3)
        self.label_39.setObjectName(u"label_39")

        self.gridLayout_6.addWidget(self.label_39, 0, 0, 1, 1)

        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")

        self.gridLayout_6.addWidget(self.label, 0, 6, 1, 1)

        self.dn_axis1 = QSpinBox(self.groupBox_3)
        self.dn_axis1.setObjectName(u"dn_axis1")
        self.dn_axis1.setMinimum(1)
        self.dn_axis1.setMaximum(999)
        self.dn_axis1.setValue(1)

        self.gridLayout_6.addWidget(self.dn_axis1, 2, 7, 1, 1)

        self.vend_axis1 = QDoubleSpinBox(self.groupBox_3)
        self.vend_axis1.setObjectName(u"vend_axis1")
        self.vend_axis1.setEnabled(False)
        self.vend_axis1.setMinimumSize(QSize(0, 20))
        self.vend_axis1.setDecimals(6)
        self.vend_axis1.setMinimum(-10000.000000000000000)
        self.vend_axis1.setMaximum(10000.000000000000000)
        self.vend_axis1.setValue(360.000000000000000)

        self.gridLayout_6.addWidget(self.vend_axis1, 2, 5, 1, 1)

        self.sn_axis0 = QSpinBox(self.groupBox_3)
        self.sn_axis0.setObjectName(u"sn_axis0")
        self.sn_axis0.setMinimum(2)
        self.sn_axis0.setMaximum(9999)
        self.sn_axis0.setValue(360)

        self.gridLayout_6.addWidget(self.sn_axis0, 1, 6, 1, 1)

        self.label_12 = QLabel(self.groupBox_3)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_6.addWidget(self.label_12, 0, 3, 1, 1)

        self.rb_end_axis1 = QRadioButton(self.groupBox_3)
        self.rb_end_axis1.setObjectName(u"rb_end_axis1")
        self.rb_end_axis1.setMaximumSize(QSize(20, 16777215))

        self.gridLayout_6.addWidget(self.rb_end_axis1, 2, 4, 1, 1)

        self.dn_axis0 = QSpinBox(self.groupBox_3)
        self.dn_axis0.setObjectName(u"dn_axis0")
        self.dn_axis0.setMinimum(1)
        self.dn_axis0.setMaximum(999)
        self.dn_axis0.setValue(36)

        self.gridLayout_6.addWidget(self.dn_axis0, 1, 7, 1, 1)

        self.cb_qmap_axis1 = QComboBox(self.groupBox_3)
        self.cb_qmap_axis1.setObjectName(u"cb_qmap_axis1")

        self.gridLayout_6.addWidget(self.cb_qmap_axis1, 2, 0, 1, 1)

        self.label_10 = QLabel(self.groupBox_3)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_6.addWidget(self.label_10, 0, 7, 1, 1)

        self.vend_axis0 = QDoubleSpinBox(self.groupBox_3)
        self.vend_axis0.setObjectName(u"vend_axis0")
        self.vend_axis0.setEnabled(False)
        self.vend_axis0.setMinimumSize(QSize(0, 20))
        self.vend_axis0.setDecimals(6)
        self.vend_axis0.setMinimum(-10000.000000000000000)
        self.vend_axis0.setMaximum(10000.000000000000000)
        self.vend_axis0.setValue(0.004200000000000)

        self.gridLayout_6.addWidget(self.vend_axis0, 1, 5, 1, 1)

        self.partition_style_axis0 = QComboBox(self.groupBox_3)
        self.partition_style_axis0.addItem("")
        self.partition_style_axis0.addItem("")
        self.partition_style_axis0.setObjectName(u"partition_style_axis0")

        self.gridLayout_6.addWidget(self.partition_style_axis0, 1, 8, 1, 1)

        self.vbeg_axis0 = QDoubleSpinBox(self.groupBox_3)
        self.vbeg_axis0.setObjectName(u"vbeg_axis0")
        self.vbeg_axis0.setEnabled(False)
        self.vbeg_axis0.setMinimumSize(QSize(0, 20))
        self.vbeg_axis0.setDecimals(6)
        self.vbeg_axis0.setMinimum(-10000.000000000000000)
        self.vbeg_axis0.setMaximum(10000.000000000000000)
        self.vbeg_axis0.setValue(0.002700000000000)

        self.gridLayout_6.addWidget(self.vbeg_axis0, 1, 3, 1, 1)

        self.rb_end_axis0 = QRadioButton(self.groupBox_3)
        self.rb_end_axis0.setObjectName(u"rb_end_axis0")
        self.rb_end_axis0.setMaximumSize(QSize(20, 16777215))

        self.gridLayout_6.addWidget(self.rb_end_axis0, 1, 4, 1, 1)

        self.rb_beg_axis1 = QRadioButton(self.groupBox_3)
        self.rb_beg_axis1.setObjectName(u"rb_beg_axis1")
        self.rb_beg_axis1.setMaximumSize(QSize(20, 16777215))

        self.gridLayout_6.addWidget(self.rb_beg_axis1, 2, 2, 1, 1)

        self.btn_compute_qpartition = QPushButton(self.groupBox_3)
        self.btn_compute_qpartition.setObjectName(u"btn_compute_qpartition")

        self.gridLayout_6.addWidget(self.btn_compute_qpartition, 2, 9, 1, 1)


        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.groupBox_6 = QGroupBox(self.layoutWidget2)
        self.groupBox_6.setObjectName(u"groupBox_6")
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        sizePolicy12.setHeightForWidth(self.groupBox_6.sizePolicy().hasHeightForWidth())
        self.groupBox_6.setSizePolicy(sizePolicy12)
        self.gridLayout_7 = QGridLayout(self.groupBox_6)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.output_method = QComboBox(self.groupBox_6)
        self.output_method.addItem("")
        self.output_method.addItem("")
        self.output_method.addItem("")
        self.output_method.setObjectName(u"output_method")
        sizePolicy1.setHeightForWidth(self.output_method.sizePolicy().hasHeightForWidth())
        self.output_method.setSizePolicy(sizePolicy1)

        self.gridLayout_7.addWidget(self.output_method, 0, 2, 1, 1)

        self.pushButton = QPushButton(self.groupBox_6)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout_7.addWidget(self.pushButton, 0, 3, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(172, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_5, 0, 0, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(171, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_4, 0, 4, 1, 1)


        self.verticalLayout_2.addWidget(self.groupBox_6)

        self.splitter.addWidget(self.layoutWidget2)

        self.gridLayout_16.addWidget(self.splitter, 0, 0, 1, 1)

        SimpleMask.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(SimpleMask)
        self.statusbar.setObjectName(u"statusbar")
        SimpleMask.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.plot_log, self.plot_index)
        QWidget.setTabOrder(self.plot_index, self.infobar)

        self.retranslateUi(SimpleMask)
        self.mask_list_include.toggled.connect(self.mask_list_radius.setEnabled)
        self.mask_list_include.toggled.connect(self.mask_list_variation.setEnabled)
        self.rb_beg_axis0.toggled.connect(self.vbeg_axis0.setEnabled)
        self.rb_beg_axis1.toggled.connect(self.vbeg_axis1.setEnabled)
        self.rb_end_axis0.toggled.connect(self.vend_axis0.setEnabled)
        self.rb_end_axis1.toggled.connect(self.vend_axis1.setEnabled)

        self.metaTab.setCurrentIndex(0)
        self.MaskWidget.setCurrentIndex(4)


        QMetaObject.connectSlotsByName(SimpleMask)
    # setupUi

    def retranslateUi(self, SimpleMask):
        SimpleMask.setWindowTitle(QCoreApplication.translate("SimpleMask", u"SimpleMask(beta)", None))
        self.groupBox.setTitle(QCoreApplication.translate("SimpleMask", u"Input", None))
        self.cb_beamline.setItemText(0, QCoreApplication.translate("SimpleMask", u"APS-8ID-I", None))
        self.cb_beamline.setItemText(1, QCoreApplication.translate("SimpleMask", u"APS-9ID-C", None))
        self.cb_beamline.setItemText(2, QCoreApplication.translate("SimpleMask", u"APS-12ID-B", None))
        self.cb_beamline.setItemText(3, QCoreApplication.translate("SimpleMask", u"NativeTypes", None))

        self.label_9.setText(QCoreApplication.translate("SimpleMask", u"begin index:", None))
        self.label_28.setText(QCoreApplication.translate("SimpleMask", u"num_frames", None))
        self.btn_load.setText(QCoreApplication.translate("SimpleMask", u"load data", None))
        self.label_17.setText(QCoreApplication.translate("SimpleMask", u"Scattering File:", None))
        self.fname.setPlaceholderText(QCoreApplication.translate("SimpleMask", u"filename", None))
        self.btn_select_raw.setText(QCoreApplication.translate("SimpleMask", u"...", None))
        self.label_4.setText(QCoreApplication.translate("SimpleMask", u"energy (keV):", None))
        self.label_7.setText(QCoreApplication.translate("SimpleMask", u"detector shape:", None))
        self.label_6.setText(QCoreApplication.translate("SimpleMask", u"pixel size (mm):", None))
        self.label_2.setText(QCoreApplication.translate("SimpleMask", u"center x:", None))
        self.label_5.setText(QCoreApplication.translate("SimpleMask", u"detector distance (mm):", None))
        self.label_21.setText(QCoreApplication.translate("SimpleMask", u"detector pitch (deg):", None))
        self.label_3.setText(QCoreApplication.translate("SimpleMask", u"center y:", None))
        self.metaTab.setTabText(self.metaTab.indexOf(self.tab), QCoreApplication.translate("SimpleMask", u"Transmission", None))
        self.label_51.setText(QCoreApplication.translate("SimpleMask", u"center x:", None))
        self.btn_swapxy_3.setText(QCoreApplication.translate("SimpleMask", u"<>", None))
        self.label_53.setText(QCoreApplication.translate("SimpleMask", u"center y:", None))
        self.label_52.setText(QCoreApplication.translate("SimpleMask", u"energy (keV):", None))
        self.label_55.setText(QCoreApplication.translate("SimpleMask", u"detector distance (mm):", None))
        self.label_49.setText(QCoreApplication.translate("SimpleMask", u"pixel size (mm):", None))
        self.label_50.setText(QCoreApplication.translate("SimpleMask", u"detector shape:", None))
        self.label_54.setText(QCoreApplication.translate("SimpleMask", u"incident angle (deg):", None))
        self.btn_find_center_3.setText(QCoreApplication.translate("SimpleMask", u"Find Center", None))
        self.btn_update_parameters_3.setText(QCoreApplication.translate("SimpleMask", u"Update Parameters", None))
        self.metaTab.setTabText(self.metaTab.indexOf(self.tab_8), QCoreApplication.translate("SimpleMask", u"Reflection", None))
        self.btn_swapxy.setText(QCoreApplication.translate("SimpleMask", u"Swap x-y", None))
        self.btn_find_center.setText(QCoreApplication.translate("SimpleMask", u"Find Center", None))
        self.btn_update_parameters.setText(QCoreApplication.translate("SimpleMask", u"Update Parameters", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("SimpleMask", u"Plot", None))
        self.plot_center.setText(QCoreApplication.translate("SimpleMask", u"show center", None))
        self.plot_log.setText(QCoreApplication.translate("SimpleMask", u"log scale", None))
        self.plot_cmap.setItemText(0, QCoreApplication.translate("SimpleMask", u"jet", None))
        self.plot_cmap.setItemText(1, QCoreApplication.translate("SimpleMask", u"cool", None))
        self.plot_cmap.setItemText(2, QCoreApplication.translate("SimpleMask", u"ocean", None))
        self.plot_cmap.setItemText(3, QCoreApplication.translate("SimpleMask", u"prism", None))
        self.plot_cmap.setItemText(4, QCoreApplication.translate("SimpleMask", u"coolwarm", None))
        self.plot_cmap.setItemText(5, QCoreApplication.translate("SimpleMask", u"seismic", None))
        self.plot_cmap.setItemText(6, QCoreApplication.translate("SimpleMask", u"gray", None))
        self.plot_cmap.setItemText(7, QCoreApplication.translate("SimpleMask", u"viridis", None))
        self.plot_cmap.setItemText(8, QCoreApplication.translate("SimpleMask", u"inferno", None))
        self.plot_cmap.setItemText(9, QCoreApplication.translate("SimpleMask", u"plasma", None))
        self.plot_cmap.setItemText(10, QCoreApplication.translate("SimpleMask", u"magma", None))

        self.btn_plot.setText(QCoreApplication.translate("SimpleMask", u"Plot", None))
        self.plot_invert.setText(QCoreApplication.translate("SimpleMask", u"invert", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("SimpleMask", u"Mask", None))
        self.btn_mask_reset.setText(QCoreApplication.translate("SimpleMask", u"Reset", None))
        self.btn_mask_undo.setText(QCoreApplication.translate("SimpleMask", u"Undo", None))
        self.btn_mask_apply.setText(QCoreApplication.translate("SimpleMask", u"Apply", None))
        self.btn_mask_redo.setText(QCoreApplication.translate("SimpleMask", u"Redo", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("SimpleMask", u"Blemish File:", None))
        self.label_30.setText(QCoreApplication.translate("SimpleMask", u"Blemish file:", None))
        self.btn_select_blemish.setText(QCoreApplication.translate("SimpleMask", u"Select", None))
        self.label_31.setText(QCoreApplication.translate("SimpleMask", u"HDF path:", None))
        self.blemish_path.setText(QCoreApplication.translate("SimpleMask", u"/lambda_pre_mask", None))
        self.btn_apply_blemish.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("SimpleMask", u"Additional File (hdf/tiff/mat):", None))
        self.label_18.setText(QCoreApplication.translate("SimpleMask", u"File name:", None))
        self.maskfile_fname.setText("")
        self.btn_select_maskfile.setText(QCoreApplication.translate("SimpleMask", u"Select", None))
        self.label_32.setText(QCoreApplication.translate("SimpleMask", u"HDF path:", None))
        self.maskfile_path.setText(QCoreApplication.translate("SimpleMask", u"/xpcs/mask", None))
        self.btn_apply_maskfile.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.MaskWidget.setTabText(self.MaskWidget.indexOf(self.tab_6), QCoreApplication.translate("SimpleMask", u"Blemish", None))
        self.groupBox_21.setTitle(QCoreApplication.translate("SimpleMask", u"Settings", None))
        self.cb_selector_mode.setItemText(0, QCoreApplication.translate("SimpleMask", u"exclusive", None))
        self.cb_selector_mode.setItemText(1, QCoreApplication.translate("SimpleMask", u"inclusive", None))

        self.btn_mask_draw_add.setText(QCoreApplication.translate("SimpleMask", u"Add", None))
        self.btn_mask_draw_evaluate.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.label_22.setText(QCoreApplication.translate("SimpleMask", u"type:", None))
        self.plot_width.setSpecialValueText("")
        self.label_14.setText(QCoreApplication.translate("SimpleMask", u"color:", None))
        self.label_23.setText(QCoreApplication.translate("SimpleMask", u"shape:", None))
        self.label_8.setText(QCoreApplication.translate("SimpleMask", u"linewidth:", None))
        self.cb_selector_type.setItemText(0, QCoreApplication.translate("SimpleMask", u"Circle", None))
        self.cb_selector_type.setItemText(1, QCoreApplication.translate("SimpleMask", u"Polygon", None))
        self.cb_selector_type.setItemText(2, QCoreApplication.translate("SimpleMask", u"Ellipse", None))
        self.cb_selector_type.setItemText(3, QCoreApplication.translate("SimpleMask", u"Rectangle", None))

        self.cb_selector_color.setItemText(0, QCoreApplication.translate("SimpleMask", u"red", None))
        self.cb_selector_color.setItemText(1, QCoreApplication.translate("SimpleMask", u"green", None))
        self.cb_selector_color.setItemText(2, QCoreApplication.translate("SimpleMask", u"yellow", None))
        self.cb_selector_color.setItemText(3, QCoreApplication.translate("SimpleMask", u"blue", None))
        self.cb_selector_color.setItemText(4, QCoreApplication.translate("SimpleMask", u"cyan", None))
        self.cb_selector_color.setItemText(5, QCoreApplication.translate("SimpleMask", u"magenta", None))
        self.cb_selector_color.setItemText(6, QCoreApplication.translate("SimpleMask", u"black", None))
        self.cb_selector_color.setItemText(7, QCoreApplication.translate("SimpleMask", u"white", None))

        self.MaskWidget.setTabText(self.MaskWidget.indexOf(self.tab_2), QCoreApplication.translate("SimpleMask", u"Draw", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("SimpleMask", u"Global histogram", None))
        self.btn_mask_threshold_evaluate.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.binary_scale.setItemText(0, QCoreApplication.translate("SimpleMask", u"linear scale", None))
        self.binary_scale.setItemText(1, QCoreApplication.translate("SimpleMask", u"log scale", None))

        self.label_26.setText(QCoreApplication.translate("SimpleMask", u"low:", None))
        self.label_27.setText(QCoreApplication.translate("SimpleMask", u"high:", None))
        self.groupBox_19.setTitle(QCoreApplication.translate("SimpleMask", u"Local: ring azimuthal average", None))
        self.label_16.setText(QCoreApplication.translate("SimpleMask", u"num. circular ROI:", None))
        self.label_15.setText(QCoreApplication.translate("SimpleMask", u"cutoff (\u00b1std):", None))
        self.btn_mask_outlier_evaluate.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.groupBox_20.setTitle(QCoreApplication.translate("SimpleMask", u"Local:", None))
        self.label_35.setText(QCoreApplication.translate("SimpleMask", u"num. circular ROI:", None))
        self.label_40.setText(QCoreApplication.translate("SimpleMask", u"cutoff (\u00b1std):", None))
        self.btn_mask_outlier_evaluate_2.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.MaskWidget.setTabText(self.MaskWidget.indexOf(self.tab_5), QCoreApplication.translate("SimpleMask", u"Threshold", None))
        self.groupBox_12.setTitle(QCoreApplication.translate("SimpleMask", u"Import from a file", None))
        self.mask_list_1based.setText(QCoreApplication.translate("SimpleMask", u"1-based", None))
        self.mask_list_rowcol.setText(QCoreApplication.translate("SimpleMask", u"row-col", None))
        self.btn_mask_list_load.setText(QCoreApplication.translate("SimpleMask", u"Load File", None))
        self.groupBox_11.setTitle(QCoreApplication.translate("SimpleMask", u"xy list", None))
        self.btn_mask_list_clear.setText(QCoreApplication.translate("SimpleMask", u"Clear List", None))
        self.btn_mask_list_evaluate.setText(QCoreApplication.translate("SimpleMask", u"Evaluate", None))
        self.groupBox_13.setTitle(QCoreApplication.translate("SimpleMask", u"Input coordinates", None))
        self.btn_mask_list_add.setText(QCoreApplication.translate("SimpleMask", u"Add points", None))
        self.mask_list_input.setText("")
        self.mask_list_input.setPlaceholderText(QCoreApplication.translate("SimpleMask", u"(x1, y1), (x2, y2), ...", None))
        self.groupBox_14.setTitle(QCoreApplication.translate("SimpleMask", u"Select with mouse double-click", None))
        self.mask_list_include.setText(QCoreApplication.translate("SimpleMask", u"Include points with simiar intensity", None))
        self.label_19.setText(QCoreApplication.translate("SimpleMask", u"Radius:", None))
        self.label_24.setText(QCoreApplication.translate("SimpleMask", u"Intensity Variation (%):", None))
        self.MaskWidget.setTabText(self.MaskWidget.indexOf(self.tab_3), QCoreApplication.translate("SimpleMask", u"Manual", None))
        self.groupBox_10.setTitle(QCoreApplication.translate("SimpleMask", u"Binary operation on the mask", None))
        self.pushButton_5.setText(QCoreApplication.translate("SimpleMask", u"erode", None))
        self.pushButton_9.setText(QCoreApplication.translate("SimpleMask", u"dilate", None))
        self.pushButton_16.setText(QCoreApplication.translate("SimpleMask", u"open", None))
        self.pushButton_15.setText(QCoreApplication.translate("SimpleMask", u"close", None))
        self.pushButton_2.setText(QCoreApplication.translate("SimpleMask", u"fill holes", None))
        self.MaskWidget.setTabText(self.MaskWidget.indexOf(self.tab_7), QCoreApplication.translate("SimpleMask", u"Binary", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("SimpleMask", u"Scattering, Mask and Partitions", None))
        self.label_11.setText(QCoreApplication.translate("SimpleMask", u"coordinates:", None))
        self.plot_index.setItemText(0, QCoreApplication.translate("SimpleMask", u"scattering", None))
        self.plot_index.setItemText(1, QCoreApplication.translate("SimpleMask", u"scattering * mask", None))
        self.plot_index.setItemText(2, QCoreApplication.translate("SimpleMask", u"mask", None))
        self.plot_index.setItemText(3, QCoreApplication.translate("SimpleMask", u"dynamic_q_partition", None))
        self.plot_index.setItemText(4, QCoreApplication.translate("SimpleMask", u"static_q_partition", None))
        self.plot_index.setItemText(5, QCoreApplication.translate("SimpleMask", u"preview", None))

        self.groupBox_3.setTitle(QCoreApplication.translate("SimpleMask", u"Partition", None))
        self.label_13.setText(QCoreApplication.translate("SimpleMask", u"end", None))
        self.rb_beg_axis0.setText("")
        self.label_20.setText(QCoreApplication.translate("SimpleMask", u"Style", None))
        self.label_33.setText(QCoreApplication.translate("SimpleMask", u"unit", None))
        self.unit_axis0.setText(QCoreApplication.translate("SimpleMask", u"unit", None))
        self.partition_style_axis1.setItemText(0, QCoreApplication.translate("SimpleMask", u"linear", None))
        self.partition_style_axis1.setItemText(1, QCoreApplication.translate("SimpleMask", u"logarithmic", None))

        self.unit_axis1.setText(QCoreApplication.translate("SimpleMask", u"unit", None))
        self.label_39.setText(QCoreApplication.translate("SimpleMask", u"qmap", None))
        self.label.setText(QCoreApplication.translate("SimpleMask", u"static bins", None))
        self.label_12.setText(QCoreApplication.translate("SimpleMask", u"begin", None))
        self.rb_end_axis1.setText("")
        self.label_10.setText(QCoreApplication.translate("SimpleMask", u"dynamic bins", None))
        self.partition_style_axis0.setItemText(0, QCoreApplication.translate("SimpleMask", u"linear", None))
        self.partition_style_axis0.setItemText(1, QCoreApplication.translate("SimpleMask", u"logarithmic", None))

        self.rb_end_axis0.setText("")
        self.rb_beg_axis1.setText("")
        self.btn_compute_qpartition.setText(QCoreApplication.translate("SimpleMask", u"compute", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("SimpleMask", u"Output", None))
        self.output_method.setItemText(0, QCoreApplication.translate("SimpleMask", u"nexus", None))
        self.output_method.setItemText(1, QCoreApplication.translate("SimpleMask", u"numpy", None))
        self.output_method.setItemText(2, QCoreApplication.translate("SimpleMask", u"tensor", None))

        self.pushButton.setText(QCoreApplication.translate("SimpleMask", u"save", None))
    # retranslateUi

