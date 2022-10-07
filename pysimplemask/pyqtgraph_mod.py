import pyqtgraph as pg
from .mpl_cmaps_in_ImageItem import pg_get_cmap
import matplotlib.pyplot as plt
import numpy as np


class ImageViewROI(pg.ImageView):
    def __init__(self, *arg, **kwargs):
        super(ImageViewROI, self).__init__(*arg, **kwargs)
        self.removeItem(self.roi)
        self.roi = {}
        self.roi_idx = 0
        self.ui.roiBtn.setDisabled(True)
        self.ui.menuBtn.setDisabled(True)

    def roiClicked(self):
        pass

    def roiChanged(self):
        pass

    def adjust_viewbox(self):
        vb = self.getView()
        xMin, xMax = vb.viewRange()[0]
        yMin, yMax = vb.viewRange()[1]

        vb.setLimits(xMin=xMin,
                     xMax=xMax,
                     yMin=yMin,
                     yMax=yMax,
                     minXRange=(xMax - xMin) / 50,
                     minYRange=(yMax - yMin) / 50)
        vb.setMouseMode(vb.RectMode)
        vb.setAspectLocked(1.0)

    def reset_limits(self):
        """
        reset the viewbox's limits so updating image won't break the layout;
        """
        self.view.state['limits'] = {'xLimits': [None, None],
                                     'yLimits': [None, None],
                                     'xRange': [None, None],
                                     'yRange': [None, None]
                                     }

    def set_colormap(self, cmap):
        pg_cmap = pg.colormap.getFromMatplotlib(cmap)
        # pg_cmap = pg_get_cmap(plt.get_cmap(cmap))
        self.setColorMap(pg_cmap)
    
    def remove_rois(self, filter_str=None):
        # if filter_str is None; then remove all rois
        keys = list(self.roi.keys()).copy()
        if filter_str is not None:
            keys = list(filter(lambda x: x.startswith(filter_str), keys))
        for key in keys:
            self.remove_item(key)

    def clear(self):
        self.remove_rois()
        self.roi = {}
        super(ImageViewROI, self).clear()
        self.reset_limits()
        # incase the signal isn't connected to anything.
        # try:
        #     self.scene.sigMouseMoved.disconnect()
        # except:
        #     pass

    def add_item(self, t, label=None):
        if label is None:
            label = f"roi_{self.roi_idx:06d}"
            self.roi_idx += 1

        if label in self.roi:
            self.remove_item(label)

        self.roi[label] = t
        self.addItem(t)
        return label

    def remove_item(self, label):
        t = self.roi.pop(label, None)
        if t is not None:
            self.removeItem(t)

    def updateImage(self, autoHistogramRange=True):
        # Redraw image on screen
        if self.image is None:
            return

        image = self.getProcessedImage()

        lmin = np.min(image[self.currentIndex])
        lmax = np.max(image[self.currentIndex])
        if autoHistogramRange:
            # self.ui.histogram.setHistogramRange(lmin, lmax)
            self.setLevels(rgba=[(lmin, lmax)])

        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == 'col-major':
            axorder = ['t', 'x', 'y', 'c']
        else:
            axorder = ['t', 'y', 'x', 'c']
        axorder = [self.axes[ax]
                   for ax in axorder if self.axes[ax] is not None]
        image = image.transpose(axorder)

        # Select time index
        if self.axes['t'] is not None:
            self.ui.roiPlot.show()
            image = image[self.currentIndex]

        self.imageItem.updateImage(image)


class LineROI(pg.ROI):
    r"""
    Rectangular ROI subclass with scale-rotate handles on either side. This
    allows the ROI to be positioned as if moving the ends of a line segment.
    A third handle controls the width of the ROI orthogonal to its "line" axis.
    
    ============== =============================================================
    **Arguments**
    pos1           (length-2 sequence) The position of the center of the ROI's
                   left edge.
    pos2           (length-2 sequence) The position of the center of the ROI's
                   right edge.
    width          (float) The width of the ROI.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """
    def __init__(self, pos1, pos2, width, **args):
        pos1 = pg.Point(pos1)
        pos2 = pg.Point(pos2)
        d = pos2-pos1
        l = d.length()
        ra = d.angle(pg.Point(1, 0), units="radians")
        c = pg.Point(width/2. * np.sin(ra), -width/2. * np.cos(ra))
        pos1 = pos1 + c
        
        pg.ROI.__init__(self, pos1, size=pg.Point(l, width), 
                        angle=np.rad2deg(ra), **args)
        # self.addScaleRotateHandle([0, 0.5], [1, 0.5])
        self.addScaleRotateHandle([1, 0.5], [0, 0])
        # self.addScaleHandle([0.5, 1], [0.5, 0.5])
