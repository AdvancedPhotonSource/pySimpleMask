import pyqtgraph as pg
import numpy as np

# Images are indexed (row, col); match pyqtgraph's display orientation.
pg.setConfigOptions(imageAxisOrder="row-major")


class ImageViewROI(pg.ImageView):
    def __init__(self, *arg, **kwargs):
        super(ImageViewROI, self).__init__(*arg, **kwargs)
        self.removeItem(self.roi)
        self.roi = {}
        self.roi_idx = 0
        self.ui.roiBtn.setDisabled(True)
        self.ui.menuBtn.setDisabled(True)
        # Hide the ROI/time-series plot panel — it was only used when the
        # ImageView held a 3D stack with a time axis.  Now we feed 2D slices
        # directly, so the panel is empty and wastes vertical space.
        self.ui.roiPlot.hide()

    def roiClicked(self):
        pass

    def roiChanged(self):
        pass

    def adjust_viewbox(self):
        vb = self.getView()
        vb.setMouseMode(vb.RectMode)
        vb.setAspectLocked(1.0)
        # Do NOT set absolute xMin/xMax/yMin/yMax limits: those lock the view
        # to the range captured at call time and break zoom-out and window
        # resize (pyqtgraph cannot auto-range past the stale limits).
        # Only cap zoom-in to 1/50 of the image size, derived from the image
        # dimensions rather than from the current (possibly zoomed) view range.
        if self.image is not None:
            h = self.image.shape[-2]
            w = self.image.shape[-1]
            # xMin/xMax/yMin/yMax are position limits (pan bounds).
            # Setting them to ±1 image dimension means an image edge can reach
            # at most the centre of the view — the image never fully leaves the
            # canvas even on fast trackpad swipes.
            # maxXRange/maxYRange = 2× caps zoom-out so the image is never
            # smaller than 25% of the view.
            # minXRange/minYRange caps zoom-in to 1/50 of image size.
            # All values are derived from the image dimensions, NOT from the
            # current view range, so window resize and repeated plot() calls
            # cannot lock the view into a stale zoomed-in state.
            vb.setLimits(
                xMin=-w, xMax=2 * w, yMin=-h, yMax=2 * h,
                minXRange=max(w / 50, 1), minYRange=max(h / 50, 1),
                maxXRange=w * 2,          maxYRange=h * 2,
            )

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
        seg_len = d.length()
        ra = d.angle(pg.Point(1, 0), units="radians")
        c = pg.Point(width/2. * np.sin(ra), -width/2. * np.cos(ra))
        pos1 = pos1 + c

        pg.ROI.__init__(self, pos1, size=pg.Point(seg_len, width),
                        angle=np.rad2deg(ra), **args)
        # self.addScaleRotateHandle([0, 0.5], [1, 0.5])
        self.addScaleRotateHandle([1, 0.5], [0, 0])
        # self.addScaleHandle([0.5, 1], [0.5, 0.5])
