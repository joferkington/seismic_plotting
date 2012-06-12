from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib as mpl
import geoprobe

import utilities

class Map(object):
    def __init__(self, ax, vol, horizonname=None, extents=None, name=''):
        if extents is None:
            extents = [vol.xmin, vol.xmax, vol.ymin, vol.ymax]
        self.name = name
        self.extents = extents
        self.vol = vol
        self.ax = ax
        if horizonname is not None:
            self.base_hor = geoprobe.horizon(horizonname)
            self.base_hor.grid_extents = extents
            self.baseim = self.plot_horizon(self.base_hor)
        self.ax.axis(extents)
        self.ax.set_aspect(self.vol.dyW / self.vol.dxW)
        self.ax.set_title(name)
        self.bounds = utilities.extents_to_poly(*extents)

    def plot_horizon(self, hor, colormap=None):
        if isinstance(hor, basestring):
            hor = geoprobe.horizon(hor)
        if colormap is None:
            colormap = mpl.cm.jet
        hor.grid_extents = self.extents
        data = -hor.grid
        ls = utilities.Shader(azdeg=315, altdeg=45)
        rgb = ls.shade(data, cmap=colormap, mode='overlay')
        im = self.ax.imshow(rgb, extent=self.extents, origin='lower')
        return im

    def plot_scalebar(self, **kwargs):
        length = kwargs.pop('length', 1)
        title = kwargs.pop('title', '{} world units'.format(length))
        dat_bar_length = length / self.vol.dxW
        
        kwargs['loc'] = kwargs.get('loc', 1)
        kwargs['pad'] = kwargs.get('pad', 0.5)
        kwargs['borderpad'] = kwargs.get('borderpad', 0.5)
        kwargs['sep'] = kwargs.get('sep', 5)
        kwargs['frameon'] = kwargs.get('frameon', True)
        self.sizebar = AnchoredSizeBar(self.ax.transData, dat_bar_length, title, 
                                       **kwargs)
        self.ax.add_artist(self.sizebar)
        return self.sizebar


