import numpy as np
import geoprobe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage
import shapely.geometry
import itertools
import csv
import os
import glob

class SectionManager(object):
    def __init__(self, vol, colormap, ve=2.0, horizons=None, horizon_styles=None, 
                 wells=None, well_marker_style=None, section_line_style=None, 
                 well_annotation_style=None, well_threshold=5, name='', 
                 resample_factor=2):
        # If horizon_styles is just a single dict, pass in a list of it
        if isinstance(horizon_styles, dict):
            horizon_styles = [horizon_styles]
        if horizons is not None:
            horizons = HorizonSet(horizons, horizon_styles)

        if well_marker_style is None:
            well_marker_style = dict(marker='o', ls='none', 
                                   color='green', markersize=10)
        if section_line_style is None:
            section_line_style = dict(marker='o')
        if well_annotation_style is None:
            well_annotation_style = dict(xytext=(0, 10), ha='center',
                                       textcoords='offset points')

        self.section_line_style = section_line_style
        self.well_marker_style = well_marker_style
        self.well_annotation_style = well_annotation_style
        self.wells = WellSet(wells, well_threshold)
        self.ve = ve
        self.name = name
        self.resample_factor = resample_factor
        self.vol = geoprobe.volume(vol)
        self.colormap = geoprobe.colormap(colormap)
        self.horizons = horizons
        self.sections = []
        self.maps = []

    def setup_figure(self, **kwargs):
        kwargs['figsize'] = kwargs.get('figsize', (16,13))
        return plt.figure(**kwargs)

    def setup_axis(self, fig, **kwargs):
        return fig.add_subplot(111, **kwargs)

    def add_map(self, horizon=None, extents=None, annotate=False, ax=None,
                name=None):
        if ax is None:
            ax = self.setup_axis(self.setup_figure())
        mapview = Map(ax, self.vol, horizon, extents, name=name)
        mapview.ax.autoscale(False)
        self.maps.append(mapview)
        #mapview.plot_scalebar(barlen)
        if self.wells is not None:
            self.wells.plot_on_map(mapview, **self.well_marker_style)
            if annotate:
                self.wells.annotate_map(mapview, **self.well_annotation_style)
        return mapview

    def _additional_plotting(self, sec, plot_wells=True, plot_horizons=True, 
            plot_on_map=True, **kwargs):
        sec.ax.set_title(sec.name)
        sec.ax.set_xlabel('Distance (meters) V.E.: %0.1f' % sec.ve)
        self.sections.append(sec.name)
        sec.plot_image(**kwargs)
        if self.horizons is not None and plot_horizons:
            sec.plot_horizons(self.horizons)
        if self.wells is not None and plot_wells:
            self.wells.plot_on_section(sec)
        if plot_on_map:
            for mapview in self.maps:
                sec.plot_on_map(mapview, **self.section_line_style)
        return sec

    def add_section(self, x, y, zmin=None, zmax=None, name='Cross Section',
                    resample_factor=None, ax=None, colormap=None, style=None,
                    **kwargs):
        if resample_factor is None:
            resample_factor = self.resample_factor
        if ax is None:
            ax = self.setup_axis(self.setup_figure())
        if colormap is None:
            colormap = self.colormap.as_matplotlib

        if style is None:
            sec_type = Section
        else:
            sec_type = SketchSection

        sec = sec_type(self.vol, x, y, ax, colormap, zmin=zmin, zmax=zmax, 
                ve=self.ve, name=name, resample_factor=resample_factor)
        self._additional_plotting(sec, **kwargs)
        return sec

    def add_2d_line(self, line, zmin=None, zmax=None, name=None, 
                    colormap=None, resample_factor=None, ax=None, 
                    world_coords=False, **kwargs):
        if resample_factor is None:
            resample_factor = self.resample_factor
        if ax is None:
            ax = self.setup_axis(self.setup_figure())
        if colormap is None:
            colormap = self.colormap.as_matplotlib

        try:
            line = geoprobe.data2d(line)
        except ValueError:
            pass

        if name is None:
            name = line.name

        if world_coords:
            line.x, line.y = self.vol.world2model(line.x, line.y)
            
        dx, dy = self.vol.dxW, self.vol.dyW
        sec = Section2D(line, ax, colormap, zmin=zmin, zmax=zmax, dx=dx, 
                        dy=dy, ve=self.ve, resample_factor=resample_factor)
        self._additional_plotting(sec, **kwargs)
        return sec

    def add_inline(self, xval, **kwargs):
        ymin = kwargs.pop('ymin', self.vol.ymin)
        ymax = kwargs.pop('ymax', self.vol.ymax)
        kwargs['name'] = kwargs.get('name', 'Inline {}'.format(int(xval)))

        sec = self.add_section([xval, xval], [ymin, ymax], **kwargs)
        locator = ScaledLocator(self.vol.dyW, ymin)
        formatter = ScaledFormatter(self.vol.dyW, ymin)
        sec.ax.xaxis.set_major_locator(locator)
        sec.ax.xaxis.set_major_formatter(formatter)
        sec.ax.set_xlabel('Crossline V.E.:%0.1f' % sec.ve)
        return sec
 
    def add_crossline(self, yval, **kwargs):
        xmin = kwargs.pop('xmin', self.vol.xmin)
        xmax = kwargs.pop('xmax', self.vol.xmax)
        kwargs['name'] = kwargs.get('name', 'Crossline {}'.format(int(yval)))

        sec = self.add_section([xmin, xmax], [yval, yval], **kwargs)
        locator = ScaledLocator(self.vol.dxW, xmin)
        formatter = ScaledFormatter(self.vol.dxW, xmin)
        sec.ax.xaxis.set_major_locator(locator)
        sec.ax.xaxis.set_major_formatter(formatter)
        sec.ax.set_xlabel('Inline V.E.:%0.1f' % sec.ve)
        return sec

    def save(self, prefix='', suffix='.pdf', **kwargs):
        for i, mapview in enumerate(self.maps):
            name = '{}{}_map_{}{}'.format(prefix, mapview.name, i, suffix)
            mapview.ax.figure.savefig(name, **kwargs)
        for i, sec in enumerate(self.sections):
            name = '{}{}_section_{}{}'.format(prefix, sec.name, i, suffix)
            sec.ax.figure.savefig(name, **kwargs)

    def show(self):
        plt.show()

class AutoManager(SectionManager):
    basedir = None
    default_vol = None
    default_colormap = None
    default_well_list = None
    horizon_dir = None
    def __init__(self, **kwargs):
        kwargs['vol'] = kwargs.get('vol', self.default_vol)
        kwargs['colormap'] = kwargs.get('colormap', self.default_colormap)

        well_list = kwargs.pop('well_list', self.default_well_list)
        if well_list is not None:
            well_list = WellDatabase(well_list)
            wells = kwargs.get('wells', well_list.keys())
            kwargs['wells'] = [well_list[name] for name in wells]

        horizons = kwargs.get('horizons', [])
        horizons = [os.path.join(self.horizon_dir, item) for item in horizons]
        kwargs['horizons'] = horizons

        SectionManager.__init__(self, **kwargs)

class GeoprobeManager(AutoManager):
    def __init__(self, basedir, **kwargs):
        self.basedir = basedir
        self.horizon_dir = os.path.join(self.basedir, 'Horizons')


class ExampleSections(AutoManager):
    basedir = '/data/MyCode/Geoprobe/distribution/examples/data/'
    default_vol = basedir + 'Volumes/example.vol'
    default_colormap = basedir + 'Colormaps/blue-orange-tweaked.colormap'
    default_well_list = '/data/nankai/data/well_locations_with_coords.csv'
    horizon_dir = basedir + 'Horizons/'

class NankaiSections(ExampleSections):
    basedir = '/data/nankai/data/'
    default_vol = basedir + 'Volumes/kumdep_1500_meter_agc.hdf'
    default_colormap = basedir + 'Colormaps/AGC'
    default_well_list = '/data/nankai/data/well_locations_with_coords.csv'
    horizon_dir = basedir + 'Horizons/'

class Section(object):
    def __init__(self, vol, x, y, ax, colormap, zmin=None, zmax=None, ve=2.0, 
                 name='Cross Section', resample_factor=2):
        self.ax = ax
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.vol = vol
        self.dxw, self.dyw = self.vol.dxW, self.vol.dyW
        if zmin is None:
            zmin = self.vol.zmin
        if zmax is None:
            zmax = self.vol.zmax
        self.zmin = max(self.vol.zmin, zmin)
        self.zmax = min(self.vol.zmax, zmax)
        self.ve = ve
        self.colormap = colormap
        self.resample_factor = resample_factor
        self.name = name

        self.horizon_lines = None
        self.loc_line = None

    def extract_section(self, vol=None):
        if vol is None:
            vol = self.vol
        data, xi, yi = vol.extract_section(self.x, self.y, self.zmin, self.zmax)
        data = self.make_raw_image(data)
        distance = self.calculate_distance_along_section(xi, yi)
        extent = [distance.min(), distance.max(), self.zmin, self.zmax]
        return data, extent

    def update_position(self, x, y):
        self.x, self.y = x, y
        data, extent = self.extract_section()
        self.im.set_data(data)
        self.im.set_extent(extent)

    def update_horizons(self, hor_set):
        for line, hor in zip(self.horizon_lines, hor_set.horizons):
            x, y = self.slice_horizon(hor)
            line.set_data(x, y)

    @property
    def line(self):
        return shapely.geometry.LineString(zip(self.x, self.y))

    def calculate_distance_along_section(self, xi, yi):
        start, _ = self.project_onto(xi[0], yi[0])
        distance = np.hypot(self.dxw * np.diff(xi), 
                            self.dyw * np.diff(yi))
        distance = np.cumsum(np.r_[start, distance])
        return distance

    def plot_image(self, **kwargs):
        data, extent = self.extract_section()
        self.im = self.ax.imshow(data, origin='lower', extent=extent, 
                interpolation='bilinear', aspect=self.ve, cmap=self.colormap,
                **kwargs)
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()
        return self.im

    def plot_scalebar(self, length=1, title=None):
        if title is None:
            title = '{} world units'.format(length)
        self.sizebar = AnchoredSizeBar(self.ax.transData, length,
                                       title, loc=1, pad=0.5, 
                                       borderpad=0.5, sep=5, frameon=True)
        self.ax.add_artist(self.sizebar)
        return self.sizebar

    def plot_on_map(self, mapview, **kwargs):
        kwargs.pop('label', None)
        kwargs['label'] = self.name
        self.loc_line, = mapview.ax.plot(self.x, self.y, **kwargs)
        return self.loc_line

    def seafloor_mute(self, seafloor, pad=0, color=(1.0, 1.0, 1.0), value=None):
        seafloor = geoprobe.horizon(seafloor)
        dist, z = self.slice_horizon(seafloor)
        z -= pad
        if value is not None:
            color = self.im.cmap(self.im.norm(value))
        collection = self.ax.fill_between(dist, self.zmin, z, facecolor=color, 
                                          edgecolor='none')
        return collection

    def mark_intersection(self, *args, **kwargs):
        if len(args) == 1:
            other = args[0]
            x, y = other.x, other.y
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError('Input must be either another section or x, y')
        x, _ = self.project_onto(x, y)
        return self.ax.axvline(x, **kwargs)

    def label_endpoints(self, template='XL: {}, IL: {}', **kwargs):
        kwargs.pop('xy', None)
        kwargs['textcoords'] = kwargs.get('textcoords', 'offset points')
        kwargs['xycoords'] = kwargs.get('xycoords', 'axes fraction')
        kwargs['xytext'] = kwargs.get('xytext', (15, 15))
        kwargs['arrowprops'] = kwargs.get('arrowprops', dict(arrowstyle='->'))
        self.endpoint_labels = []
        for limit, position in zip(self.ax.get_xlim(), ['left', 'right']):
            x, y = self.line.interpolate(limit).coords[0]
            kwargs['ha'] = position
            if position is 'left':
                kwargs['xy'] = (0, 0)
            if position is 'right':
                kwargs['xy'] = (1, 0)
                xtext, ytext = kwargs['xytext']
                kwargs['xytext'] = (-xtext, ytext)
            anno = self.ax.annotate(template.format(x,y), **kwargs)
            self.endpoint_labels.append(anno)
        return self.endpoint_labels

    def dip_rose(self, pos=(1, 1), values=None, width=40, **kwargs):
        """
        Plot a dip rose on the section.
        
        Parameters
        ----------
            `pos` : Position given as a tuple of x,y in axes fraction. The 
                default position is (1,1) -- The upper-right corner of the plot.
            `values` : A sequence of dips to plot. Defaults to 15 degree 
                increments between 0 and 75 (inclusive).
            `width` : The length of the 0 degree bar in points. The default is
                40 points.
            Additional keyword arguments are passed on to `annotate`.

        Returns
        -------
            A sequence of annotation objects.
        """
        def apparent_dip(theta, ve):
            theta = np.radians(theta)
            dx, dy = np.cos(theta), np.sin(theta)
            return np.arctan(ve * dy / dx)

        ve = self.ax.get_aspect()
        if values is None:
            values = range(0, 90, 15)

        x, y = pos
        dx = -1 if x > 0.5 else 1
        dy = -1 if y > 0.5 else 1

        artists = []
        for theta in values:
            app_theta = apparent_dip(theta, ve)
            x = width * np.cos(app_theta)
            y = width * np.sin(app_theta)
            ha = {1:'left', -1:'right'}[dx]

            props = dict(xy=pos, xytext=(dx * x, dy * y), 
                        xycoords='axes fraction', textcoords='offset points', 
                        va='center', ha=ha, rotation_mode='anchor',
                        rotation=dx*dy*np.degrees(app_theta), 
                        arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=4))
            kwargs.update(props)

            artists.append(self.ax.annotate(r'%i$^{\circ}$' % theta, **kwargs))
        return artists

    def project_onto(self, *args):
        """Returns the distance along the section to the point/line defined by
        *x*, *y* and the mininum distance between it and the section."""
        if len(args) == 1:
            # Assume a shapely geometry has been passed in
            other = args[0]
        else:
            try:
                x, y = args
            except ValueError:
                raise ValueError('Expecting a shapely geometry or x and y!')
            # Try to build a shapely geometry from x, y
            try:
                length = len(x)
            except TypeError:
                length = 1
            if length > 1:
                other = shapely.geometry.LineString(zip(x, y))
                other = self.line.intersection(other)
            else:
                other = shapely.geometry.Point(x, y)
        position = self.line.project(other, normalized=True)
        total_distance = np.hypot(self.dxw * np.diff(self.x), 
                                  self.dyw * np.diff(self.y)).sum()
        return position * total_distance, self.line.distance(other)

    def plot_horizons(self, hor_set):
        self.horizon_lines = hor_set.plot(self)

    def slice_horizon(self, hor):
        """Slices a geoprobe horizon along the section line. Returns a sequence
        of distances along the section and a sequence of z-values."""
        # Get only the portions of the section inside the horizon extents
        try:
            bounds = hor.bounds
        except AttributeError:  
            hor.bounds = extents_to_poly(*hor.grid_extents)
            bounds = hor.bounds

        if not self.line.intersects(bounds):
            return np.array([]), np.array([])

        inside = self.line.intersection(bounds)
        x, y = inside.xy

        # Extract the section
        xmin, xmax, ymin, ymax = hor.grid_extents
        x, y = x - xmin, y - ymin
        z, xi, yi = geoprobe.utilities.extract_section(hor.grid.T, x, y)

        # Put the distances back in to distance along the section line
        start, _ = self.project_onto(xi[0]+xmin, yi[0]+ymin)
        distance = np.hypot(self.dxw * np.diff(xi), 
                            self.dyw * np.diff(yi))
        z = np.ma.squeeze(z)
        distance = np.cumsum(np.r_[start, distance])
        distance = np.ma.masked_array(distance, z.mask)
        return distance, z

    def make_raw_image(self, data):
        if self.resample_factor != 1:
            data = scipy.ndimage.interpolation.zoom(data, 
                    (1, self.resample_factor), 
                    output=data.dtype, order=1, prefilter=False)
        return data.T

    def save(self, filename=None, **kwargs):
        if filename is None:
            filename = self.name + '.png'
        fig = self.ax.figure
        fig.savefig(filename, **kwargs)

class Section2D(Section):
    def __init__(self, line, ax, colormap, dx=1.0, dy=1.0, zmin=None, zmax=None, 
                 ve=2.0, name='2D Section', resample_factor=2):
        self.dxw, self.dyw = dx, dy
        self.x, self.y = line.x, line.y
        self.line2d = line
        self.ax = ax

        if zmin is None:
            zmin = self.line2d.zmin
        if zmax is None:
            zmax = self.line2d.zmax

        self.zmin = max(self.line2d.zmin, zmin)
        self.zmax = min(self.line2d.zmax, zmax)
        self.ve = ve
        self.colormap = colormap
        self.resample_factor = resample_factor
        self.name = name

    def extract_section(self):
        data = self.line2d.scaled_data
        zstart = np.searchsorted(self.line2d.z, self.zmin)
        zstop = np.searchsorted(self.line2d.z, self.zmax)
        data = data[:, zstart:zstop]

        data = self.make_raw_image(data)
        distance = self.calculate_distance_along_section(self.x, self.y)
        extent = [distance.min(), distance.max(), self.zmin, self.zmax]
        return data, extent


class HorizonSet(object):
    def __init__(self, horizons, styles=None):
        self.horizons = [geoprobe.horizon(hor) for hor in horizons]
        if styles is None:
            styles = [dict() for _ in horizons]
        self.styles = styles

    def plot(self, sec):
        styles = itertools.cycle(self.styles)
        def plot_item(hor, style):
            x, y = sec.slice_horizon(hor)
            # Skip first and last points...
            l, = sec.ax.plot(x, y, **style)
            return l
        limits = sec.ax.axis()
        lines = []
        for hor, style in zip(self.horizons, styles):
            lines.append(plot_item(hor, style))
        sec.ax.axis(limits)
        return lines

class Well(object):
    def __init__(self, x, y, name, top=0, bottom=0):
        self.x, self.y = x, y
        self.name = name
        self.top, self.bottom = top, bottom
        self.depth = self.bottom - self.top
        self.point = shapely.geometry.Point(x, y)

    def plot_on_section(self, section, threshold=1, ticks=True, 
                        tick_interval=100, text_kw=None, **kwargs):
        pos, dist = section.project_onto(self.point)
        if dist > threshold:
            return
        if text_kw is None:
            text_kw = {}
        kwargs['color'] = kwargs.get('color', 'black')
        section.ax.plot([pos, pos], [self.top, self.bottom], **kwargs)
        section.ax.annotate(self.name, (pos, self.top), xytext=(0, 12), 
                textcoords='offset points', ha='center', **text_kw)
        if ticks:
            for i in range(0, int(self.depth), tick_interval):
                section.ax.annotate(repr(i), (pos, self.top+i), xytext=(10, 0), 
                                    textcoords='offset points', va='center',
                                    arrowprops=dict(arrowstyle='-', shrinkB=0))

    def plot_on_map(self, mapview, **kwargs):
        kwargs.pop('label', None)
        kwargs['label'] = self.name
        p, = mapview.ax.plot(self.x, self.y, **kwargs)
        return p

    def annotate_map(self, mapview, **kwargs):
        return mapview.ax.annotate(self.name, xy=(self.x, self.y), **kwargs)

class WellDatabase(dict):
    def __init__(self, filename):
        dict.__init__(self)
        self.wells = {}
        with open(filename, 'rb') as infile:
            for line in csv.DictReader(infile):
                name = line['Hole']
                x, y = float(line['Inline']), float(line['Crossline'])
                seafloor = float(line['water_dept'])
                depth = float(line['penetrated'])
                well = Well(x, y, name, top=seafloor, bottom=seafloor+depth)
                self[name] = well

class WellSet(object):
    def __init__(self, wells, threshold=1):
        self.threshold = threshold
        self._dict = dict()
        for item in wells:
            self._dict[item.name] = item
            
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    @property
    def wells(self):
        return self._dict.values()

    def plot_on_section(self, section, **kwargs):
        kwargs['threshold'] = kwargs.get('threshold', self.threshold)
        for well in self.wells:
            well.plot_on_section(section, **kwargs)

    def plot_on_map(self, mapview, **kwargs):
        for well in self.wells:
            well.plot_on_map(mapview, **kwargs)

    def annotate_map(self, mapview, **kwargs):
        for well in self.wells:
            well.annotate_map(mapview, **kwargs)
 
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
        self.bounds = extents_to_poly(*extents)

    def plot_horizon(self, hor, colormap=None):
        if isinstance(hor, basestring):
            hor = geoprobe.horizon(hor)
        if colormap is None:
            colormap = mpl.cm.jet
        hor.grid_extents = self.extents
        data = -hor.grid
        ls = Shader(azdeg=315, altdeg=45)
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

def extents_to_poly(xmin, xmax, ymin, ymax):
    x = [xmin, xmin, xmax, xmax]
    y = [ymin, ymax, ymax, ymin]
    return shapely.geometry.Polygon(zip(x, y))

class ScaledLocator(mpl.ticker.MaxNLocator):
    def __init__(self, dx=1.0, x0=0.0):
        self.dx = dx
        self.x0 = x0
        mpl.ticker.MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])

    def rescale(self, x):
        return x / self.dx + self.x0
    def inv_rescale(self, x):
        return  (x - self.x0) * self.dx

    def __call__(self): 
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = self.rescale(vmin), self.rescale(vmax)
        vmin, vmax = mpl.transforms.nonsingular(vmin, vmax, expander = 0.05)
        locs = self.bin_boundaries(vmin, vmax)
        locs = self.inv_rescale(locs)
        prune = self._prune
        if prune=='lower':
            locs = locs[1:]
        elif prune=='upper':
            locs = locs[:-1]
        elif prune=='both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

class ScaledFormatter(mpl.ticker.OldScalarFormatter):
    def __init__(self, dx=1.0, x0=0.0, **kwargs):
        self.dx, self.x0 = dx, x0

    def rescale(self, x):
        return x / self.dx + self.x0

    def __call__(self, x, pos=None):
        xmin, xmax = self.axis.get_view_interval()
        xmin, xmax = self.rescale(xmin), self.rescale(xmax)
        d = abs(xmax - xmin)
        x = self.rescale(x)
        s = self.pprint_val(x, d)
        return s

class Shader(mpl.colors.LightSource):
    def shade(self, data, shadedata=None, cmap=mpl.cm.jet, fraction=1.0, 
              mask=None, mode='overlay'):
        if shadedata is None:
            shadedata = data
        data = data.astype(float)
        shadedata = shadedata.astype(float)

        rgba = cmap((data-data.min())/(data.max()-data.min()))
        rgb = self.shade_rgb(rgba, elevation=shadedata)

        if mask is not None:
            for i in range(3):
                rgba[:,:,i] = np.where(mask, rgb[:,:,i], rgba[:,:,i])
        else:
            rgba[:,:,:3] = rgb[:,:,:3]
        return rgba

    def hillshade(self, data, fraction=1.0):
        az = np.radians(self.azdeg)
        alt = np.radians(self.altdeg)

        dx, dy = np.gradient(data)
        slope = 0.5 * np.pi - np.arctan(np.hypot(dx, dy))
        aspect = np.arctan2(dx, dy)
        intensity = (np.sin(alt) * np.sin(slope) 
                   + np.cos(alt) * np.cos(slope) 
                     * np.cos(-az - aspect - 0.5 * np.pi))

        intensity = (intensity - intensity.min()) / intensity.ptp()
        if fraction != 1.0:
            intensity = fraction * (intensity - 0.5) + 0.5
            if np.abs(fraction) > 1:
                np.clip(intensity, 0, 1, intensity)
        return intensity

    def shade_rgb(self, rgb, elevation, fraction=1.0, mode='overlay'):
        intensity = self.hillshade(elevation, fraction)
        intensity = intensity[:,:,np.newaxis]
        if mode == 'hsv':
            return mpl.colors.LightSource.shade_rgb(rgb, elevation)
        func = {'overlay':self.overlay, 'soft':self.soft_light}[mode]
        return func(rgb, intensity)

    def soft_light(self, rgb, intensity):
        return 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2 

    def overlay(self, rgb, intensity):
        low = 2 * intensity * rgb
        high = 1 - 2 * (1 - intensity) * (1 - rgb)
        return np.where(rgb <= 0.5, low, high)

class CoRenderedSection(Section):
    def __init__(self, vol, coherence_vol, x, y, ax, colormap, **kwargs):
        self.coherence_vol = coherence_vol
        Section.__init__(self, vol, x, y, ax, colormap, **kwargs)

    def _make_rgb(self, data, mode='overlay', fraction=0.85, mask=None):
        coh_data, coh_extent = self.extract_section(self.coherence_vol)

        if mask is None:
            mask = coh_data < (coh_data.mean() + 0.5 * coh_data.std())

        shader = Shader(azdeg=90)
        rgb = shader.shade(data, coh_data, self.colormap, mask=mask, 
                           fraction=fraction, mode=mode)
        return rgb

    def plot_image(self, mode='overlay', fraction=0.85, mask=None, **kwargs):
        data, extent = self.extract_section()
        rgb = self._make_rgb(data, mode, fraction, mask)
        self.im = self.ax.imshow(rgb, origin='lower', extent=extent,
                    interpolation='bilinear', aspect=self.ve, **kwargs)
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()
        return self.im

class SketchSection(Section):
    def plot_image(self, radius=4, **kwargs):
        data, extent = self.extract_section()
        rgb = self.colormap(data.astype(float) / 255)
        rgb = self.sketch_filter(rgb, radius)

        self.im = self.ax.imshow(rgb, origin='lower', extent=extent,
                    interpolation='bilinear', aspect=self.ve, **kwargs)
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()
        return self.im
        
    def sketch_filter(self, rgb, radius=4):
        hsv = mpl.colors.rgb_to_hsv(rgb[:,:,:3])
        hsv[:,:,1] = 0
        rgb = mpl.colors.hsv_to_rgb(hsv)
        original = rgb
        blur = scipy.ndimage.gaussian_filter(rgb, (radius, radius, 0))
        blur = 1 - blur
        rgb = 0.5 * rgb + 0.5 * blur
        rgb = original[:,:,:3] / (1 - rgb)
        hsv = mpl.colors.rgb_to_hsv(rgb.clip(0, 1))
        hsv[:,:,1] = 0
        return mpl.colors.hsv_to_rgb(hsv)


if __name__ == '__main__':
    basedir = '/data/MyCode/Geoprobe/distribution/examples/data/Horizons/'
    horizons = [basedir+item for item in ['seafloor.hzn', 'channels.hzn']]
    zmin, zmax = 1000, 4000
    sec = ExampleSections(horizons=horizons, horizon_styles=dict(lw=2))
    mapview = sec.add_map(horizon=basedir+'seafloor.hzn', annotate=True)
    mapview.plot_scalebar(1000, '1 km')
    sec.add_crossline(4950, zmin=zmin, zmax=zmax)
    sec.add_inline(sec.vol.xmin + 10, zmin=zmin, zmax=zmax)
    sec.add_section([sec.vol.xmin, sec.vol.xmax], 
                    [sec.vol.ymin, sec.vol.ymax], zmin=zmin, zmax=zmax)
    sec.show()

