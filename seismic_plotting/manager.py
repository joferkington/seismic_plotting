import os
import glob
import matplotlib.pyplot as plt

import geoprobe

import utilities
import sections
from horizons import HorizonSet
from maps import Map
from wells import WellSet, WellDatabase

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
            sec_type = sections.Section
        else:
            sec_type = sections.SketchSection

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
        locator = utilities.ScaledLocator(self.vol.dyW, ymin)
        formatter = utilities.ScaledFormatter(self.vol.dyW, ymin)
        sec.ax.xaxis.set_major_locator(locator)
        sec.ax.xaxis.set_major_formatter(formatter)
        sec.ax.set_xlabel('Crossline V.E.:%0.1f' % sec.ve)
        return sec
 
    def add_crossline(self, yval, **kwargs):
        xmin = kwargs.pop('xmin', self.vol.xmin)
        xmax = kwargs.pop('xmax', self.vol.xmax)
        kwargs['name'] = kwargs.get('name', 'Crossline {}'.format(int(yval)))

        sec = self.add_section([xmin, xmax], [yval, yval], **kwargs)
        locator = utilities.ScaledLocator(self.vol.dxW, xmin)
        formatter = utilities.ScaledFormatter(self.vol.dxW, xmin)
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


