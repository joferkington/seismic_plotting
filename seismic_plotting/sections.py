import numpy as np
import geoprobe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib as mpl
import scipy.ndimage
import shapely.geometry

import utilities

class Section(object):
    """
    An "arbitrary" cross section plot extracted from a geoprobe volume.
    """
    def __init__(self, vol, x, y, ax, colormap, zmin=None, zmax=None, ve=2.0, 
                 name='Cross Section', resample_factor=2):
        """
        Make a new cross section along `x` and `y` from seismic data in `vol`
        on the matplotlib axes `ax`.

        Parameters:
        -----------
            vol : A geoprobe volume object containing the seismic data to 
                be displayed on the cross section.
            x : A sequence of x-coordinates (in inline/crossline) representing
                points along the cross section line
            y : A sequence of y-coordinates (in inline/crossline) representing
                points along the cross section line
            ax : A matplotlib axes
            colormap : A matplotlib colormap
            zmin : The minimum (top) depth/time for the cross section
            zmax : The maximum (bottom) depth/time for the cross section
            ve : The vertical exaggeration of the displayed cross section. If
                the seismic data is in depth, then this is the true vertical 
                exaggeration. 
            name : The title of the cross section
            resample_factor : Interpolation factor for the "raw" seismic data.
                If > 1, the seismic data will be linearly interpolated before
                display.
        """
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
        """
        Extract data along this cross section's profile from a geoprobe volume. 
        In most cases, this method will only be called internally, and you won't
        need to call it explictly. However, it is often useful for constructing
        "unusual" plots.

        Parameters
        ----------
            vol : (optional) A geoprobe volume instance
                Defaults to the geoprobe volume specified at initalization.
        Returns
        -------
            data : A 2D numpy array of seismic data
            extent : A 4-element list of the minimum and maximum distances along
                this cross section's profile line and the minimum and maximum
                z-values of the seismic data.
        """
        if vol is None:
            vol = self.vol
        data, xi, yi = vol.extract_section(self.x, self.y, self.zmin, self.zmax)
        data = self.make_raw_image(data)
        distance = self.calculate_distance_along_section(xi, yi)
        extent = [distance.min(), distance.max(), self.zmin, self.zmax]
        return data, extent

    def update_position(self, x, y):
        # TODO: Finish this...
        self.x, self.y = x, y
        data, extent = self.extract_section()
        self.im.set_data(data)
        self.im.set_extent(extent)

    def update_horizons(self, hor_set):
        # TODO: Finish this...
        for line, hor in zip(self.horizon_lines, hor_set.horizons):
            x, y = self.slice_horizon(hor)
            line.set_data(x, y)

    @property
    def line(self):
        """
        A shapely LineString representing this cross section's profile line.
        """
        return shapely.geometry.LineString(zip(self.x, self.y))

    def calculate_distance_along_section(self, xi, yi):
        """
        Calculates the distance along this cross section's profile line to the
        specified point. 

        Parameters:
        -----------
            xi, yi : Sequences of x and y coordinates
        Returns:
        --------
            distance : The distance along the profile line to the point
        """
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
        """
        Plot labeled endpoint coordinates on the section. 
        Additional keyword arguments are passed on to `annotate`, which allows
        the position, style, etc of the labels to be controlled. By default,
        the labels will be something like "<-- XL: x0, IL: y0" and 
        "XL: x1, IL: y1 -->", with arrows pointing to each lower corner.

        Parameters:
        -----------
            template : The formatting template for the endpoint coords.
            Additional keyword arguments are passed on to `annotate`

        Returns:
        --------
            A 2-item list of matplotlib Annotation objects
        """
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
            hor.bounds = utilities.extents_to_poly(*hor.grid_extents)
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

class CoRenderedSection(Section):
    def __init__(self, vol, coherence_vol, x, y, ax, colormap, **kwargs):
        self.coherence_vol = coherence_vol
        Section.__init__(self, vol, x, y, ax, colormap, **kwargs)

    def _make_rgb(self, data, mode='overlay', fraction=0.85, mask=None):
        coh_data, coh_extent = self.extract_section(self.coherence_vol)

        if mask is None:
            mask = coh_data < (coh_data.mean() + 0.5 * coh_data.std())

        shader = utilities.Shader(azdeg=90)
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


