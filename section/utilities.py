import numpy as np
import matplotlib as mpl
import shapely

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


