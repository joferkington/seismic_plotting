import itertools

import geoprobe

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

