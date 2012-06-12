import csv

import shapely
import geoprobe

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
 
