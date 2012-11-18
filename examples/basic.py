from seismic_plotting import ExampleSections

basedir = '/data/MyCode/Geoprobe/distribution/examples/data/Horizons/'
horizons = [basedir+item for item in ['seafloor.hzn', 'channels.hzn']]
zmin, zmax = 1000, 4000
sec = ExampleSections(horizons=horizons, horizon_styles=dict(lw=2))
mapview = sec.add_map(horizon=basedir+'seafloor.hzn', annotate=True)
mapview.plot_scalebar(length=1000, title='1 km')
sec.add_crossline(4950, zmin=zmin, zmax=zmax)
sec.add_inline(sec.vol.xmin + 10, zmin=zmin, zmax=zmax)
sec.add_section([sec.vol.xmin, sec.vol.xmax], 
                [sec.vol.ymin, sec.vol.ymax], zmin=zmin, zmax=zmax)
sec.show()
