# get sentinel1 file closest to labeled image date
# %%

mode = 'IW'

import ee
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

import matplotlib.pyplot as plt
import numpy as np
import rioxarray

# loop through all 90 images
# import labeled AOI, reproject, assign box coordinates

# open file

# reproject
# what is sentinel1 crs???
# %%
from os import listdir
filenames = listdir(data_folder+'\\other_usgs')

# %%
lat1,lon1 = aoi.lat.min(),aoi.lon.min()
lat2,lon2 = aoi.lat.max(),aoi.lon.min()
lat3,lon3 = aoi.lat.max(),aoi.lon.max()
lat4,lon4 = aoi.lat.min(),aoi.lon.max()

date2,date1 = aoi_date, aoi_date- 6days

geoJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              lon1,
              lat1
            ],
            [
              lon2,
              lat2
            ],
            [
              lon3,
              lat3
            ],
            [
              lon4,
              lat4
            ],
            [
              lon1,
              lat1
            ]
          ]
        ]
      }
    }
  ]
}
coords = geoJSON['features'][0]['geometry']['coordinates']
aoi = ee.Geometry.Polygon(coords)

im_coll = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
           .filterBounds(aoi)
           .filterDate(ee.Date(date1),ee.Date(date2))
           #.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
           #.filter(ee.Filter.eq('relativeOrbitNumber_start', 154))
           .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
           .sort('date'))

timestamplist = (im_coll.aggregate_array('date')
                 .map(lambda d: ee.String('T').cat(ee.String(d)))
                 .getInfo())
timestamplist

# select data closest to labeled image date
select ejfeijf 

# clip image to date
def clip_img(img):
    """Clips a list of images."""
    return ee.Image(img).clip(aoi)

im_list = im_coll.toList(im_coll.size())
im_list = ee.List(im_list.map(clip_img))

im_list.length().getInfo()

# save clip

# look at image
# look at histogram

# deal with speckle?
# save as numpy? --regrid to what?

# get S1_GRD_FLOAT
# make sure sample covers entire AOI, if it doesn't orbits, modes, swaths and scenes need to match
