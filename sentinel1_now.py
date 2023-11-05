# get sentinel1 file closest to labeled image date
# how to deal with compose the full AOI image cover by compositing two adjacent scenes in this case
# deleted 48 because images need to be composed
# https://developers.google.com/earth-engine/tutorials/community/sar-basics#sentinel-1_coverage

# %%
import ee
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
from os import listdir
import os
from datetime import datetime

# Trigger the authentication flow.# Initialize the library.
ee.Authenticate()
ee.Initialize()
# %%
filenames = listdir(os.getcwd()+'\\labels')
dates = [filenames[i][6:-4] for i in range(90)]


# %%
for i in range(len(filenames)):
    print(i)
    label = rxr.open_rasterio('labels\\'+filenames[i])
    label = label.rio.reproject("EPSG:4326")

    date = dates[i]
    date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")

    lat1,lon1 = float(label.y.min().values),float(label.x.min().values)
    lat2,lon2 = float(label.y.max().values),float(label.x.min().values)
    lat3,lon3 = float(label.y.max().values),float(label.x.max().values)
    lat4,lon4 = float(label.y.min().values),float(label.x.max().values)

    geojson_object = {
        'type': 'Polygon',
        'coordinates': [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]}

    aoi = ee.Geometry(geojson_object)

    endDate = ee.Date(date)
    endDate = endDate.advance(1, 'days')
    startDate = endDate.advance(-10, 'days')

    POLARIZATION = ['HH','HV','VV','VH'] 
    for j in POLARIZATION:
        im_coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(aoi)
                .filterDate(startDate,endDate)
                .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                .sort('date')
                .sort('system:time_start', False)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', j))
                .select(j))


        timestamplist = (im_coll.aggregate_array('date')
                        .map(lambda d: ee.String('T').cat(ee.String(d)))
                        .getInfo())
        print(date)
        print(timestamplist)

        try:
            # convert collection to list, select data closest to labeled image date and clip image to date
            def clip_img(img):
                """Clips a list of images."""
                return ee.Image(img).clip(aoi)

            im_list = im_coll.toList(im_coll.size())
            im_list = ee.List(im_list.map(clip_img))

            recent = ee.Image(im_list.get(0))

            # save clip, same resolution as labeled image

            projection = recent.projection().getInfo()


            task = ee.batch.Export.image.toDrive(image=recent,
                                                description=filenames[i][0:5],
                                                region = aoi,
                                                fileNamePrefix=filenames[i][0:5]+j+timestamplist[0],
                                                crs=projection['crs'],
                                                crsTransform= projection['transform'],
                                                fileFormat='GeoTIFF')
            task.start()
        except Exception:
            print('missing '+filenames[i][0:5])
            pass


# %%
