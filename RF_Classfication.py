# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:26:43 2020

@author: mmo990
"""

import ogr
import osr
import gdal
import pandas as pd
import numpy as np
import geopandas as gpd
from collections import Counter
import math
import sys
from osgeo import ogr, gdal_array, gdalconst
from scipy.ndimage import label, binary_dilation
from shapely.geometry import box

import rasterio
from rasterio import warp
from rasterio.crs import CRS
from rasterio.mask import mask

import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from RandomForestClassifier import get_params
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pprint

import skimage
from skimage.filters.rank import modal
from skimage.morphology import disk



# --------------------------------------------------------
# function to parse feature from GeoDataFrame in such a manner that rasterio wants them
def getFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


# Conver coordinates from one EPSG to another EPSG
def coord_conversion(inputEPSG, outputEPSG, coords):
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    x_list = []
    y_list = []
    
    for index, row in coords.iterrows():
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(row['x'],row['y'])
        point.Transform(coordTransform)
        x_list.append(point.GetX())
        y_list.append(point.GetY())
        newcoords = pd.DataFrame({'x':x_list, 'y':y_list})
    
    return newcoords


def clipping(in_raster, out_raster, mixx_, miny_, maxx_, maxy_):
    driver = gdal.GetDriverByName('GTiff')
    in_ras = rasterio.open(in_raster)
    bbox = box(mixx_, miny_, maxx_, maxy_)

   # insert bbox into GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index = [0], crs = in_ras.crs)

   # coordinates in format accepted by rasterio
    coords = getFeatures(geo)      
    print(coords)  
        
    # clip using mask
    out_img, out_transform = mask(dataset = in_ras, shapes = coords, crop = True)   

    # copy metadata
    out_meta = in_ras.meta.copy()

    # update  metadata with new dimensions, transform (affine), CRS
    out_meta.update({'driver': 'GTiff',
                 'height': out_img.shape[1],
                 'width': out_img.shape[2],
                 'transform': out_transform,
                 'crs': in_ras.crs})

    with rasterio.open(out_raster, 'w', **out_meta) as dest:
        dest.write(out_img)
        
    return out_raster

# Calculate Earth-Sun distance
def Earth_Sun_distance(year, month, day, hh, mm, ss):
    # Universal time
    UT = hh + (mm/60) + (ss/3600)
    
    # Julian Day (JD)
    A = int(year/100)
    B = 2 - A + int(A/4)
    JD = int(365.25*(year + 4716) + int(30.6001 * (month + 1)) + day + (UT/24) + B - 1524.5)
    
    # Earth-Sun distance (dES)
    D = JD - 2451545
    g  = 357.529 + 0.98560028 * D # (in degrees)
    dES = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * np.cos(np.deg2rad(2*g))
    
    return dES

# Generate top pf atmosphere reflectance and surface reflectance raster from DN numbers - needs dES from previous function
def refl(in_raster, absFactors, effBandWidths, ESUN, dES, theta, out_raster_TOA, out_raster_surf):
    drv = gdal.GetDriverByName('GTiff')
    raster = gdal.Open(in_raster)
    
    referenceProj = raster.GetProjection()
    referenceTrans = raster.GetGeoTransform()
    x = raster.RasterXSize
    y = raster.RasterYSize
    n = raster.RasterCount
    
    raster_array = raster.ReadAsArray()
    data = np.single(raster_array)
    data = data.T
    data[data == 0] = np.nan
    
    # Plot 4 bans of multi raster
    fig = plt.subplots(figsize=(13,7))

    plt.subplot(221)
    plt.imshow(data[:,:,0]/255)
    title('Blue')
    plt.colorbar

    plt.subplot(222)
    plt.imshow(data[:,:,1]/255)
    title('Green')
    plt.colorbar

    plt.subplot(223)
    plt.imshow(data[:,:,2]/255)
    title('Red')
    plt.colorbar

    plt.subplot(224)
    plt.imshow(data[:,:,3]/255)
    title('NIR')
    plt.colorbar

    plt.show()

    # Emtpy matrices that will store the TOA and surface reflectance data
    refl_TOA = np.zeros(data.shape)
    refl_surf = np.zeros(data.shape)

    for i in range(4):
        im = data[:,:,i]
        # Calculate DN to radiance
        L = (im * absFactors[i])/effBandWidths[i]
        L = np.squeeze(L)
        # Calculates the theoretical radiance of a dark object as 1% of the max possible radiance
        L1percent = (0.01 * ESUN[i] * np.cos(np.deg2rad(theta))) / (dES**2 * math.pi)
        # Find darkest pixel in image
        Lmini = np.nanmin(L)
        # The difference between the theoretical 1% radiance of a dark object and the radiance of the darkest image pixel is due to the atm (empirical)
        Lhaze = Lmini - L1percent
        # TOA reflectance
        refl_TOA[:, :, i] = (L * math.pi * dES**2) / (ESUN[i] * np.cos(np.deg2rad(theta)))
        # Surface reflectance
        refl_surf[:, :, i] = (math.pi * (L-Lhaze) * dES**2) / (ESUN[i] * np.cos(np.deg2rad(theta)))
    
    # Save to rasters
    refl_TOA = np.float32(refl_TOA)
    output_TOA = drv.Create(out_raster_TOA, x, y, n, gdal.GDT_Float32)
    if output_TOA is None:
        print('The output raster could not be created')
        sys.exit(-1)
    output_TOA.SetGeoTransform(referenceTrans)
    output_TOA.SetProjection(referenceProj)
    
    refl_TOA = refl_TOA.T        
    for i, image in enumerate(refl_TOA, 1):
        output_TOA.GetRasterBand(i).WriteArray(image)
        output_TOA.GetRasterBand(i).SetNoDataValue(0)
    
    output_TOA = None    
    
    refl_surf = np.float32(refl_surf)
    output_surf = drv.Create(out_raster_surf, x, y, n, gdal.GDT_Float32)
    if output_surf is None:
        print('The output raster could not be created')
        sys.exit(-1)
    output_surf.SetGeoTransform(referenceTrans)
    output_surf.SetProjection(referenceProj)
    
    refl_surf = refl_surf.T
    for i, image in enumerate(refl_surf, 1):
        output_surf.GetRasterBand(i).WriteArray(image)
        output_surf.GetRasterBand(i).SetNoDataValue(0)
    
    output_surf = None  
       
    return output_TOA, output_surf




# This creates interget id attributes in shapefile based on class names and then saves training data as a raster file
def rasterize_train_data(ref_ras, shp_file, csv_class_name_id, shp_file_with_ids, out_ras):
    # Create id in shp corresponding to class name in shapefile
    gdf = gpd.read_file(shp_file)
    class_names = gdf['Class name'].unique()
    print('class names', class_names)
    class_ids = np.arange(class_names.size) + 1
    print('class ids', class_ids)
    df = pd.DataFrame({'label':class_names, 'id':class_ids})
    df.to_csv(csv_class_name_id)
    print('gdf without ids', gdf.head())
    gdf['id'] = gdf['Class name'].map(dict(zip(class_names, class_ids)))
    print('gdf with ids', gdf.head())

    gdf.to_file(shp_file_with_ids)
    
    # Open reference raster
    in_ras = gdal.Open(ref_ras)
    
    train_ds = ogr.Open(shp_file_with_ids)
    lyr = train_ds.GetLayer()
    
    # Create new raster layer (training data raster)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(out_ras, in_ras.RasterXSize, in_ras.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(in_ras.GetGeoTransform())
    target_ds.SetProjection(in_ras.GetProjection())
    
    # Rasterise training points
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    
    
    # Check generated training data raster and display basic stats
    data = target_ds.GetRasterBand(1).ReadAsArray()
    print('min', data.min(), 'max', data.max(), 'mean', data.mean())

    # Save raster file
    target_ds = None


    
    


# i is a float (e.g. 0.3) used to divide test and training data
def classification(img, trai, i, out):
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in raster and training raster
    img_ds = gdal.Open(img)
    trai_ds = gdal.Open(trai)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), \
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    
    # Read each band into an array
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
    trai = trai_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    # Check NoData value = 0
    print(trai_ds.GetRasterBand(1).GetNoDataValue())

    # Find how many non-zero entries we have -- i.e how many trainign data samples?
    n_samples = (trai > 0).sum()
    print('We have {n} samples'.format(n=n_samples))

    # What are the classficiation labels?
    labels = np.unique(trai[trai>0])
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

    # We will need a 'X' matrix containing out features - data in training matrix rows and a 'y' array containing our labels - cols, these will have n sample rows
    X = img[trai>0, :]
    y = trai[trai>0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i)

    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y matrix is sized: {sz}'.format(sz=y.shape))
    

    # Initialise our model with 250 trees - optimize
    rf = RandomForestClassifier(n_estimators = 75, oob_score=True)

    # Fit our model to training data
    rf = rf.fit(X_train,y_train)
  
    for b, imp in zip(range(img_ds.RasterCount), rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
        
    # Look at parameters used by current forest
    #print('Parameters currently in use:\n')
    #pprint(rf.get_params())
    
    # Take our full image and reshape it into a long 2d array (nrow * ncol, nband) for classification
    # Change to img.shape[2]-1 when using 4 bands (stacked NDWI + RGB)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)

    # F1 score
    #score = metrics.f1_score(y_test, class_prediction, pos_label=list(set(y_test)))
    # Overall accuracy
    #pscore = metrics.accuracy_score(y_test, )
    
    print('Our prediction matrix is sized: {sz}'.format(sz=class_prediction.shape))
    print('Our y test matrix is sized: {sz}'.format(sz=y_test.shape))
    y2 = y_test.reshape((len(y_test), 1))
    pred2 = class_prediction.reshape((len(class_prediction), 1))
    print('Our updated prediction matrix is sized: {sz}'.format(sz=pred2.shape))
    print('Our updated y test matrix is sized: {sz}'.format(sz=y2.shape))
    
    # Check accuracy - these didn't work for me
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y2, pred2))
    #print('Mean Square Error:', metrics.mean_squared_error(y_test, class_prediction))
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, class_prediction)))
    
    # Metrics to evaluate classification - these didn't work for me
    #print(metrics.confusion_matrix(y_test, class_prediction))
    #print(metrics.classification_report(y_test, class_prediction))
    #print(metrics.accuracy_score(y_test, class_prediction))
    #print(metrics.cohen_kappa_score(y_test, class_prediction))

    # Reshape classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)

    print('Our OOB prediction of accuracy is: {oob}'.format(oob=rf.oob_score_*100))

    trai_ds = None
     
    geo = img_ds.GetGeoTransform()
    proj = img_ds.GetProjectionRef()

    ncol = img_ds.RasterXSize
    nrow = img_ds.RasterYSize

    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(out, ncol, nrow, 1, gdal.GDT_Byte)

    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)

    ds.GetRasterBand(1).WriteArray(class_prediction)

    ds = None
    
    return rf, X_train, y_train, X_test, y_test

## Clean up salt and pepper from classification
def cleanup(in_raster, out_raster, disk_size):
    drv = gdal.GetDriverByName('GTiff')
    raster = gdal.Open(in_raster)
    ncol = raster.RasterXSize
    nrow = raster.RasterYSize
    # Fetch projection and extent
    proj = raster.GetProjectionRef()
    ext = raster.GetGeoTransform()
    
    raster_array = raster.ReadAsArray()
    
    # Create mask so that ponds are not included in filter
    mask_ = np.zeros(raster_array.shape, dtype=np.uint8)
    boolean = raster_array != 2
    mask_[boolean] = 1

    # Apply filter while ignoring ponds
    cleaned = skimage.filters.rank.modal(raster_array, skimage.morphology.disk(disk_size), mask = mask_)
 
    # If pond has been changed to tundra, change it back           
    cleaned_with_ponds = np.where(((raster_array == 2) & (cleaned == 4)), 2, cleaned)  
 
    # Fill nodata (0) with the most common surrounding value
    mask2 = cleaned_with_ponds == 0
    labels, count = label(mask2)
    arr_out = cleaned_with_ponds.copy()
    for idx in range(1, count + 1):
        hole = labels == idx
        surrounding_values = cleaned_with_ponds[binary_dilation(hole) & ~hole]
        most_frequent = Counter(surrounding_values).most_common(1)[0][0]
        arr_out[hole] = most_frequent
    

    out_ras = drv.Create(out_raster, ncol, nrow, 1, gdal.GDT_Byte)
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(ext)       
    out_ras.GetRasterBand(1).WriteArray(arr_out)
    out_ras = None
    


## Calculate area covered by each water body based on class pixels
def calc_wb_areas(in_raster, wbs_dict):
    # Open reprojected raster
    raster = gdal.Open(in_raster)
    raster_array = raster.GetRasterBand(1).ReadAsArray()

    # Sum pixel amount of each class
    count = Counter(raster_array.flatten())

    # Extract total of each class into list
    wbs = []
    wb_pixel_sum = []
    for class_, i in count.items():
        wb_pixel_sum.append(i)
        wbs.append(list(wbs_dict)[class_-1])
            

    # Get pixel size and calculate area
    gt = raster.GetGeoTransform()
    pixel_area = gt[1] * (-gt[5])
    
    total_area = sum(wb_pixel_sum) * pixel_area

    # Multiply each total in list with pixel_area to get area covered by wb in km2
    # Create list of names matching numbers
    wb_areas = {}
    for i,j in zip(wb_pixel_sum, wbs):
        wb_areas[j] = round(i*pixel_area*10**-6,2)
    
    return wb_areas, wb_pixel_sum, total_area
    

# The next two functions are extra, not sure they work yet
def fitting_random_search(img, trai, i):
    
    # Create ranfom hyperparameter grid
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    pprint(random_grid)
       
    # Use the random grid to search for best hyperparameters
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_inter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    
    # View parameters from fitting random search
    return rf_random.best_params_, X_train, y_train, X_test, y_test

def evaluate(model, test_features, test_labels):
    pred = model.predict(test_features)
    errors = abs(pred - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {0.2f}%.'.format(accuracy))
    return accuracy
    