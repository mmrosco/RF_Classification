# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:26:43 2020

@author: Melanie Martyn Rosco

This script is based on the Classification script from Chris Holden and Florian Beyer
"""

import ogr
import osr
import gdal
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import sys
from osgeo import ogr, gdal_array, gdalconst
from shapely.geometry import box

import rasterio
from rasterio.mask import mask

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble.RandomForestClassifier import get_params
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV



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


# This creates interger id attributes in shapefile based on class names, splots the training data into trainig and test subsets and then saves them to a raster file
def rasterize_and_split_train_data(ref_ras, shp_file, csv_class_name_id, train_file_with_ids, test_file_with_ids, train_out_ras, test_out_ras):
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

    train, test = train_test_split(gdf, test_size = 0.3, stratify = gdf['id'], random_state = 1)
    train.to_file(train_file_with_ids)
    test.to_file(test_file_with_ids)
    # Open reference raster
    in_ras = gdal.Open(ref_ras)
    
    train_ds = ogr.Open(train_file_with_ids)
    test_ds = ogr.Open(test_file_with_ids)
    train_lyr = test_ds.GetLayer()
    test_lyr = train_ds.GetLayer()
    
    # Create new training raster layer
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(train_out_ras, in_ras.RasterXSize, in_ras.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(in_ras.GetGeoTransform())
    target_ds.SetProjection(in_ras.GetProjection())
    
    # Create new test raster layer
    driver = gdal.GetDriverByName('GTiff')
    target_ds_2 = driver.Create(test_out_ras, in_ras.RasterXSize, in_ras.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds_2.SetGeoTransform(in_ras.GetGeoTransform())
    target_ds_2.SetProjection(in_ras.GetProjection())
    
    # Rasterise training points
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], train_lyr, options=options)
    gdal.RasterizeLayer(target_ds_2, [1], test_lyr, options=options)
    
    # Check generated training data raster and display basic stats
    data = target_ds.GetRasterBand(1).ReadAsArray()
    print('Train', 'min', data.min(), 'max', data.max(), 'mean', data.mean())
    
    # Check generated training data raster and display basic stats
    data = target_ds_2.GetRasterBand(1).ReadAsArray()
    print('Test','min', data.min(), 'max', data.max(), 'mean', data.mean())

    # Save raster file
    target_ds = None
    target_ds_2 = None


def classification_with_hyperparameter_optimisation(img, trai, val, out_ras_prediction, out_ras_clean, f):
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in image to be classified
    img_ds = gdal.Open(img)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), \
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))   
    # Read each band into an array
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
            
    # Read in training data raster
    trai_ds = gdal.Open(trai)        
    trai = trai_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    # Check NoData value = 0
    print(trai_ds.GetRasterBand(1).GetNoDataValue())
    
    # Read in validation data raster
    val_ds = gdal.Open(val)        
    val = val_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    # Check NoData value = 0
    print(val_ds.GetRasterBand(1).GetNoDataValue())
    
    # Display satellite image and training data
    plt.subplot(121)
    plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    plt.title('RS image - first band')

    plt.subplot(122)
    plt.imshow(trai, cmap=plt.cm.Spectral)
    plt.title('Training data')
    
    # Find how many non-zero entries we have -- i.e how many training data samples?
    n_samples = (trai > 0).sum()
    print('We have {n} training samples'.format(n=n_samples))

    # What are the classficiation labels?
    labels = np.unique(trai[trai>0])
    print('The training data includes {n} classes: {classes}'.format(n=labels.size, classes=labels))
     
    # We will need a 'X' matrix containing our features - data in training matrix rows and a 'y' array containing our labels - cols, these will have n sample rows
    y = trai[trai>0]
    x = img[trai>0, :]

    print('Our x matrix is sized: {sz}'.format(sz=x.shape))
    print('Our y matrix is sized: {sz}'.format(sz=y.shape))
        
    # Initialise Random Forest Classifier, n_jobs = -1 utilises all available cores
    rf = RandomForestClassifier(oob_score=True, n_jobs = -1)

    # Fit our model to training data
    rf = rf.fit(x,y)
    
    # Check the "Out-of-Bag" (OOB) prediction score
    print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
        
    # Show the band importance:
    bands = range(1,img_ds.RasterCount+1)

    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    # Take our full image and reshape it into a long 2d array (nrow * ncol, nband) for classification
    # Change to img.shape[2]-1 when using 4 bands (stacked NDWI + RGB)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)
        
    # Reshape back into original size
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    print('Reshaped back to {}'.format(class_prediction.shape))
    
    # Subset the classification image with the validation image = X
    # Mask the classes on the validation dataset = y
    # These will have n_samples rows
    X_v = class_prediction[val > 0]
    y_v = val[val > 0]

    # Check accuracy of model wtih default scikit-learn parameters
    print(confusion_matrix(y_v, X_v))
    print(classification_report(y_v, X_v))
    print(accuracy_score(y_v, X_v))
  
    # Set parameters and ranges to be tested
    n_estimators = np.arange(25, 100, 5)
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    
    # Set parameters and ranges into a dictionary
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    
    # Perform grid search to find the best parameters
    gridF = GridSearchCV(rf, hyperF, cv = 3, verbose = 1, n_jobs = -1)
    bestF = gridF.fit(x, y)
    print(bestF.best_params_)
    
    # Apply optimised model to entire image as array
    prediction = bestF.predict(img_as_array)
    
    # Reshape back into original size
    class_prediction = prediction.reshape(img[:, :, 0].shape)
    print('Reshaped back to {}'.format(class_prediction.shape))
    
    # Subset the classification image with the validation image = X
    # Mask the classes on the validation dataset = y
    # These will have n_samples rows
    X_v = class_prediction[val > 0]
    y_v = val[val > 0]

    # Check accuracy of model wtih default scikit-learn parameters
    print(confusion_matrix(y_v, X_v))
    print(classification_report(y_v, X_v))
    print(accuracy_score(y_v, X_v))
    
    # Get info to save output raster from input raster
    geo = img_ds.GetGeoTransform()
    proj = img_ds.GetProjectionRef()

    ncol = img.shape[1]
    nrow = img.shape[0]

    drv = gdal.GetDriverByName('GTiff')
    
    class_prediction.astype(np.float16)
    
    out_ras_prediction = drv.Create(out_ras_prediction, ncol, nrow, 1, gdal.GDT_UInt16)
    out_ras_prediction.SetProjection(proj)
    out_ras_prediction.SetGeoTransform(geo)       
    out_ras_prediction.GetRasterBand(1).WriteArray(class_prediction)
    out_ras_prediction = None

    return 


# Classification without GridSearch to find the best hyperparameters (faster)
def classification(img, trai, val, out_ras_prediction, out_ras_clean, f):
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in image to be classified
    img_ds = gdal.Open(img)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), \
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))   
    # Read each band into an array
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
            
    # Read in training data raster
    trai_ds = gdal.Open(trai)        
    trai = trai_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    # Check NoData value = 0
    print(trai_ds.GetRasterBand(1).GetNoDataValue())
    
    # Read in validation data raster
    val_ds = gdal.Open(val)        
    val = val_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    # Check NoData value = 0
    print(val_ds.GetRasterBand(1).GetNoDataValue())
    
    # Display satellite image and training data
    plt.subplot(121)
    plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    plt.title('RS image - first band')

    plt.subplot(122)
    plt.imshow(trai, cmap=plt.cm.Spectral)
    plt.title('Training data')
    
    # Find how many non-zero entries we have -- i.e how many training data samples?
    n_samples = (trai > 0).sum()
    print('We have {n} training samples'.format(n=n_samples))

    # What are the classficiation labels?
    labels = np.unique(trai[trai>0])
    print('The training data includes {n} classes: {classes}'.format(n=labels.size, classes=labels))
     
    # We will need a 'X' matrix containing our features - data in training matrix rows and a 'y' array containing our labels - cols, these will have n sample rows
    y = trai[trai>0]
    x = img[trai>0, :]

    print('Our x matrix is sized: {sz}'.format(sz=x.shape))
    print('Our y matrix is sized: {sz}'.format(sz=y.shape))
        
    # Initialise Random Forest Classifier, n_jobs = -1 utilises all available cores, add parameters as wanted
    rf = RandomForestClassifier(oob_score=True, n_jobs = -1)

    # Fit our model to training data
    rf = rf.fit(x,y)
    
    # Check the "Out-of-Bag" (OOB) prediction score
    print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
        
    # Show the band importance:
    bands = range(1,img_ds.RasterCount+1)

    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    # Take our full image and reshape it into a long 2d array (nrow * ncol, nband) for classification
    # Change to img.shape[2]-1 when using 4 bands (stacked NDWI + RGB)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)
        
    # Reshape back into original size
    class_prediction_reshape = class_prediction.reshape(img[:, :, 0].shape)
    print('Reshaped back to {}'.format(class_prediction_reshape.shape))
    
    # Subset the classification image with the validation image = X
    # Mask the classes on the validation dataset = y
    # These will have n_samples rows
    X_v = class_prediction_reshape[val > 0]
    y_v = val[val > 0]

    # Check accuracy of model wtih default scikit-learn parameters
    print(confusion_matrix(y_v, X_v))
    print(classification_report(y_v, X_v))
    print(accuracy_score(y_v, X_v))
    
    # Get info to save output raster from input raster
    geo = img_ds.GetGeoTransform()
    proj = img_ds.GetProjectionRef()

    ncol = img.shape[1]
    nrow = img.shape[0]

    drv = gdal.GetDriverByName('GTiff')
    
    class_prediction_reshape.astype(np.float16)
    
    out_ras_pred = drv.Create(out_ras_prediction, ncol, nrow, 1, gdal.GDT_UInt16)
    out_ras_pred.SetProjection(proj)
    out_ras_pred.SetGeoTransform(geo)       
    out_ras_pred.GetRasterBand(1).WriteArray(class_prediction_reshape)
    out_ras_pred = None
    
    return 
  
# Function for salt and pepper cleanup, first cleanup was done in QGIS with gdal.sieve
# Adjust limits to boundaries of input 
def cleanup(in_ras, out_ras):
    raster = gdal.Open(in_ras)  
    raster_array = raster.GetRasterBand(1).ReadAsArray() 
    
    geo =  raster.GetGeoTransform()
    proj =  raster.GetProjectionRef()    
  
    rows = raster_array.shape[0]
    cols = raster_array.shape[1]
    
    # Adjust depending on boundaries of input raster and sub_array size used in loop
    limits = [1636, 1637, 1638, 1639, 2715, 2716, 2717]
    
    for index, value in np.ndenumerate(raster_array):
        # Skips to next loop if index contains limit number
            if any(t in index for t in limits):
                continue
            # Adjust value depending on class value used and being considered - here 2 is a fluvial pixel
            if value == 2:
                row_indices = [index[0]-2, index[0]-1, index[0], index[0]+1, index[0]+2] # 5x5, adjust to change sub-array size
                col_indices = [index[1]-2, index[1]-1, index[1], index[1]+1, index[1]+2] # 5x5
                sub_arr = raster_array[np.ix_(row_indices,col_indices)] # Create sub-array around element being considered
                idxs = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
                idxs = [i*sub_arr.shape[1]+j for i, j in idxs]
                sub_arr = np.delete(sub_arr, idxs)  # Delete inner array elements to only consider outer row and columns (position in flattened array = row * no_of_columns + column)
                if np.sum(sub_arr == 2) < 4:
                    print('Replacing stream with pond')
                    print(index[0], index[1])
                    raster_array[index[0],index[1]] = 1
                elif np.any(sub_arr == 0):
                    print('Replacing stream with lake')
                    print(index[0], index[1])
                    raster_array[index[0],index[1]] = 0     
            # Adjust value depending on class value used and being considered - here 1 is a pond pixel
            elif value == 1:
                row_indices = [index[0]-2, index[0]-1, index[0], index[0]+1, index[0]+2] # 5x5, adjust to change sub-array size
                col_indices = [index[1]-2, index[1]-1, index[1], index[1]+1, index[1]+2] # 5x5  
                sub_arr = raster_array[np.ix_(row_indices,col_indices)] # Create sub-array around element being considered
                idxs = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
                idxs = [i*sub_arr.shape[1]+j for i, j in idxs]
                sub_arr = np.delete(sub_arr, idxs)  # Delete inner array elements to only consider outer row and columns (position in flattened array = row * no_of_columns + column)
                if np.sum(sub_arr == 2) > 10:
                    print('Replacing pond with stream')
                    print(index[0], index[1])
                    raster_array[index[0],index[1]] = 2

    drv = gdal.GetDriverByName('GTiff')           
    out_ras_cleaned = drv.Create(out_ras, cols, rows, 1, gdal.GDT_Byte)
    out_ras_cleaned.SetProjection(proj)
    out_ras_cleaned.SetGeoTransform(geo)       
    out_ras_cleaned.GetRasterBand(1).WriteArray(raster_array)
    out_ras_cleaned = None      

    return out_ras_cleaned       
 
# Returns confusion matrix, classification report and accuracy score for classified raster pre and post cleanup    
def quality_check(class_raster, cleaned_raster, test_raster):
    # Read in classified raster
    img_ds = gdal.Open(class_raster)
    img = img_ds.GetRasterBand(1).ReadAsArray() 
        
    # Read in cleaned classification raster
    clean_ds = gdal.Open(cleaned_raster)
    clean = clean_ds.GetRasterBand(1).ReadAsArray()
           
    # Read in training data raster
    test_ds = gdal.Open(test_raster)        
    test = test_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    X_v = img[test > 0]
    y_v = test[test > 0]

    # Check accuracy of model wtih default scikit-learn parameters
    print(confusion_matrix(y_v, X_v))
    print(classification_report(y_v, X_v))
    print(accuracy_score(y_v, X_v))
    
    X_v = clean[test > 0]
    y_v = test[test > 0]

    # Check accuracy of model wtih default scikit-learn parameters
    print(confusion_matrix(y_v, X_v))
    print(classification_report(y_v, X_v))
    print(accuracy_score(y_v, X_v))
    
    return

# Calculate area covered by each water body based on class pixels
def calc_wb_areas(in_raster, wbs_dict):
    # Open reprojected raster
    raster = gdal.Open(in_raster)  
    raster_array = raster.GetRasterBand(1).ReadAsArray()  

    # Create dictionary with class name : amount of pixels
    dicts = {}
    for key, value in wbs_dict.items():
        dicts[key] = len(raster_array[raster_array==value])
           
    # Get pixel size and calculate area
    gt = raster.GetGeoTransform()
    pixel_area = gt[1] * (-gt[5])
    
    total_area = sum(dicts.values()) * pixel_area

    # Multiply each total in list with pixel_area to get area covered by wb in km2
    # Create dictionary of classes and corresponding area in km2     
    wb_areas = {}
    for key, value in dicts.items():
       wb_areas[key] = round(value*pixel_area*10**-6,2)
       
    # Create dictionary of classes and corresponding area in % of total study area   
    wb_areas_perc = {}
    for key, value in dicts.items():
       wb_areas_perc[key] = round((value*pixel_area)/total_area, 2) * 100    
    
    return wb_areas, wb_areas_perc




