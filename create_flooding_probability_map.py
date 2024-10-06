# Import the necessary libraries
import io
import ee
# import folium
# import geemap
import random
import time

# Authenticate with Earth Engine
ee.Authenticate()
ee.Initialize()

#################################################################################################################
# This code uses the Earth Engine Python API to load data from a table called "neotreep_v4"
# for the species Luehea divaricata, defining a spatial resolution of 1000 meters.
# Then, the function "RemoveDuplicates" is defined to remove duplicate records from the table.
# For this, a random image is created with the EPSG:4326 projection and null spatial resolution.
# Next, a point sampling is performed with a scale of 10 meters for the table data, and the
# "distinct" function is applied to the random point values, returning a data collection without duplicates.
# It's important to note that the effectiveness of the "RemoveDuplicates" function depends on how well
# the data is spatially distributed. If there are clusters of nearby points, the function may not remove
# all duplicate records.
#################################################################################################################

# Load data from the neotreep table for the species Luehea divaricata
# https://code.earthengine.google.com/?asset=projects/ee-kikosmoura/assets/df_neotree_amazonia
# Data = ee.FeatureCollection('projects/ee-kikosmoura/assets/df_neotree_Euterpe_oleracea')
Data = ee.FeatureCollection('projects/ee-kikosmoura/assets/df_enchentes')
# Data = ee.FeatureCollection('users/kikosmoura_ml_01/neotreep_v4')

# Define the spatial resolution in meters
# GrainSize = 1000
GrainSize = 90

def RemoveDuplicates(data):
    randomraster = ee.Image.random().reproject('EPSG:4326', None, GrainSize)
    randpointvals = randomraster.sampleRegions(collection=ee.FeatureCollection(data), scale=10, geometries=True)
    return randpointvals.distinct('random')

Data = RemoveDuplicates(Data)

###############################################################################################################
# This code uses the Earth Engine Python API to load a collection of global administrative boundary geometries,
# called 'USDOS/LSIB_SIMPLE/2017', and filters only the geometries corresponding to Brazil, based on the country's
# ISO code ('BR').
# Then, the "geometry()" function is applied to the filtered collection to extract the country's geometry and
# store it in the variable "AOI" (Area of Interest). The resulting geometry is a polygonal representation of
# Brazil's land area, which can be used to perform image processing operations and data
# spatially limited to the country.
###############################################################################################################

# Defining Brazil as the region of interest
# AOI = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'BR')).geometry()
rs_uf = ee.FeatureCollection('projects/ee-kikosmoura/assets/RS_UF')
firstFeature = rs_uf.first()

# Extract the geometry of the first feature
AOI = firstFeature.geometry()

####################################################################################################################
# This code uses the Earth Engine Python API to create a dataset of predictor variables
# for modeling and spatial analysis.
# First, three images are loaded from Earth Engine:
# 'WORLDCLIM/V1/BIO': represents data of 19 bioclimatic variables, such as temperature, precipitation, and humidity,
# at a spatial resolution of 1 km.
# 'USGS/SRTMGL1_003': represents elevation data at a spatial resolution of 30 meters.
# 'MODIS/006/MOD44B': represents global vegetation cover data, derived from MODIS sensor images,
# at a spatial resolution of 500 meters.
# Then, preprocessing operations are applied to the data to create a set of predictor variables.
# The first step is to calculate the median of the vegetation cover for the period from 2003 to 2020, in order to obtain
# an estimate of the average vegetation cover for the region of interest.
# Next, the three images are combined using the "addBands" function to form a single image with the
# predictor variables. The resulting image is then clipped to the area of interest defined by the variable
# 'AOI', and a water mask is created using the elevation (elevation values greater than 0 are considered land).
# Finally, the predictor image is updated with the water mask, and only the bands of interest
# are selected ("bio04", "bio05", "bio06", "bio12", "elevation", and "Percent_Tree_Cover").
# The final result is a dataset of predictor variables that can be used for modeling or
# spatial analysis.
####################################################################################################################

LCT = ee.ImageCollection('MODIS/061/MCD12Q1').select(['LC_Type1']).median()

SO2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2')\
    .filterDate('2019-06-01', '2019-06-11')\
    .select(['SO2_column_number_density']).median()
       
PH = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')

# Load a multiband image from the data catalog
BIO = ee.Image("WORLDCLIM/V1/BIO")

# Load elevation data from the data catalog and calculate the slope, aspect, and a simple shadow of the digital elevation model
Terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))

# Load the 250 m NDVI collection and estimate the annual average tree cover per pixel
MODIS = ee.ImageCollection("MODIS/006/MOD44B")
MedianPTC = MODIS.filterDate('2020-01-01', '2023-12-31')\
    .select(['Percent_Tree_Cover']).median()

# Combine bands into a single multiband image
predictors = BIO.addBands(Terrain)\
    .addBands(MedianPTC)\
    .addBands(PH)\
    .addBands(SO2)\
    .addBands(LCT)

# Mask ocean pixels in the predictor image
watermask = Terrain.select('elevation').gt(0)  # Create a water mask
predictors = predictors.updateMask(watermask).clip(AOI)

# Select the subset of bands to keep for habitat suitability modeling
bands = ['bio04', 'bio05', 'bio06', 'bio12', 'elevation', 'Percent_Tree_Cover', 'b100', 'LC_Type1', 'SO2_column_number_density']
# predictors = predictors.select(bands)

flow_accumulation_collection = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/base-network-layers/flow_accumulation")
flow_direction = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/base-network-layers/flow_direction")
drainage_basin = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/base-network-layers/drainage_basin").mosaic()
sub_catchment = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/base-network-layers/sub_catchment").mosaic()
flow_accumulation = flow_accumulation_collection.mosaic().select('b1')
drainage_direction = flow_direction.mosaic().select('b1')
flow_direction = flow_direction.mosaic()

flow_accumulation = flow_accumulation.select(['b1']).rename(['b1_flow_accumulation'])
flow_direction = flow_direction.select(['b1']).rename(['b1_flow_direction'])
drainage_basin = drainage_basin.select(['b1']).rename(['b1_drainage_basin'])
sub_catchment = sub_catchment.select(['b1']).rename(['b1_sub_catchment'])
drainage_direction = drainage_direction.select(['b1']).rename(['b1_drainage_direction'])

outlet_diff_dw_basin = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/outlet_diff_dw_basin").mosaic()
outlet_diff_dw_scatch = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/outlet_diff_dw_scatch").mosaic()
outlet_dist_dw_basin = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/outlet_dist_dw_basin").mosaic()
outlet_dist_dw_scatch = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/outlet_dist_dw_scatch").mosaic()
stream_diff_dw_near = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_diff_dw_near").mosaic()
stream_diff_dw_far = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_diff_up_farth").mosaic()
stream_diff_up_near = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_diff_up_near").mosaic()
stream_dist_dw_near = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_dist_dw_near").mosaic()
stream_dist_proximity = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_dist_proximity").mosaic()
stream_dist_up_farth = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_dist_up_farth").mosaic()
stream_dist_up_near = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-outlet-distance/stream_dist_up_near").mosaic()

outlet_diff_dw_basin = outlet_diff_dw_basin.select(['b1']).rename(['b1_outlet_diff_dw_basin'])
outlet_diff_dw_scatch = outlet_diff_dw_scatch.select(['b1']).rename(['b1_outlet_diff_dw_scatch'])
outlet_dist_dw_basin = outlet_dist_dw_basin.select(['b1']).rename(['b1_outlet_dist_dw_basin'])
outlet_dist_dw_scatch = outlet_dist_dw_scatch.select(['b1']).rename(['b1_outlet_dist_dw_scatch'])
stream_diff_dw_near = stream_diff_dw_near.select(['b1']).rename(['b1_stream_diff_dw_near'])
stream_diff_dw_far = stream_diff_dw_far.select(['b1']).rename(['b1_stream_diff_dw_far'])
stream_diff_up_near = stream_diff_up_near.select(['b1']).rename(['b1_stream_diff_up_near'])
stream_dist_dw_near = stream_dist_dw_near.select(['b1']).rename(['b1_stream_dist_dw_near'])
stream_dist_proximity = stream_dist_proximity.select(['b1']).rename(['b1_stream_dist_proximity'])
stream_dist_up_farth = stream_dist_up_farth.select(['b1']).rename(['b1_stream_dist_up_farth'])
stream_dist_up_near = stream_dist_up_near.select(['b1']).rename(['b1_stream_dist_up_near'])

cti = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti").mosaic()
spi = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/spi").mosaic()
sti = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/sti").mosaic()

cti = cti.select(['b1']).rename(['b1_cti'])
spi = spi.select(['b1']).rename(['b1_spi'])
sti = sti.select(['b1']).rename(['b1_sti'])

channel_elv_dw_cel = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-channel/channel_elv_dw_cel").mosaic()
channel_dist_dw_seg = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-channel/channel_dist_dw_seg").mosaic()
channel_dist_up_seg = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-channel/channel_dist_up_seg").mosaic()
channel_dist_up_cel = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/stream-channel/channel_dist_up_cel").mosaic()

channel_elv_dw_cel = channel_elv_dw_cel.select(['b1']).rename(['b1_channel_elv_dw_cel'])
channel_dist_dw_seg = channel_dist_dw_seg.select(['b1']).rename(['b1_channel_dist_dw_seg'])
channel_dist_up_seg = channel_dist_up_seg.select(['b1']).rename(['b1_channel_dist_up_seg'])
channel_dist_up_cel = channel_dist_up_cel.select(['b1']).rename(['b1_channel_dist_up_cel'])

predictors = (flow_accumulation
              .addBands(flow_direction)
              .addBands(drainage_basin)
              .addBands(sub_catchment)
              .addBands(drainage_direction)
              .addBands(slope_curv_max_dw_cel)
              .addBands(slope_curv_max_dw_cel)
              .addBands(slope_curv_min_dw_cel)
              .addBands(slope_elv_dw_cel)
              .addBands(slope_grad_dw_cel)
              .addBands(outlet_diff_dw_basin)
              .addBands(outlet_diff_dw_scatch)
              .addBands(outlet_dist_dw_basin)
              .addBands(outlet_dist_dw_scatch)
              .addBands(stream_diff_dw_near)
              .addBands(stream_diff_dw_far)
              .addBands(stream_diff_up_near)
              .addBands(stream_dist_dw_near)
              .addBands(stream_dist_proximity)
              .addBands(stream_dist_up_farth)
              .addBands(stream_dist_up_near)
              .addBands(cti)
              .addBands(spi)
              .addBands(sti) 
              .addBands(channel_elv_dw_cel) 
              .addBands(channel_dist_dw_seg) 
              .addBands(channel_dist_up_seg) 
              .addBands(channel_dist_up_cel) 
              
              .addBands(BIO)
              .addBands(Terrain)
              .addBands(MedianPTC)
              .addBands(PH)
              .addBands(SO2)
              .addBands(LCT)
             )

bands = ['b1_flow_accumulation',
         'b1_flow_direction',
         'b1_drainage_basin',
         'b1_sub_catchment',
         'b1_drainage_direction',
         'b1_slope_curv_max_dw_cel',
         'slope_curv_min_dw_cel',
         'b1_slope_elv_dw_cel',
         'b1_slope_grad_dw_cel',
         'b1_outlet_diff_dw_basin',
         'b1_outlet_diff_dw_scatch',
         'b1_outlet_dist_dw_basin',
         'b1_outlet_dist_dw_scatch',
         'b1_stream_diff_dw_near',
         'b1_stream_diff_dw_far',
         'b1_stream_diff_up_near',
         'b1_stream_dist_dw_near',
         'b1_stream_dist_proximity',
         'b1_stream_dist_up_farth',
         'b1_stream_dist_up_near',
         'b1_cti',
         'b1_spi',
         'b1_sti',    
         'elevation',
         'bio04',
         'bio05',
         'bio06',
         'bio12',
         'Percent_Tree_Cover',
         'b100', 
         'LC_Type1'  
    
]


##################################################################################################################
# This code performs two main operations using the previously created predictor variables.
# The first operation is to randomly sample a number of pixels from the predictors image. This is done using
# the "sample" method, which takes as arguments the sampling scale (in meters), the number of pixels to be 
# sampled, and a boolean value to indicate whether the geometries (in this case, the area of interest defined by 
# the "DataCor" variable) should be included in the sampling results.
# The second operation is to sample the values of the predictor variables at points of interest defined by 
# the "DataCor" variable. This is done using the "sampleRegions" method, which takes as arguments the collection of 
# points of interest, the sampling scale (in meters), and the "tileScale" (which controls the size of the tiles used 
# for parallel processing). The result of this operation is a data table containing the values of all predictor 
# bands for each point of interest in the study area.
##################################################################################################################

DataCor = predictors.sample(scale=GrainSize, numPixels=5000, geometries=True) # Generate 5000 random points
PixelVals = predictors.sampleRegions(collection=DataCor, scale=GrainSize, tileScale=16) # Extract covariate values

##############################################################################################################
# This code aims to create a mask for the area of interest (AOI) by applying a segmentation based on clustering
# (K-means), using a random sample of pixels for grouping.
# Reduce the Data table to a binary image where each pixel is equal to 1 if there is at least one non-null value 
# in the 'random' column, and 0 otherwise. The image is reprojected to the EPSG:4326 projection and with the 
# resolution specified in GrainSize. The image is masked to exclude pixels that are outside the area of interest.
# Randomly sample 200 pixels from the predictors image using the sampleRegions function.
# Use the K-means clustering algorithm from the Weka package to cluster these sampled pixels into 2 distinct groups.
# Create an image in which each pixel belongs to one of the two groups and is represented by a random color.
# This image is added to the map.
# Randomly select another 200 pixels from the clustered image and assign each the corresponding cluster number.
# Then, calculate the mode of the cluster in which these pixels fall.
# Use the binary image generated in step 1 to create a mask and apply a second mask based on the mode of the 
# cluster calculated in step 5. Return a clipped image to the area of interest representing the area for which 
# preservation actions can be recommended based on the generated clusters.
##############################################################################################################

mask = Data.reduceToImage(
    properties=['random'],
    reducer=ee.Reducer.first()
).reproject('EPSG:4326', None, ee.Number(GrainSize)).mask().neq(1).selfMask()

# Extract environmental values for a random subset of presence data
PixelVals = predictors.sampleRegions(
    collection=Data.randomColumn().sort('random').limit(200),
    properties=[],
    tileScale=16,
    scale=GrainSize
)

# Perform k-means clustering and train it based on Euclidean distance.
clusterer = ee.Clusterer.wekaKMeans(
    nClusters=2,
    distanceFunction="Euclidean"
).train(PixelVals)

# Assign pixels to clusters using the trained clusterer
Clresult = predictors.cluster(clusterer)

# Display the cluster results and identify the cluster IDs for pixels similar to and different from the presence data
right = ee.Image(0).addBands(Clresult.randomVisualizer())
# Map.addLayer(right, {}, 'Clusters', 0)

# Mask pixels that are different from the presence data.
# Obtain the cluster ID similar to the presence data and use the opposite cluster to define the permitted area
# to create pseudo-absences
clustID = Clresult.sampleRegions(
    collection=Data.randomColumn().sort('random').limit(200),
    properties=[],
    tileScale=16,
    scale=GrainSize
)
clustID = ee.FeatureCollection(clustID).reduceColumns(
    reducer=ee.Reducer.mode(),
    selectors=['cluster']
)
clustID = ee.Number(clustID.get('mode')).subtract(1).abs()
mask2 = Clresult.select(['cluster']).eq(clustID)
AreaForPA = mask.updateMask(mask2).clip(AOI)

################################################################################################################
# This code defines the function makeGrid that creates a grid of polygonal cells of size defined by the scale 
# parameter within a given geometry defined by geometry. To create the grid, the code uses the lonLat image from 
# the Google Earth Engine (GEE) platform, which contains longitude and latitude information for each pixel in the 
# image. Then, this image is used to create lonGrid and latGrid images that contain grids of longitude and latitude, 
# respectively.
# The reduceToVectors function is then applied to the grid of polygonal cells to calculate the average value of a 
# given image (watermask in this case) within each polygonal cell. The filter function is used to remove cells 
# without value (where the mean is equal to None). The final result is a feature collection (Grid) containing 
# the polygonal cells with the average value of the watermask image.
################################################################################################################

# Define a function to create a grid over the AOI
def makeGrid(geometry, scale):
    # pixelLonLat returns an image with each pixel labeled with longitude and latitude values.
    lonLat = ee.Image.pixelLonLat()
    # Select the longitude and latitude bands, multiply by a large number, and truncate to integers.
    lonGrid = lonLat.select('longitude') \
                   .multiply(100000) \
                   .toInt()
    latGrid = lonLat.select('latitude') \
                   .multiply(100000) \
                   .toInt()
    return lonGrid.multiply(latGrid) \
                .reduceToVectors(geometry=geometry.buffer(distance=20000, maxError=1000), # The buffer allows you to check if the grid includes the AOI boundaries.
                                 scale=scale,
                                 geometryType='polygon')

# Create grid and remove cells outside the AOI
Scale = 200000  # Define the interval in m to create spatial blocks
grid = makeGrid(AOI, Scale)
Grid = watermask.reduceRegions(collection=grid, reducer=ee.Reducer.mean()).filter(ee.Filter.neq('mean', None))

# Define a function to generate a vector of random numbers between 1 and 1000
def runif(length):
    return [random.randint(1, 1000) for i in range(length)]

###################################################################################################################
# This code defines a function SDM(x) that performs a Species Distribution Modeling (SDM) analysis in Google Earth 
# Engine. The purpose of the function is to train a classification model using the Random Forest Classifier to 
# predict the presence or absence of a given species in a given location based on environmental variables such as 
# temperature, precipitation, elevation, among others.
# The function uses presence and absence data of the species (or simulated presence points), as well as training 
# and testing points, which are generated using a spatial grid created by makeGrid(). The function also uses a set 
# of predictor data (or environmental variables) and a Random Forest classification model to generate a 
# classification map of the species (probabilities or binary values). The function returns a list containing the 
# classified probability map, the classified binary map, the training set, and the testing set.
###################################################################################################################

def SDM(x):
    Seed = ee.Number(x)
    
    # Randomly divided blocks for training and validation
    GRID = ee.FeatureCollection(Grid).randomColumn(seed=Seed).sort('random')
    TrainingGrid = GRID.filter(ee.Filter.lt('random', split))  # Filter points with 'random' property < split percentage
    TestingGrid = GRID.filter(ee.Filter.gte('random', split))  # Filter points with 'random' property >= split percentage

    # Presence
    PresencePoints = ee.FeatureCollection(Data)
    PresencePoints = PresencePoints.map(lambda feature: feature.set('PresAbs', 1))
    TrPresencePoints = PresencePoints.filter(ee.Filter.bounds(TrainingGrid))  # Filter presence points for training
    TePresencePoints = PresencePoints.filter(ee.Filter.bounds(TestingGrid))  # Filter presence points for testing
    
    # Pseudo-absences
    TrPseudoAbsPoints = AreaForPA.sample(region=TrainingGrid, scale=GrainSize, numPixels=TrPresencePoints.size().add(300), seed=Seed, geometries=True)  # Add extra points to account for those falling in masked areas of the raster and being discarded. This ensures a balanced presence/pseudo-absence dataset
    TrPseudoAbsPoints = TrPseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TrPresencePoints.size()))  # Randomly retain the same number of pseudo-absences as presence data
    TrPseudoAbsPoints = TrPseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))
    
    TePseudoAbsPoints = AreaForPA.sample(region=TestingGrid, scale=GrainSize, numPixels=TePresencePoints.size().add(100), seed=Seed, geometries=True)  # Add extra points to account for those falling in masked areas of the raster and being discarded. This ensures a balanced presence/pseudo-absence dataset
    TePseudoAbsPoints = TePseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TePresencePoints.size()))  # Randomly retain the same number of pseudo-absences as presence data
    TePseudoAbsPoints = TePseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))

    # Merge presence and pseudo-absence points
    trainingPartition = TrPresencePoints.merge(TrPseudoAbsPoints)
    testingPartition = TePresencePoints.merge(TePseudoAbsPoints)

    # Extract local covariate values from the multiband predictor image at the training points
    trainPixelVals = predictors.sampleRegions(collection=trainingPartition, properties=['PresAbs'], scale=GrainSize, tileScale=16, geometries=True)

    # Classify using random forest
    Classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=500,  # The number of decision trees to be created.
        variablesPerSplit=None,  # The number of variables per split. If not specified, uses the square root of the number of variables.
        minLeafPopulation=10,  # Create only nodes whose training set contains at least this amount of points. Integer number, default: 1
        bagFraction=0.5,  # The fraction of input for bagging per tree. Default: 0.5.
        maxNodes=None,  # The maximum number of leaf nodes in each tree. If not specified, the default is no limit.
        seed=Seed  # Randomization seed.
    )
    
    # Probability of presence
    ClassifierPr = Classifier.setOutputMode('PROBABILITY').train(trainPixelVals, 'PresAbs', bands)
    ClassifiedImgPr = predictors.select(bands).classify(ClassifierPr)

    # Binary map of absence and presence
    ClassifierBin = Classifier.setOutputMode('CLASSIFICATION').train(trainPixelVals, 'PresAbs', bands)
    ClassifiedImgBin = predictors.select(bands).classify(ClassifierBin)

    return ee.List([ClassifiedImgPr, ClassifiedImgBin, trainingPartition, testingPartition])
   
  
###################################################################################################################
# This code defines a variable "split" to determine the proportion of blocks used to select training data.
# Then, it defines the variable "numiter" as 10 and applies the "SDM" function to a list of provided numbers: 
# [35, 68, 43, 54, 17, 46, 76, 88, 24, 12].
# The "SDM" function is called for each of these numbers in the list. For each number, the "SDM" function performs 
# a species distribution modeling analysis. The output of the "SDM" function for each number is stored in a "results" 
# list. Then, the "results" list is flattened into a single list of results.
###################################################################################################################

# Define partition for training and testing data
split = 0.70  # The proportion of blocks used to select training data

# Define the number of repetitions
numiter = 10

# Although the runif function can be used to generate random seeds, we map the SDM function over randomly created numbers
# for reproducibility of results
results = ee.List([35, 68, 43, 54, 17, 46, 76, 88, 24, 12]).map(SDM)

# Extract results from the list
results = results.flatten()

##################################################################################################################
# This code creates a classification model based on a Random Forest algorithm in a collection of images from 
# Google Earth Engine.
# First, the code uses the ee.List.sequence() function to create a list of integers, which is used as indices 
# to access the images generated by the classification algorithm. Then, the mean of these images is calculated 
# using the ee.ImageCollection.fromImages().mean() function, resulting in an average model.
# Finally, a distribution map is calculated, representing the class with the highest frequency in each pixel 
# among the images in the list, using the ee.ImageCollection.fromImages().mode() function.
##################################################################################################################

# Extract all model predictions
images = ee.List.sequence(0, ee.Number(numiter).multiply(4).subtract(1), 4).map(lambda x: results.get(x))

# Calculate the mean of all individual model runs
ModelAverage = ee.ImageCollection.fromImages(images).mean()

# Extract all model predictions
images2 = ee.List.sequence(1, ee.Number(numiter).multiply(4).subtract(1), 4).map(lambda x: results.get(x))

# Calculate the mode of all individual model runs
DistributionMap = ee.ImageCollection.fromImages(images2).mode()

# Export the image to Google Drive
task = ee.batch.Export.image.toDrive(
  image=DistributionMap,  # Object of the export
  description='DistributionMap_Enchente_nasa2024',  # File name
  scale=GrainSize,  # Spatial resolution of the exported raster file
  maxPixels=1e10,
  region=AOI  # Area of interest
)

# Start the export task
task.start()

# Export the image to Google Drive
task = ee.batch.Export.image.toDrive(
  image=ModelAverage,  # Object of the export
  description='ModelAverage_Enchente_nasa_2024',  # File name
  scale=GrainSize,  # Spatial resolution of the exported raster file
  maxPixels=1e10,
  region=AOI  # Area of interest
)

# Start the export task
task.start()
