import xesmf as xe
import xarray as xr
import numpy as np
import cesm2_lens_utils
import os

def regrid_SMYLE(ds, glat=1, glon=1): # from Jacob's notebook
    """
    Inputs:
        ds: xr.DataArray with coordinates that include TLAT and TLONG
    Returns:
        Regridded xr.DataArray with coordinates lat and lon
    """
    ds = ds.rename(({'ULONG': 'lon', 'ULAT': 'lat'}))
    ds_out = xe.util.grid_global(glon, glat)
    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=True)
    regridded = regridder(ds)
    new_coords = regridded.assign_coords({'y': regridded.lat[:, 0].values, 'x': regridded.lon[0].values})
    return new_coords.drop_vars(['lat', 'lon']).rename({'x': 'lon', 'y': 'lat'})

def LENS_for_regridding():
    # LENS for regridding purposes
    ens_memb_index = 0
    comp = 'atm'; var = 'AREA'
    directory = f'/glade/campaign/cgd/cesm/CESM2-LE/{comp}/proc/tseries/month_1/{var}/'
    
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory, var=var,comp=comp, 
        index_hist = ens_memb_index)
    
    # FOSI is from 1958 to 2020
    CESMLENS_hist_var = ds_var_hist_var[var].sel(time=slice('1958-01', '2015-01')).compute()
    CESMLENS_fut_var = ds_var_fut_var[var].sel(time=slice('2015-02', '2020-12')).compute()
    
    CESMLENS_var = xr.concat(
        [CESMLENS_hist_var, CESMLENS_fut_var], 
        dim='time')
    return CESMLENS_var

def get_ds_var(directory, var, comp, index_hist):
    path_intermed_hist, path_intermed_fut = get_var_paths(directory, var)
    filename_identifier = '.'.join(path_intermed_hist[index_hist].rsplit('.', 5)[1:4])
    index_fut = find_identifier_with_index(path_intermed_hist, filename_identifier)[0][1]
    hist_file_paths = get_hist_file_paths(var,directory, path_intermed_hist, index_hist)
    fut_file_paths = get_fut_file_paths(var, directory, path_intermed_fut, index_fut)
    ds_var_fut = file_path_to_var_ds(fut_file_paths)
    ds_var_hist = file_path_to_var_ds(hist_file_paths)
    return ds_var_hist, ds_var_fut

# Accessing glade CESM LENS2
def get_var_paths(directory, var):
    # Prefixes to match for future and historical datasets
    prefixes_to_match_fut = ['b.e21.BSSP370cmip6.', 'b.e21.BSSP370smbb.']
    prefixes_to_match_hist = ['b.e21.BHISTcmip6.', 'b.e21.BHISTsmbb.']
        
    # Sets to store unique prefixes for future and historical filenames
    prefixes_fut = list()
    prefixes_hist = list()
        
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check and add prefixes for future scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_fut) and filename.endswith('.nc'):
            prefixes_fut.append(filename.rsplit('.', 3)[0])
            
        # Check and add prefixes for historical scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_hist) and filename.endswith('.nc'):
            prefixes_hist.append(filename.rsplit('.', 3)[0])
        
    prefixes_hist_set = set(prefixes_hist)
    sorted_unique_list_hist = sorted(prefixes_hist_set)
    
    prefixes_fut_set = set(prefixes_fut)
    sorted_unique_list_fut = sorted(prefixes_fut_set)
    
    path_intermed_fut = sorted_unique_list_fut
    path_intermed_hist = sorted_unique_list_hist

    return path_intermed_hist, path_intermed_fut

def find_identifier_with_index(prefixes, identifier):
    """
    Find prefixes that contain a specific identifier and their indices.

    Parameters:
        prefixes (list): A list of prefixes to search through.
        identifier (str): The identifier to search for in the prefixes.

    Returns:
        list: A list of tuples containing matching prefixes and their indices.
    """
    matching_prefixes_with_indices = []  # Initialize a list to store matches and their indices

    for index, prefix in enumerate(prefixes):  # Use enumerate to get both index and prefix
        if identifier in prefix:  # Check if the identifier is in the prefix
            matching_prefixes_with_indices.append((prefix, index))  # Add the prefix and index as a tuple

    return matching_prefixes_with_indices  # Return the list of matching prefixes and indices

def get_hist_file_paths(var, directory, path_intermed_hist, index):
    attrib_title = path_intermed_hist[index]
    file_paths = []
    for start_year in range(1850, 2010, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.201001-201412.nc'
    file_paths.append(last_file_path)
    return file_paths

def get_fut_file_paths(var, directory, path_intermed_fut, index):
    attrib_title = path_intermed_fut[index]
    file_paths = []
    for start_year in range(2015, 2095, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.209501-210012.nc'
    file_paths.append(last_file_path)
    return file_paths

def file_path_to_var_ds(file_paths):
    var_ds = xr.open_mfdataset(file_paths, 
                                 concat_dim='time', 
                                 combine='nested', 
                                 parallel=True)
    return var_ds

def calculate_anomalies_trend_features_linear(ds):
    """
    Decompose an SST time series into mean, linear trend, seasonal cycle,
    and detrended/deseasonalized anomalies, then flag anomalies exceeding
    the 90th percentile threshold.

    Fits a 6-coefficient harmonic regression model at each grid point via
    least-squares (mean + linear trend + annual sine/cosine + semiannual
    sine/cosine), then reconstructs each component and computes residual
    anomalies (observed minus full model fit). A per-gridpoint 90th
    percentile threshold (computed over time from the residual anomalies)
    is used to flag anomalously warm periods.

    Parameters
    ----------
    ds : xr.DataArray
        SST (or other tracer) data with dimensions ('time', 'lat', 'lon')
        and a 'time' coordinate with monthly (or similar) datetime values
        accessible via `.dt.year` / `.dt.month`.

    Returns
    -------
    mean : xr.DataArray
        Reconstructed time-mean field, dims ('time', 'lat', 'lon')
        (broadcast constant in time).
    trend : xr.DataArray
        Reconstructed linear trend component, dims ('time', 'lat', 'lon').
    seas : xr.DataArray
        Reconstructed seasonal cycle (annual + semiannual harmonics),
        dims ('time', 'lat', 'lon').
    features_notrend : xr.DataArray
        Detrended/deseasonalized anomalies masked to only retain values
        at or above the local 90th-percentile threshold (all other
        timesteps set to NaN), dims ('time', 'lat', 'lon').
    ssta_notrend : xr.DataArray
        Full detrended/deseasonalized anomaly field (observed minus the
        6-coefficient model fit), dims ('time', 'lat', 'lon'), before
        percentile masking.

    Notes
    -----
    - The regression model design matrix is:
      [1, (year_frac - mean(year_frac)), sin(2*pi*year_frac),
       cos(2*pi*year_frac), sin(4*pi*year_frac), cos(4*pi*year_frac)]
      where year_frac = year + month/12.
    - Coefficients are estimated via the Moore-Penrose pseudo-inverse
      (`np.linalg.pinv`), i.e. ordinary least squares.
    - The 90th-percentile threshold is computed independently at each
      grid point over the full time dimension of `ssta_notrend`.
    - If `ssta_notrend` is chunked (dask-backed), it is rechunked to a
      single chunk along 'time' before the quantile calculation, since
      `.quantile()` requires an unchunked (or fully loaded) time
      dimension.
    """
    sst = ds
    dyr = ds.time.dt.year + ds.time.dt.month/12
    # Our 6 coefficient model is composed of the mean, trend, annual sine and cosine harmonics, & semi-annual sine and cosine harmonics
    model = np.array([np.ones(len(dyr))] + [dyr-np.mean(dyr)] + [np.sin(2*np.pi*dyr)] + [np.cos(2*np.pi*dyr)] + [np.sin(4*np.pi*dyr)] + [np.cos(4*np.pi*dyr)])
    
    # Take the pseudo-inverse of model to 'solve' least-squares problem
    pmodel = np.linalg.pinv(model)
    
    # Convert model and pmodel to xaray DataArray
    model_da = xr.DataArray(model.T, dims=['time','coeff'], coords={'time':sst.time.values, 'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, dims=['coeff','time'], coords={'coeff':np.arange(1,7,1), 'time':sst.time.values})  
    
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(sst), dims=['coeff','lat','lon'], coords={'coeff':np.arange(1,7,1), 'lat':sst.lat.values, 'lon':sst.lon.values})
    
    # Construct mean, trend, and seasonal cycle
    mean = model_da[:,0].dot(sst_mod[0,:,:])
    trend = model_da[:,1].dot(sst_mod[1,:,:])
    seas = model_da[:,2:].dot(sst_mod[2:,:,:])
    
    # compute anomalies by removing all  the model coefficients 
    ssta_notrend = sst-model_da.dot(sst_mod)
    
    # Use the 90th percentile as a threshold and find anomalies that exceed it. 
    if ssta_notrend.chunks:
        ssta_notrend = ssta_notrend.chunk({'time': -1})
    
    threshold = ssta_notrend.quantile(.9, dim=('time'))
    features_notrend = ssta_notrend.where(ssta_notrend>=threshold, other=np.nan)
    return mean, trend, seas, features_notrend, ssta_notrend

def calculate_pop_grid_anomalies_trend_features(ds, pop_grid_ds=None):
    """
    Calculate SST anomalies, trends, and features using POP grid information.
    
    Parameters:
    -----------
    ds : xarray.Dataset or DataArray
        Input SST data (with time, z_t/nlat/nlon or similar dims)
    pop_grid_ds : xarray.Dataset, optional
        POP grid dataset containing TAREA, ULAT, ULON if needed
        
    Returns:
    --------
    tuple of xarray.DataArray:
        (mean, trend, seasonal_cycle, features, anomalies)
    """
    # Ensure we're working with a DataArray
    if isinstance(ds, xr.Dataset):
        var = ds[list(ds.data_vars)[0]]  # take first variable if Dataset
    else:
        var = ds
    
    # Handle POP grid coordinates if provided
    if pop_grid_ds is not None:
        # Rename coordinates to match POP grid if needed
        if 'nlat' in pop_grid_ds.dims and 'nlon' in pop_grid_ds.dims:
            sst = sst.rename({'lat': 'nlat_t', 'lon': 'nlon_t'})
        # Ensure coordinates align
        sst = sst.assign_coords({
            'nlat_t': pop_grid_ds.nlat,
            'nlon_t': pop_grid_ds.nlon
        })
    
    # Calculate decimal years for model
    dyr = var.time.dt.year + var.time.dt.month/12
    
    # Build our 6-coefficient model
    model_components = [
        np.ones(len(dyr)),                         # Mean
        dyr - np.mean(dyr),                        # Trend
        np.sin(2 * np.pi * dyr),                   # Annual sine
        np.cos(2 * np.pi * dyr),                   # Annual cosine
        np.sin(4 * np.pi * dyr),                   # Semi-annual sine
        np.cos(4 * np.pi * dyr)                    # Semi-annual cosine
    ]
    
    model = np.array(model_components)
    pmodel = np.linalg.pinv(model)  # Pseudo-inverse for least squares
    
    # Convert to xarray DataArrays with proper dimensions
    model_da = xr.DataArray(
        model.T, 
        dims=['time', 'coeff'],
        coords={'time': var.time, 'coeff': np.arange(6)}
    )
    
    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=['coeff', 'time'],
        coords={'coeff': np.arange(6), 'time': var.time}
    )
    
    # Calculate model coefficients
    var_mod = xr.DataArray(
        pmodel_da.dot(var),
        dims=['coeff', 'nlat_t', 'nlon_t'],
        coords={'coeff': np.arange(6), 'nlat_t': var.nlat_t, 'nlon_t': var.nlon_t}
    )
    
    # Construct components
    mean = model_da[:, 0].dot(var_mod[0, :, :])
    trend = model_da[:, 1].dot(var_mod[1, :, :])
    seasonal = model_da[:, 2:].dot(var_mod[2:, :, :])
    
    # Calculate anomalies by removing all model components
    var_notrend = var - model_da.dot(var_mod)
        
    return mean, trend, seasonal, var_notrend
