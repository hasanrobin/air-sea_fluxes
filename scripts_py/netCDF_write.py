
import numpy as np
from netCDF4 import Dataset

#-------------------------------------------------------------->
#--------------> Fucntion to write netcdf file <------------------
#-------------------------------------------------------------->

def write_netcdf(path_outdata,ntime,latitude,longitude, hflux, dataset, hflux_var,inYr,endYr):
    
    nlon = len(longitude)
    nlat = len(latitude)

    yr_exp=inYr
    yr_end= endYr 
    path="path/to/outfile"  
    file_name = "{}_{}_{}_{}".format(dataset, hflux_var, yr_exp, yr_end) +".nc"
    path_outdata= path+file_name
    
    print(path_outdata)
    
    ncfile = Dataset(path_outdata, 'w', format='NETCDF3_CLASSIC')
    
    #----> Open a new NetCDF file in write mode (specify format NETCDF3)

 	# Create netCDF dimensions (interv, lat and longitude)
    ncfile.createDimension("lat",nlat)
    ncfile.createDimension("lon",nlon)    
    ncfile.createDimension("time",ntime)

    # Create a netCDF variable (coordinate variable and variable).
    lat_nc  = ncfile.createVariable("latitude",np.float32,("lat",))
    lat_nc[:]  = latitude.values
    lat_nc.units  = latitude.attrs['units']

    lon_nc = ncfile.createVariable("longitude", np.float32,("lon",))
    lon_nc[:] = longitude.values
    lon_nc.units = longitude.attrs['units']
    
    time_long= len(heat_all[:,0,0])
    time_nc = ncfile.createVariable("time", np.float32, ("time",))
    time_nc[:] = time_long
    time_nc.units = "days since 2006-01-01 00:00:00"
   
    if dataset == 'sensible': 
        sen_ht = ncfile.createVariable("sensible_heat",np.float32,("time", "lat","lon"))
        sen_ht[:,:,:] = hflux
        #sen_ht.long_name = long1.attrs['long_name'] 

    if dataset == 'latent':
        lat_ht = ncfile.createVariable("latent_heat", np.float32,("time", "lat","lon"))
        lat_ht[:,:,:] = hflux
        #lat_ht.long_name = long1.attrs['long_name']

    if dataset =='longwave':
        lat_ht = ncfile.createVariable("longwave", np.float32,("time", "lat","lon"))
        lat_ht[:,:,:] = hflux
        #lat_ht.long_name = long1.attrs['long_name']
    
    # time_nc.units = ntime.attrs['units']
    # Close the file.
    ncfile.close()


#-------------------------------------------------------------->
#---------> Fucntion to write netcdf file for short wave<-------
#-------------------------------------------------------------->
def write_swv_netcdf(path_outdata,ntime, longitude,latitude,swv,inYr,endYr):
 
    nlon = len(longitude)
    nlat = len(latitude)
    #time = ncfile.variables["time"]
    ntime =ntime 

    yr_exp=inYr
    yr_end= endYr 
    path="path/to/outfile"  
    file_name = "{}_'swv'_{}_{}".format(dataset, yr_exp, yr_end) +".nc"
    path_outdata= path+file_name

    ncfile = Dataset(filepath_outdata,'w', format='NETCDF3_CLASSIC')
    
    # Create netCDF dimensions (interv, lat and longitude)
    ncfile.createDimension("lat",nlat)
    ncfile.createDimension("lon",nlon)
    ncfile.createDimension("time", ntime)
    ncfile.createDimension("hour", hr_len)

    # Create a netCDF variable (coordinate variable and variable).
    lat_nc  = ncfile.createVariable("latitude",np.float32,("lat",))
    lat_nc[:]  = latitude.values
    lat_nc.units  = latitude.attrs['units']
    
    lon_nc = ncfile.createVariable("longitude",np.float32,("lon",))
    lon_nc[:] = longitude.values
    lon_nc.units = longitude.attrs['units'] 
    
    print(swv.dims)
    if swv.dims==3:
        short = ncfile.createVariable("shortwave",np.float32,("time", "lat","lon"))
        short[:,:,:] = short1
        short.long_name = "Short wave radiation"
    else :
        short = ncfile.createVariable("shortwave",np.float32,("time","hour", "lat","lon"))
        short[:,:,:,:] = short1
        short.long_name = "Short wave radiation"

    hr_short = len(short[0,:,0,0])
    hour_nc = ncfile.createVariable("hour", np.float32, ("hour",))
    hour_nc[:] = np.arange(24)  
    
    time_short = len(short[:,0,0,0])
    time_nc = ncfile.createVariable("time", np.float32, ("time",))
    time_nc[:] =np.arange(ntime)

    # Close the file
    ncfile.close()
