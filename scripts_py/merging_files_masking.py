
# --Requried libraries -- 
import sys
import os
import numpy as np
from glob import glob
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
#-------------------------------
# register package system path
package_root_directory = Path(__file__).resolve()
if package_root_directory not in sys.path:
    sys.path.append(str(package_root_directory))

print(package_root_directory)

# ---> Reading netcdf files merging time dimension 
def read_netcdfs_file(files, dim):
    paths = sorted(glob(files))
    datasets =[xr.open_dataset(p) for p in paths]
    combined =xr.concat(datasets, dim)
    return combined

# ---> Loading lat, lon and masking Atlantic Sea part from the Mediterranean

def single_file(file,lati, longi,lsm=None):

    file_read= xr.open_dataset(file)
    #print(file_read)
    lati= file_read[lati]
    longi= file_read[longi]

    nlat=len(lati)
    nlon=len(longi)

    if (lsm=="ecmwf"):
        lsm = file_read.LSM[0].values
        lsm[lsm[:,:]==1]=np.nan 
        lsm[lsm[:,:]==0]=1

        lsm[0:31,0:56]=np.nan
        lsm[0:20,90:146]=np.nan
        lsm[0:54,265:357]=np.nan
        lsm[60:80,300:350]=np.nan
        lsm[100:110,100:120]=np.nan
        lsm[0:5,198:202]=np.nan
        lsm[118:128,340:350]=np.nan
        lsm[130:,310:320]=np.nan
        lsm[41:50,215:225]=np.nan
        lsm[78:105,0:10]=np.nan  

    if np.all(lsm=="era5"):
        lsm = file_read.LSM.values
        lsm[lsm[:,:,:]>0.2]=np.nan
        lsm[lsm<0.2]=1
        
        lsm[:,30:40,158:165]=np.nan
        lsm[:,40:77,135:177]=np.nan
        lsm[:,0:31,183:245]=np.nan
        lsm[:,52:70,0:25]=np.nan
        #lsm[:,0:54,264:357]=np.nan

    if  np.all(lsm=="era_intermin"):
        lsm = file_read.LSM.values
        lsm[lsm[:,:,:]==32767]=np.nan
        lsm[lsm[:,:,:]==-32765]=1
        lsm[0,0:60,0:18]=np.nan
        lsm[0,0:10,60:85]=np.nan
        lsm[0,0:7,0:25]=np.nan
    #print(lati, longi, nlat,nlon)
    return lati, longi, nlat, nlon,lsm 


# -----> Loading target years files into a combined array 

def target_years_files(path, yr_start=None, yr_end=None,in_mon=None, out_mon=None, domain=None, target_files=None, comb_files=False):

    in_yr= str(yr_start)
    end_yr = str(yr_end)
    i_mon=1 # int(in_mon)
    o_mon=12 #int(out_mon)
  
    atm_path= path
    
    yrlist= []
    for r in range(int(in_yr),int(end_yr)+1):
        for s in range(int(i_mon),int(o_mon)+1):
            yrlist.append('%02i' %r + '%02i' %s)
    print("yr-month=",yrlist)

    filelist=sorted(list())
    for (path, dirs, files) in os.walk(atm_path):
        filelist+=[file for file in files]

    target_atm= list()
    for f in filelist: 
        if domain=="atm":
            for g in range(len(yrlist)):
                if target_files == 'ecmwf':
                    if f.startswith(str(yrlist[g])): 
                        target_atm.append(f)
                elif target_files=="era5"+"_hrly":
                    if f.startswith(str(yrlist[g])): # change ERA5 initial file names 
                        target_atm.append(f)
                elif target_files=='era5'+'_daily':
                    if f.startswith("ERA5_dm_"+ str(yrlist[g])): # change ERA5 initial file names 
                        target_atm.append(f)
        # split target_files string and read first part only here 
        
    
        if domain=='sst':
            for g in range(len(yrlist)):
                if f.startswith(str(yrlist[g])): # change ERA5 initial file names 
                            target_atm.append(f)

    input_atm= sorted(target_atm)
    print("input_files_sorted=",len(input_atm))

    i=0
    fls =[]
    for i in range(len(input_atm)):
        fl= atm_path+ input_atm[i]
        fls.append(fl)

    print(fls[0])
    print(fls[-1]) 
    print("files_in_list=",len(fls))

    if comb_files:
        print("Combining files")
        comb_files = xr.open_mfdataset(
            fls,
            combine='nested',
            parallel=False,
            concat_dim='time',
            chunks={'time': 100}  # Adjust chunk size as needed
        )

    if domain == 'atm':
        fls= sorted(fls)
        if target_files == 'era5_hrly':
            lsm_type = 'era5'
        elif target_files == 'era5_daily':
            lsm_type = 'era5'
        else:
            lsm_type = target_files
        lat, lon, nlat, nlon, lsm = single_file(fls[-1], 'lat', 'lon', lsm=lsm_type)
        print("nlat=",nlat, "nlon=",nlon, "lsm=", np.shape(lsm)) 
        if comb_files:
            return comb_files, lat, lon, nlat, nlon, lsm
        else:
            print("Return file list")
            return fls, lat, lon, nlat, nlon, lsm 

    if domain == 'sst':
        if comb_files :
            return comb_files
    
