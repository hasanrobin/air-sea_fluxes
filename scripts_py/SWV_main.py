
# --Requried libraries -- 
from netCDF4 import Dataset
import numpy as np
from glob import glob
import xarray as xr 
import os, sys
import pandas as pd  
import tensorflow as tf 
from netCDF_write import * 
#--------------------------------------------------
#-----> if GPUs avaiable 
os.environ['TF_TRT_DISABLE_CONVERSION'] = '1'  # for tensorRT warning message
gpus=(tf.config.list_physical_devices('GPU'))
print(gpus)
if gpus: 
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True )
#tf.config.set_visible_devices(gpus[0], 'GPU')  # Use only the first GPU
mgpus = tf.distribute.MirroredStrategy() 
print(f"Number of devices: {mgpus.num_replicas_in_sync}")
#----------------------------------------------------

# -----> constant parameters for short wave calculaiton 

alpha= [0.095,0.08,0.065,0.065,0.06,0.06,0.06,0.06,0.065,0.075,0.09,0.10]
tau=0.7 
rpi=3.141592653589793
eclipse =23.439 *(3.141592653589793/180)
solar =1350
aozone=0.09
yrdays=360
rad= 3.141592653589793/180


#----------------------------------------------------------->
#************* function for short wave ***************
#------------------------------------------------------------> 

@tf.function
def Qsw(lt2d_tf,ln2d_tf, var_tf, ):

    pro_id =os.getpid()
    start = datetime.now()
    print("process id"+str(pro_id) + "start at {}".format(start))
     
    ndays=len(ndy)
    n_lat=len(lat1)
    n_lon=len(lon1)
    hrs =hr_len

    ncld= tf.clip_by_value(var_tf*0.01, 0.0, 1)   # cloud normalization 
    print("ccld_tf", ncld.shape)

    qsw_tf = tf.zeros((ndays, hrs, n_lat, n_lon), dtype=tf.float32)
    #qsw = xr.DataArray(np.zeros((ndays,hrs, n_lat,sub_lon)),dims=['time','hours','lat','lon'])

    
    for it in range(ndays):
        for hr in range(hrs): 
            #for it in range(ndays):
            if (it<=30):
                albedo=(alpha[0])
                #print((albedo))
            elif (it>=31 and it<=58):
                albedo=(alpha[1])
            elif (it>=59 and it<=89):
                albedo=alpha[2]
                #print(albedo)
            elif (it>=90 and it<=119):
                albedo=alpha[3]                
                #print(albedo)
            elif (it>=120 and it <=150):
                    albedo=alpha[4]
            elif (it>=151 and it<=180):
                albedo=alpha[5]                                   
            elif (it>=181 and it<=211):
                albedo=alpha[6]
            elif (it>=212 and it<=242):
                albedo=alpha[7]
            elif (it>=243 and it<=272):
                albedo=alpha[8]
            elif (it>=273 and it<=303):
                albedo=alpha[9]
            elif (it>=304 and it <=333):
                albedo=alpha[10]
            elif (it>=334 and it<=364):
                (albedo==alpha[11])

            days= tf.cast(it, tf.float32) 
            sun0=2*tf.constant(np.pi, dtype=tf.float32)*days/ tf.constant(yrdays, dtype=tf.float32) 
            sun2=2*sun0
            sun3=3*sun0
        
            sundec= (0.006918-0.399912*tf.cos(sun0)+0.070257*tf.sin(sun0) - 0.006758*tf.cos(sun2) + 0.000907*tf.sin(sun2)-0.002697*tf.cos(sun3) + 0.001480*tf.sin(sun3))# *(180/rpi) 
            sun_hr = (hr-12)*15*rad+ ln2d_tf
            
            #Cosine of the solat zenith  ange
            coszen =tf.sin(lt2d_tf)*tf.sin(sundec)+tf.cos(lt2d_tf)*tf.cos(sundec)*tf.cos(sun_hr)
        
            coszen= tf.clip_by_value(coszen, 0, float('inf') )
            qatten= tau**(1/tf.maximum(coszen, 5.035E-04)) 

            qzer  = coszen * solar * (1.0+1.67E-2*tf.cos(tf.constant(np.pi, dtype=tf.float32)*2.*(days-3.0)/365.0))**2
            qdir =qzer * qatten
            qdiff = ((1.-aozone)*qzer - qdir)*0.5
            qtot = qdir + qdiff

            tjul= (days-81)*tf.constant(rad, dtype=tf.float32)  
            sunbet=tf.sin(lt2d_tf)*tf.sin(eclipse*tf.sin(tjul)) + tf.cos(lt2d_tf)*tf.cos(eclipse*tf.sin(tjul))
            #sunbet=np.arcsin(sunbet)/rad
            
            qsw_temp = tf.where(
            ncld[it, :, :] < 0.3, qtot * (1 - albedo), qtot * (1 - 0.62 * ncld[it, :, :]) + 0.0019 * sunbet * (1 - albedo))

            qsw_tf = tf.tensor_scatter_nd_update(qsw_tf, [[it, hr]], tf.expand_dims(qsw_temp, axis=0)) 
            
            # if np.all(ncld[it,:,:]<0.3):
            #     qsw[it,hr,:,:] = qtot*(1- albedo)
            # else:
            #     qsw[it,hr,:,:]= qtot*(1-0.62*ncld[it,:,:]) + 0.0019*sunbet*(1- albedo)
         
         #qtot_mn=np.mean(qtot_days, axis=0) 
        
    return qsw_tf

#----------------------------------------------------------->
#************* Main function for Short Wave ***************
#----------------------------------------------------------->

if __name__=='__main__':

    time_start = time.time()
    date_start = datetime.now()

    #--->--------------------reading years from input---<---------------
    inYear= sys.argv[1]
    outYear =sys.argv[2]
    model_name   = sys.argv[3]
    out_dir =  sys.argv[4]

    era5_path= "path/for/input-files" 
    ecmwf_path = "path/for/inputs-files"
    #-----------------------------------------------------------
    # ***** Loading the atmospheric  input files n cloud var *****
    #------------------------------------------------------------
    if dataset='ecmwf':

        atm_comb, lat, lon, nlat, nlon, lsm =  target_years_files(ecmwf_path,inYear, outYear, domain="atm", target_files='ecmwf')
        print("atm_data", len(atm_comb['time'][:]),)

        hr= pd.date_range(start='2006-01-01', periods= len(atm_comb['time']), freq='6H')
        cld = xr.DataArray(atm_comb['TCC'], dims=['time','lat','lon'], coords=[hr, lat1, lon1]).resample(time='D').mean() 
        cld_tf= tf.convert_to_tensor(cld, dtype=tf.float32 )
        cld= cld[:] 

    if dataset='era5':

        atm_comb,lat, lon, nlat, nlon, lsm  =   target_years_files(era5_path,inYear, outYear,  domain='atm', target_files='era5_daily')
        print("atm_data", len(atm_comb['time']),)
    
        dt= pd.date_range(start='2006-01-01', periods=(nday), freq='D')
        cld = xr.DataArray(atm_comb['TCC'],dims=['time','lat','lon'] ,coords=[dt, lat, lon]))
        cld_tf= tf.convert_to_tensor(cld.values, dtype=tf.float32)
        cld_tf= cld_tf[:]
    

    nday=len(atm_comb['time'][:])
    ndy= np.arange(0,nday)
    lsm= atm_comb["LSM"][0,:,:]
    print("Days", nday, "Lat=",nlat, "Lon", nlon, "LSM", np.shape(lsm))
    hours=np.arange(24)
    hr_len =int(len(hours))

    #---------converting lat and lon to 2D arrays------------------
    lat2d = xr.DataArray(np.zeros((nlat,nlon)), dims=['lat','lon'])
    lon2d = xr.DataArray(np.zeros((nlat,nlon)), dims=['lat','lon'])
    # lat2d[:,:] = np.tile(lat1, (nlon,1)).T
    # lon2d[:,:] = np.tile(lon1, (nlat,1)) 

    ilat=0
    ilon=0
    it=0
    for ilat in range(nlat):
        lon2d[ilat,:] =lon[:]
    for ilon in range(nlon):
        lat2d[:,ilon]= lat[:]

    lat2d=np.radians(lat2d)
    lon2d=np.radians(lon2d)

    lat2d_tf= tf.convert_to_tensor(lat2d.values, dtype=tf.float32)
    lon2d_tf= tf.convert_to_tensor(lon2d.values, dtype=tf.float32)

    with mgpus.scope():
        qsw_tf= Qsw(lat2d_tf, lon2d_tf, cld_tf )
    
    swv = xr.DataArray(np.zeros((int(nday),hr_len, nlat,nlon)),dims=['time','hours','lat','lon']) #coords=[ntime,latitude, longitude]
    swv[:]= qsw_tf

    swv= swv*lsm
    print(np.nanmean(solar))

    path_outdata = "path/outdir"    
    write_swv_netcdf(path_outdata,nday,lon,np.flip(lat), swv)
        
    time_end = time.time()
    elapsed_time = (time_end - time_start)
    print("elapsed time {:.3f}".format(elapsed_time)+"sec = {:.3f}".format(elapsed_time/60.)+"min")
