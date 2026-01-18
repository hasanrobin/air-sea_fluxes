
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
from mpi4py import MPI
sys.path.insert(0,"/work/cmcc/mg13420/plot_exercises/jup_nbooks/method_function/")
from  merginig_files_maskinig import * 
# -- if GPUs avaiable 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

#----------------------------------------------------------->
#************* Main function ***************
#-----------------------------_____-------____-------------->

if __name__ == "__main__":
        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("rank=", rank, "size=", size)
    start_time= time.time() 
    time_now = datetime.now() 
    print("time_now=", "{}".format(time_now.strftime("%Y-%m-%d %H:%M:%S"))) 
    
    #-----------------------------------------------
    # -----> target years, dataset, flux variables 
    #-----------------------------------------------

    inYear   = sys.argv[1] 
    outYear  = sys.argv[2]
    model_name = sys.argv[3]
    flux_var = sys.argv[4]
    out_dir =  sys.argv[5]

    dataset = model_name
    era5_path= "/"  # atmospheric files dir 
    sst_path= "/"    # sst files dir

    #--------------------------------------------------------
    # ***** Loading the atmospheric and SST input files *****
    #--------------------------------------------------------
    if dataset == 'ecmwf':

        atm_comb, lat, lon, nlat, nlon, lsm =  target_years_files(ecmwf_path,inYear, outYear, domain="atm", target_files='ecmwf')
        sst_comb = target_years_files(sst_path, inYear, outYear, domain='sst', target_files=None )
        print("atm_data", len(atm_comb['time'][:]), "sst_data",len(sst_comb['time']))
    
        # -----> compute  dailiy mean of the ECMWF atmospheric variables
        dt1= pd.date_range(start='2006-01-01', periods= len(atm_comb['time']), freq='6H')
        u10= xr.DataArray(atm_comb['U10M'], dims=['time','lat','lon'], coords=[dt1, lat1, lon1]).resample(time='D').mean() 
        v10= xr.DataArray(atm_comb['V10M'], dims=['time','lat','lon'], coords=[dt1, lat1, lon1]).resample(time='D').mean()
        t2m= xr.DataArray(atm_comb['T2M'], dims=['time','lat','lon'], coords=[dt1, lat1, lon1]).resample(time='D').mean()
        d2m = xr.DataArray(atm_comb['D2M'], dims=['time','lat','lon'], coords=[dt1, lat1, lon1]).resample(time='D').mean()
        #mslp= xr.DataArray(atm_comb['MSL'], dims=['time','lat','lon'], coords=[dt2, lat1, lon1]).resample(time='D').mean()*0.01
        windm= np.sqrt(u10**2 + v10**2) 
        windm= xr.DataArray(windm, dims=['time','lat','lon'], coords=[dt1, lat1, lon1])

        # -----> computing spcefic humadity of surface and air temp 
        sht =  (1/1.22)*onsea*param1*np.exp(param2/sst)
        shms = (1/1.22)*param1*np.exp(param2/d2m)  
        shms= xr.DataArray(shms, dims=['time','lat','lon'],) 

        print( "sst", np.shape(sst), "t2m", np.shape(t2m), "windm", np.shape(windm), "mslp", np.shape(mslp), "d2m", np.shape(d2m))


    if dataset == 'era5':

        atm_comb,lat, lon, nlat, nlon, lsm  =   target_years_files(era5_path,inYear, outYear,  domain='atm', target_files='era5_daily')
        sst_comb =   target_years_files(sst_path, inYear, outYear, domain='sst', target_files='sst')
        print("atm_data", len(atm_comb['time']), "sst_data",len(sst_comb['time']))

        dt2=pd.date_range(start='2006-01-01', periods=len(atm_comb['time']), freq='D')
        d2m=xr.DataArray(atm_comb['D2M'], dims=['time','lat','lon'], coords=dict(time=dt2))
        t2m=xr.DataArray(atm_comb['T2M'],dims=['time','lat','lon'],coords=dict(time=dt2))
        msl=xr.DataArray(atm_comb['MSL'],dims=['time','lat','lon'],coords=dict(time=dt2))
        windu=xr.DataArray(atm_comb['U10M'],dims=['time','lat','lon'],coords=dict(time=dt2))
        windv=xr.DataArray(atm_comb['V10M'],dims=['time','lat','lon'],coords=dict(time=dt2))

        windm = np.sqrt((windu[:,:,:]**2 + windv[:,:,:]**2))
        
        print(  "t2m", np.shape(t2m), "windm", np.shape(windm), "mslp", np.shape(msl), "d2m", np.shape(d2m))

        # -----> computing spcefic humadity of surface and air temp 
        sht =  (1/1.22)*onsea*param1*np.exp(param2/sst)   # specific humadity of surface water temp 
        shms = (1/1.22)*param1*np.exp(param2/d2m) 
        shms = xr.DataArray(shms, dims=['time','lat','lon'])  
    

    ndays=len(atm_comb['time'])
    print(ndays, "Lat=",nlat, "Lon", nlon)
    
    # --- Choose which SST variable to use based on availability in sst_comb
    if 'sst' in sst_comb:
        sst = xr.DataArray(sst_comb['sst'], dims=['time','lat','lon'], coords=[dt2, lat, lon])
    elif 'analysed_sst' in sst_comb:
        sst = xr.DataArray(sst_comb['analysed_sst'], dims=['time','lat','lon'], coords=[dt2, lat, lon])
    else:
        raise KeyError("Neither 'sst' nor 'analysed_sst' found in sst_comb")
    

    #-------------------------------------------------------
    # ***** Converting xarrays to tf arrays *****
    #-------------------------------------------------------
    

    t2m_tensor =  tf.convert_to_tensor(t2m.values, dtype=tf.float32)
    win_tensor  = tf.convert_to_tensor(windm.values, dtype=tf.float32)
    shms_tensor = tf.convert_to_tensor(shms.values, dtype=tf.float32)
    d2m_tensor = tf.convert_to_tensor(d2m.values, dtype=tf.float32)
    sht_tensor = tf.convert_to_tensor(sht.values, dtype=tf.float32 )
    sst_tensor = tf.convert_to_tensor(sst.values, dtype=tf.float32)

    ndays = len(t2m.time)
    print("Number of Days=", ndays)

    sub_lon = np.array_split(range(nlon), size) 
    sub_lon_part = sub_lon[rank] 
    comm.barrier() 

    # for i in range(rank, len(sub_lon), size):
    #     queue.append(sub_lon[i])
    #     print("I am rank {:d}, elaboraiton {:d}".format(rank, len(queue )) )

    if rank == 0:
        print(" Number of elaboration {:d}, running on {:d} cores".format(len(sub_lon_part), size))

    # Convert after data slicing
    t2m_sensor  = tf.gather(t2m_tensor,   sub_lon_part, axis=-1) 
    win_tensor  = tf.gather(win_tensor,   sub_lon_part, axis=-1)
    sst_tensor  = tf.gather(sst_tensor,   sub_lon_part, axis=-1)
    shms_tensor = tf.gather(shms_tensor,  sub_lon_part,  axis=-1)
    sht_tensor  = tf.gather(sht_tensor,   sub_lon_part, axis=-1)

    if flux_var='sensible':
        sen_heat = turbulent(t2m_tensor, win_tensor, sst_tensor, shms_tensor, sht_tensor)
        sen_heat_np = sen_heat.numpy()  
        heat_flat = sen_heat_np.flatten() 
       
    if flux_var='latent':
        latent_heat =turbulent(t2m_tensor, win_tensor, sst_tensor, shms_tensor,sht_tensor) 
        latent_heat_np= latent_heat.numpy()
        heat_flat=latent_heat_np.flatten()

    if flux_var='longwave':
        long_wave =turbulent(t2m_tensor, win_tensor, sst_tensor, shms_tensor,sht_tensor) 
        long_wave_np= long_wave.numpy()
        heat_flat=long_wave_np.flatten() 

    if rank == 0:

        heat_flat_all = np.empty((ndays*nlat*nlon), dtype=np.float32)
        comm.Gather(heat_flat, heat_flat_all, root=0) 
        
    if rank == 0:
       
        heat_all = heat_flat_all.reshape((nd,nlat,nlon)) 

        path_outdata= out_dir
        
        if flux_var=='sensible':

            heat_all = xr.DataArray(heat_all, dims=['time','lat','lon'], coords=[d2m.time, lat1, lon1], name='sen_heat')
    
            print("Mean Sensible Heat", np.nanmean(heat_all))
            print("Variable shape", np.shape(heat_all)) 
            sh_flux = heat_all
            
            #---> Save as .npy file
            np.save(os.path.join(path_outdata, "{}_{}_{}_{}.npy".format(dataset, flux_var, inYear, outYear)), sh_flux.values)

            #---> Write to NetCDF
            write_netcdf(path_outdata, nd, lat1, lon1, dataset, flux_var, inYear, outYear)
         
        if flux_var == 'latent':

            heat_all = xr.DataArray(heat_all, dims=['time','lat','lon'], coords=[d2m.time, lat1, lon1], name='latent_heat')

            print("Mean Latent heat", np.nanmean(heat_all))
            print("Variable shape", np.shape(heat_all)) 
            lh_flux = heat_all
            
            #-----> Save as .npy file 
            np.save(os.path.join(path_outdata, "{}_{}_{}_{}.npy".format(dataset, flux_var, inYear, outYear)), lh_flux.values)
            
            #---> Write to NetCDF 
            write_netcdf(path_outdata, nd, lat1, lon1, dataset, flux_var, inYear, outYear)
       
       if flux_var == 'longwave':
            
            heat_all = xr.DataArray(heat_all, dims=['time','lat','lon'], coords=[d2m.time, lat1, lon1], name='longwave')

            print("Mean Longwave", np.nanmean(heat_all))
            print("Variable shape", np.shape(heat_all)) 
            lh_flux = heat_all
            
            #-----> Save as .npy file 
            np.save(os.path.join(path_outdata, "{}_{}_{}_{}.npy".format(dataset, flux_var, inYear, outYear)), lh_flux.values)
            
            #---> Write to NetCDF 
            write_netcdf(path_outdata, nd, lat1, lon1, dataset, flux_var, inYear, outYear)
         

        end = time.time()
        elapsed = end - start_time
        print("Time taken to compute {} heat flux is {:.3f}".format("sensible" if sensible else "latent", elapsed) + " sec " + "{:.3f}".format(elapsed / 60) + " min")

    MPI.Finalize()
