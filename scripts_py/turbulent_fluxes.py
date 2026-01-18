# load the libraries 
import numpy as np
from netCDF4 import Dataset
import os 
import sys
import time
from datetime import datetime, date, timedelta
from glob import glob, glob1 
import warnings
warnings.filterwarnings("ignore")
import tensorflow  as tf  

#------------------------------------------------
# fixed parameters 
ps     = 1013         #surface air pressure
expsi  = 0.622        # expsi
rd     = 287          # dry air gas constant
rhoa   = 1.22         # air density
emic   = 1            # emissivity of snow or ice
vkarmn = 0.4          #von Karman constant CC
grav   = 9.80665 
stefan = 5.67e-8 
cp     = 1005         # specific heat capacity 
onsea  = 0.98         # specific humadity factors
param1 = 640380       # kg/m3
param2 = -5107.4


a_h = tf.constant([0.0,0.927,1.15,1.17,1.652], dtype=tf.float32)
b_h = tf.constant([1.185,0.0546,0.01,0.0075,-0.017], dtype=tf.float32)
c_h = tf.constant([0.0,0.0,0.0,-0.00045,0.0], dtype=tf.float32)
p_h = tf.constant([-0.157,1.0,1.0,1.0,1.0], dtype=tf.float32)

a_e = tf.constant([0.0,0.969,1.18,1.196,1.68], dtype=tf.float32) 
b_e = tf.constant([1.23,0.0521,0.01,0.008,-0.016], dtype=tf.float32) 
c_e = tf.constant([0.0,0.0,0.0,-0.0004,0.0], dtype=tf.float32) 
p_e = tf.constant([-0.16,1.0,1.0,1.0,1.0], dtype=tf.float32) 

###---------------------------------------------------------->
# ---------- Sensibile/latent  Heat calculaiton funciton -----------
###---------------------------------------------------------->

@tf.function 
def turbulent( t2m_tensor, win_tensor, sst_tensor, shms_tensor, sht_tensor):
    
    shmsnow=shms_tensor
    tnow= t2m_tensor
    wnnow= win_tensor
    sstnow= sst_tensor
    shtnow=sht_tensor
    
    wair = shmsnow/(1 - shmsnow)
    vtnow = (tnow*(expsi+wair))/(expsi*(1 +wair))
    rhom= 100*(ps/rd)/vtnow
    deltatemp = sstnow - tnow
    s = deltatemp/(wnnow**2)
    stp = s*tf.abs(s)/(tf.abs(s)+0.01)
    
    fh = tf.where((s < 0) & (stp > -3.3) & (stp <0.), 0.1 +0.03*stp+0.9*tf.exp(4.8*stp), 1.0 +0.63*tf.sqrt(stp))
    fe = tf.where((s < 0) & (stp < -3.3), 0.0, fh)
    
    kku = tf.where(tf.logical_and(wnnow >= 0, wnnow <= 2.2), 1,
                tf.where(tf.logical_and(wnnow > 2.2, wnnow <= 5.0), 2,
                        tf.where(tf.logical_and(wnnow > 5.0, wnnow <= 8.0), 3,
                                tf.where(tf.logical_and(wnnow > 8.0, wnnow <= 25.0), 4,
                                        tf.where(wnnow > 25.0, 5, 0)
                                        ))))  
    
    max_index_a = tf.shape(a_h)[0] - 1  # Assuming a_h is 1D.  Adjust if not.
    max_index_b = tf.shape(b_h)[0] - 1
    max_index_c = tf.shape(c_h)[0] - 1
    max_index_p = tf.shape(p_h)[0] - 1

    
    kku = tf.clip_by_value(kku, 0, tf.reduce_min([max_index_a, max_index_b, max_index_c, max_index_p]))
    
    if sensible:
        print("starting sensible heat calculation &", "input variables are wind, sst, shms, air temp") 
        ch = ((tf.gather(a_h, kku) + tf.gather(b_h,kku) * wnnow **  tf.gather(p_h, kku) + tf.gather(c_h, kku) * ((wnnow-8) **2)) *fh) / 1000.0
        ch = tf.where(wnnow<0.3, 1.3e-03 * fh, ch)
        ch = tf.where(wnnow>50.0, 1.25e-03 * fh, ch)
        sen_heat = -rhom*cp*ch*wnnow*deltatemp
        tf.print("Mean sensible heat:", tf.reduce_mean(sen_heat))
      
        return sen_heat  
    
    if latent: 
        print("starting latent heat calculation &", "input variables are wind, sst, shms-air, sht-sst")
        latheat= 2.501e+6  # constant latheat is better in lh computation 
        
        def hlat(t):
            #t= np.asarray(t)
            heat= 2.5008e+6-2.3e+3*(t-273.15)
            return heat

        ce =  ((tf.gather(a_e,kku) + tf.gather(b_e, kku) * wnnow ** tf.gather(p_e, kku) + tf.gather(c_e,kku) * ((wnnow-8)**2))* fe)
        ce =  (ce / 1000.0)

        ce =tf.where(wnnow <0.3, 1.5e-03 * fe, ce) 
        ce= tf.where(wnnow>50.0,1.30e-03 * fe, ce)
        
        esre  = shtnow - shmsnow    
        cseep = ce* wnnow*esre
        elat = -rhom*cseep * latheat  #  hlat(sst_tensor)
    
        return elat 
    