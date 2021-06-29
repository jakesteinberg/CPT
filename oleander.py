# Oleander data 
import numpy as np
import matplotlib.pyplot as plt 
import datetime 
import xarray as xr
import cmocean.cm as cmo
from scipy.io import loadmat
import netCDF4
from tqdm import tqdm
from vincenty import vincenty_inverse # distances on an oblate spheroid 
from toolkit import pol2cart, cart2pol, plot_pro

# -- load desired data (2015-2017 go down to 1000m, look for good data down there?)
# -- adcp
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/OL_9427294.mat'
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/OL_1325426.mat'
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/OL_1424274.mat'
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/OL_1522517.mat'
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/OL_1621870.mat'
# filepath = '/Users/jakesteinberg/Documents/CPT/oleander/oleander_2017_concat.mat'

# -- I've strung together 2004-2017 in matlab: DIMENSIONS [: X depth]
# -- in oleander_concatenate.m we combine different years of data AND select max depth to consider 
filepath = '/Users/jakesteinberg/Documents/CPT/adcp/oleander/oleander_concat.mat'
x1 = loadmat(filepath) 

# -- time bounds 
this_year = str(np.int(np.floor(np.nanmin(x1['out']['time'][0][0])/10000))) + '_' + str(np.int(np.floor(np.nanmax(x1['out']['time'][0][0])/10000)))
# this_year = '2017'

# -- tsg
# x2 = loadmat('/Users/jakesteinberg/Documents/CPT/oleander/oleander_tsg_raw.mat')

# -- bathy
bathy = xr.open_dataset('/Users/jakesteinberg/Documents/CPT/etopo_n_atl.nc') 

# ----------------------
# -- processing knobs --
savee = 0
save_nc = 0
file_out = '/Users/jakesteinberg/Documents/CPT/oleander/' + this_year + '_gridded.nc'
grid_spacing = 2  # km to interpolate to 
adcp_grid = np.arange(240, 1240, grid_spacing)  # was 220 if including gulf stream
sf = 25  # scale factor (this number x grid_spacing) (acceptable number of nans in nan segments)
lev = [10, 25, 60]  # depth level indices to plot 
# ---------------------- 

# -- define variables ADCP ------- 
adcp_lon0 = x1['out']['lon'][0][0] 
adcp_lat0 = x1['out']['lat'][0][0]
adcp_u0 = x1['out']['u'][0][0]
adcp_v0 = x1['out']['v'][0][0]
dep_levs = x1['out']['depth'][0][0][0][0:65]

# -- method for parsing individual years 
# date_order_check = np.nanmin(np.abs(x1['days'][0:np.int(len(x1['days'])/2)]))
# if date_order_check < 180:
#     adcp_days = np.abs(x1['days'])  # something is off by a day or two in dates between tsg and adcp
# else:
#     adcp_days = np.abs(np.abs(x1['days']) - 365)  # something is off by a day or two in dates between tsg and adcp

# -- with years already combined in matlab
adcp_days = x1['out']['time'][0][0]                                                # format = 19980013.5
adcp_day_list = np.unique(np.floor(adcp_days[np.isfinite(adcp_days)]))             # unique number of start days for each sample 

# try alternate method 
good = np.where(np.isfinite(adcp_days))[0]                                         # indicies in adcp_days where where expect data 
good_times = adcp_days[good, 0]                                                    # select finite indices 
time_gaps = np.diff(good_times)                                                    # time gap between good data 
splits_0 = np.where(time_gaps > 0.5)[0]                                            # search where time gaps are greater than 1 (indices of time gap, shorter array than adcp_days)
time_splits = good_times[splits_0]                                                 # 
iii, splits, iiii = np.intersect1d(adcp_days, time_splits, return_indices=True)    # find where splits occur in adcp_days (each split should bound good cruises)
## # split into "by cruise" or "by transect"
## good = np.where(np.isfinite(adcp_days))[0]  # indicies in adcp_days where where expect data 
## splits = np.where(np.diff(good) > 1)[0] 
adcp_time = []
adcp_lon = []
adcp_lat = []
adcp_u = []
adcp_v =[] 
flag = np.ones(len(splits) + 1)
cruise_time = np.nan * np.zeros(len(splits) + 1)
adcp_time_mean = np.nan * np.ones(len(splits) + 1)
test = np.nan * np.ones((len(splits) + 1, 2))
count = 0
for i in range(len(splits) + 1):
    # if i < 1:
    #     st_i = good[0]
    #     en_i = good[splits[i]]
    # elif i == len(splits):
    #     st_i = good[splits[i - 1] + 1]
    #     en_i = good[-1]
    # else:
    #     st_i = good[splits[i - 1] + 1]
    #     en_i = good[splits[i]]  
    if i < 1:  # first case
        st_i = good[0]
        en_i = splits[i] 
    elif i == len(splits): # last case    
        inin = adcp_days[splits[i - 1] + 1:]
        st_i = np.where(np.isfinite(inin))[0][0] + splits[i - 1] + 1
        # st_i = splits[i - 1] + 1
        en_i = good[-1]
    else:  # all middle cases 
        inin = adcp_days[splits[i - 1] + 1:splits[i] + 1]
        st_i = np.where(np.isfinite(inin))[0][0] + splits[i - 1] + 1
        en_i = splits[i]
        test[i, 0] = np.nanmin(adcp_days[splits[i - 1] + 1:splits[i] + 1]) 
        test[i, 1] = np.nanmax(adcp_days[splits[i - 1] + 1:splits[i] + 1])  
        # test shows the start/end times of each cruise (check that I'm splitting correctly by running)
        # test[1:, 0] - test[0:-1, 1]  ... should show time between cruises 
              
    print(str(adcp_days[st_i]) + ' -- ' + str(adcp_days[en_i]))
    # filter by flag (flag identifies incomplete crossing data, i.e. if parsed crossing delta_t < 1 day)
    if (np.abs(adcp_days[en_i] - adcp_days[st_i]) > 1) & (np.abs(adcp_days[en_i] - adcp_days[st_i]) < 5):
        cruise_time[i] = (adcp_days[en_i] - adcp_days[st_i])
        flag[i] = 0
        adcp_lon.append(adcp_lon0[st_i:en_i+1])
        adcp_lat.append(adcp_lat0[st_i:en_i+1])
        adcp_u.append(adcp_u0[st_i:en_i+1, :])
        adcp_v.append(adcp_v0[st_i:en_i+1, :])
        adcp_time.append(adcp_days[st_i:en_i+1])
        adcp_time_mean[count] = np.nanmean(adcp_days[st_i:en_i+1])
        count = count + 1

# number of crossings between NJ and Bermuda (these crossings meet selection criteria)
num_profs = len(adcp_lon)
adcp_time_mean = adcp_time_mean[np.isfinite(adcp_time_mean)]

# -- compute along-track distance and interpolate to grid 
lon_s = np.nanmin(adcp_lon0)
lat_s = np.nanmax(adcp_lat0)
lat_m = (np.nanmin(adcp_lat0) + np.nanmax(adcp_lat0))/2
adcp_u_grid_cart = np.nan * np.ones((len(dep_levs), len(adcp_grid), num_profs))
adcp_v_grid_cart = np.nan * np.ones((len(dep_levs), len(adcp_grid), num_profs))
adcp_u_grid = np.nan * np.ones((len(dep_levs), len(adcp_grid), num_profs))
adcp_v_grid = np.nan * np.ones((len(dep_levs), len(adcp_grid), num_profs))
adcp_lon_grid = np.nan * np.ones((len(adcp_grid), num_profs))
adcp_lat_grid = np.nan * np.ones((len(adcp_grid), num_profs))
good_p = np.zeros((len(dep_levs), num_profs))
dist_save = []
tracker = np.ones(num_profs)
dist_start_rec = np.ones(num_profs)
for i in tqdm(range(num_profs), ncols=100):  # loop over each pass (or track)
    
    # if all data are nans, skip 
    if np.sum(np.isfinite(adcp_lat[i])) < 1:
        dist_save.append(np.array([np.nan]))
        tracker[i] = 0
        continue
    
    # better distance estimator 
    if adcp_lat[i][np.isfinite(adcp_lat[i])][0] > adcp_lat[i][np.isfinite(adcp_lat[i])][-1]:
        this_lon = adcp_lon[i]
        this_lat = adcp_lat[i]
    else: 
        this_lon = np.flip(adcp_lon[i])
        this_lat = np.flip(adcp_lat[i])
        
    this_dist = np.nan * np.ones(2000)
    rho_track = np.nan * np.ones(2000)
    phi_track = np.nan * np.ones(2000)
    if np.isfinite(this_lon[0]):  # if first element is finite
        this_dist[0] = vincenty_inverse([lat_s, lon_s], [this_lat[0], this_lon[0]]).m  # distance from first data point on cruise to port lat/lon
        for j in range(1, len(adcp_lon[i])):
            if np.isnan(this_lon[j]):
                this_dist[j] = np.nan
                dx = np.nan
                dy = np.nan
            elif np.isnan(this_lon[j - 1]):
                last_good = np.where(np.isfinite(this_lon[0:j]))[0][-1]
                this_dist[j] = vincenty_inverse([this_lat[last_good], this_lon[last_good]], [this_lat[j], this_lon[j]]).m
                dx = 1852 * 60 * np.cos(np.deg2rad(this_lat[j])) * (this_lon[j] - this_lon[last_good])
                dy = 1852 * 60 * (this_lat[j] - this_lat[last_good]) 
            elif (this_lon[j] - this_lon[j - 1]) < 0.0001:
                this_dist[j] = np.nan 
                dx = np.nan
                dy = np.nan
            else:
                this_dist[j] = vincenty_inverse([this_lat[j - 1], this_lon[j - 1]], [this_lat[j], this_lon[j]]).m
                dx = 1852 * 60 * np.cos(np.deg2rad(this_lat[j])) * (this_lon[j] - this_lon[j - 1])
                dy = 1852 * 60 * (this_lat[j] - this_lat[j - 1]) 
  
            rho_track[j], phi_track[j] = cart2pol(dx, dy)  # rotate u,v into polar coordinates (rho = magnitude, phi = angle)
            
    else: # if first element is nan
        first_good = np.where(np.isfinite(this_lon))[0][0]
        this_dist[0] = vincenty_inverse([lat_s, lon_s], [this_lat[first_good], this_lon[first_good]]).m
        for j in range(first_good + 1, len(adcp_lon[i])):
            if np.isnan(this_lon[j]):
                this_dist[j] = np.nan
                dx = np.nan
                dy = np.nan
            elif np.isnan(this_lon[j - 1]):
                last_good = np.where(np.isfinite(this_lon[0:j]))[0][-1]
                this_dist[j] = vincenty_inverse([this_lat[last_good], this_lon[last_good]], [this_lat[j], this_lon[j]]).m
                dx = 1852 * 60 * np.cos(np.deg2rad(this_lat[j])) * (this_lon[j] - this_lon[last_good])
                dy = 1852 * 60 * (this_lat[j] - this_lat[last_good])
            elif (this_lon[j] - this_lon[j - 1]) < 0.0001:
                this_dist[j] = np.nan    
                dx = np.nan
                dy = np.nan
            else:
                this_dist[j] = vincenty_inverse([this_lat[j - 1], this_lon[j - 1]], [this_lat[j], this_lon[j]]).m
                dx = 1852 * 60 * np.cos(np.deg2rad(this_lat[j])) * (this_lon[j] - this_lon[j - 1])
                dy = 1852 * 60 * (this_lat[j] - this_lat[j - 1])
            
            rho_track[j], phi_track[j] = cart2pol(dx, dy)  # rotate u,v into polar coordinates (rho = magnitude, phi = angle)    
            
    dist = this_dist[0:len(this_lon)]
    rho_track_1 = rho_track[0:len(this_lon)]  # remove extra elements of array (should be nans after the index equal to the length of this_lon)
    phi_track_1 = phi_track[0:len(this_lon)]
    d_good = np.isfinite(dist)
    if len(d_good) < len(dist):
        print('warning = ' + str(i))
    dist[d_good] = np.cumsum(dist[d_good]) / 1000.0
    
    # if first non nan data point is greater than distance at which I start interpolating  
    dist_start_rec[i] = dist[0]
    if dist[0] > 220:
        dist_save.append(np.array([np.nan]))
        tracker[i] = -1
        continue
    
    if (dist[-1] - dist[0]) < 1000:
        dist_save.append(np.array([np.nan]))
        tracker[i] = -2
        continue    
    
    # otherwise good 
    dist_save.append(dist)

    # - Loop over depths 
    for m in range(len(dep_levs)):  # loop over each depth         
        good = np.where(np.isfinite(adcp_u[i][:, m]))[0]
        if len(good > 10):
            if np.nanmean(adcp_lat[i][0:np.int(len(adcp_lat[i])/2)]) < np.nanmean(adcp_lat[i][np.int(len(adcp_lat[i])/2):]): # dist[0] > 500:
                this_u = np.flip(adcp_u[i][:, m])
                this_v =  np.flip(adcp_v[i][:, m])
            else:
                this_u = adcp_u[i][:, m]
                this_v = adcp_v[i][:, m]
            
            along_track = np.nan * np.ones(len(this_u))
            across_track = np.nan * np.ones(len(this_v))
            for dx in range(len(this_u)):
                rho, phi = cart2pol(this_u[dx], this_v[dx])
                along_track[dx], across_track[dx] = pol2cart(rho, phi - phi_track[dx])  # phi_track is the polar angle the ship is pointing 
                
            # adcp_u_grid_cart[m, :, i] = np.interp(adcp_grid, dist, this_u)
            # adcp_v_grid_cart[m, :, i] = np.interp(adcp_grid, dist, this_v)
            adcp_u_grid[m, :, i] = np.interp(adcp_grid, dist, along_track)
            adcp_v_grid[m, :, i] = np.interp(adcp_grid, dist, across_track)    
        # if this track (at depth m and for track i) has less than __ number of nans         
        if np.sum(np.isnan(adcp_u_grid[m, :, i])) < 50:
            good_p[m, i] = 1
        # test plot 
        # if (i == 40) & (m==10):
        #     f, ax = plt.subplots()
        #     ax.quiver(adcp_lon[i][:, 0], adcp_lat[i][:, 0], )
        #     plot_pro(ax)   
    
    adcp_lon_grid[:, i] = np.interp(adcp_grid, dist, adcp_lon[i][:, 0])
    adcp_lat_grid[:, i] = np.interp(adcp_grid, dist, adcp_lat[i][:, 0])       
           
        
# -- search for good data
# adcp_u_grid size = [m, dist, i] = [depth, dist, track]
# look at each track at each depth and inventory the number of nans, but also the number of nan segments and the segment lengths 
seg_out = {}
seg_out_count = np.nan * np.ones((len(dep_levs), num_profs))
viable = np.nan * np.ones(len(dep_levs))
for m in range(len(dep_levs)):  # loop over depths 
    for i in range(num_profs):  # loop over tracks
        this_track = adcp_u_grid[m, :, i]
        bad = np.where(np.isnan(this_track))[0]  # nan indices 
        seg = []
        if ((len(bad) > 2) & (len(bad) < len(this_track))):
            breaky = np.where(np.diff(bad) > 1)[0] + 1  # look for breaks in list of nans 
            if len(breaky) > 0:      
                seg.append([bad[0], bad[breaky[0] - 1]])
                for b in range(len(breaky) - 1):
                    seg.append([bad[breaky[b]], bad[breaky[b + 1] - 1]])
                seg.append([bad[breaky[-1]], len(this_track)])    
            elif (len(breaky) == 0) & (bad[0] > 0): 
                seg = [bad[0], bad[-1]]    
            elif (len(breaky) == 0) & (bad[0] == 0):
                seg = [bad[0], bad[-1]]   
                    
        elif len(bad) == len(this_track):
            seg = len(adcp_grid)  # all are nan's 
        else:
            seg = 0  # none are nans     
        seg_out[m, i] = seg    
        # inspect seg_out to see which are good and which might meet some defined criteria 
        if (seg != 0) & (seg != len(adcp_grid)):
            spacer = np.nan * np.ones(len(seg))
            if len(np.shape(seg)) > 1:
                for b2 in range(len(seg)):
                    spacer[b2] = seg[b2][1] - seg[b2][0]
                seg_out_count[m, i] = np.nanmax(spacer)    
            else:
                seg_out_count[m, i] = seg[1] - seg[0]
        elif seg == 0:
            seg_out_count[m, i] = 0
            # should represent a count of tracks with nan segments of length less than seg_out_count[m, i]*grid_spacing km 
    viable[m] = sum(seg_out_count[m, :] < sf)            

print(viable)
# -- convert times to datetime elements
year = np.floor(adcp_time_mean/10000)
days = adcp_time_mean - np.floor(adcp_time_mean/1000)*1000
adcp_time_mean_dt = np.nan * np.ones(len(adcp_time_mean))
for i in range(len(adcp_time_mean)):
    int_dt = datetime.datetime(np.int(year[i]), 1, 1) + datetime.timedelta(np.int(days[i]) - 1)  # void info at sub day timescale 
    adcp_time_mean_dt[i] = int_dt.toordinal()

# for 'best depth', time of profiles (when do we achieve good crossings?)
viable_time = adcp_time_mean_dt[np.where(seg_out_count[lev[1], :] < sf)[0]]

# make make of data density (which depth and grid spaces have high/low numbers of nans)
nan_count = np.nan * np.ones((len(dep_levs), len(adcp_grid)))
for m in range(len(dep_levs)):
    for j in range(len(adcp_grid)):  # loop over each grid space 
        nan_count[m, j] = 100.0 * np.sum(np.isnan(adcp_u_grid[m, j, :]))/num_profs
        
# ---------------   
# -- export to nc
# ---------------
if save_nc > 0:
    OL_out = netCDF4.Dataset(file_out, 'w', format='NETCDF4_CLASSIC')
    # create dimenision
    dist_dim = OL_out.createDimension('dist_dim', len(adcp_grid))
    prof_num_dim = OL_out.createDimension('prof_dim', num_profs)
    z_dim = OL_out.createDimension('depth_dim', len(dep_levs))
    # assign variables
    dist_out = OL_out.createVariable('dist_grid', np.float64, ('dist_dim'))
    dist_out[:] = adcp_grid      
    lon_out = OL_out.createVariable('lon_grid', np.float64, ('dist_dim', 'prof_dim'))
    lon_out[:] = adcp_lon_grid
    lat_out = OL_out.createVariable('lat_grid', np.float64, ('dist_dim', 'prof_dim'))
    lat_out[:] = adcp_lat_grid
    depth_out = OL_out.createVariable('depths', np.float64, ('depth_dim'))
    depth_out[:] = dep_levs
    time_out = OL_out.createVariable('profile_year_day', np.float64, ('prof_dim'))
    time_out[:] = adcp_time_mean   
    nan_seg_out = OL_out.createVariable('max_nan_segment_length', np.float64, ('depth_dim', 'prof_dim'))
    nan_seg_out[:] = seg_out_count
    u_out = OL_out.createVariable('u', np.float64, ('depth_dim', 'dist_dim', 'prof_dim'))
    u_out[:] = adcp_u_grid
    v_out = OL_out.createVariable('v', np.float64, ('depth_dim', 'dist_dim', 'prof_dim'))
    v_out[:] = adcp_v_grid
    OL_out.close() 

# ---------------
# -- PLOTTING -- 
# ---------------

# -- plot plan view 
f, ax = plt.subplots()
for i in range(num_profs):
    # ax.scatter(adcp_lon[i], adcp_lat[i], s=5, color='r')
    ax.plot(adcp_lon[i], adcp_lat[i], linewidth=0.7, color='r', zorder=1)
samp_dist_lon = 354000/(1852*60*np.cos(np.deg2rad(lat_m))) + lon_s    
samp_dist_lat = lat_s - 354000/(1852*60)
ax.plot([lon_s, samp_dist_lon], [lat_s, samp_dist_lat], color='y', zorder=2)    
ax.text(samp_dist_lon + 0.25, samp_dist_lat, '500 km', fontsize=8, color='y')
baths = np.arange(-6000, 1000, 1000)       
cmapi = cmo.ice
ax.set_facecolor('#2E8B57')
cs = ax.contourf(bathy['longitude'].values + 360, bathy['latitude'].values, bathy['altitude'].values, cmap=cmapi, levels=baths)    
cb = plt.colorbar(cs)
cb.set_label('z [m]')
ax.set_xlabel('East Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Oleander Tracks, ' + this_year)
ax.set_xlim([280, 300])
ax.set_ylim([30, 44])
w = 1 / np.cos(np.deg2rad(35))
ax.set_aspect(w)
plot_pro(ax)
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_plan_view_tracks.jpg', dpi=300)
    
# -- plot plan view (sample crossing with velocities)
f, ax = plt.subplots()
i = 6
m = 10
baths = np.arange(-6000, 1000, 1000)       
cmapi = cmo.ice
ax.set_facecolor('#2E8B57')
cs = ax.contourf(bathy['longitude'].values + 360, bathy['latitude'].values, bathy['altitude'].values, cmap=cmapi, levels=baths)    
cb = plt.colorbar(cs)
ax.plot(adcp_lon[i], adcp_lat[i], linewidth=0.7, color='k', zorder=1)
ax.quiver(adcp_lon_grid[:, i], adcp_lat_grid[:, i], adcp_u_grid_cart[m, :, i], adcp_v_grid_cart[m, :, i], color='r', scale=10)
cb.set_label('z [m]')
ax.set_xlabel('East Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Single Oleander Track with Velocity at ' + str(dep_levs[m]) + 'm, ' + this_year)
ax.set_xlim([280, 300])
ax.set_ylim([30, 44])
w = 1 / np.cos(np.deg2rad(35))
ax.set_aspect(w)
plot_pro(ax)
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_plan_view_track_u_v.jpg', dpi=300)    

# -- plot vertical structure of acceptable passes 
f, ax = plt.subplots()
ax.plot(viable, dep_levs)
ax.set_title(this_year + ', passing profiles for ' + str(grid_spacing*sf) + 'km gaps')
ax.set_ylabel('Depth [m]')
ax.set_xlim([0, 150])
ax.invert_yaxis()
plot_pro(ax)
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_good_prof_count.jpg', dpi=300)

# -- plot timeline of good crossings    
f, ax = plt.subplots(1, 1, figsize = (6, 4))
ax.scatter(viable_time, np.zeros(len(viable_time)), s=10, color='k')
ax.set_xlabel('Year Day')
ax.set_title(this_year + ', timeline of acceptable crossings')
ax.set_yticks([])
# ax.set_xlim([20150000, 20170])
plot_pro(ax)
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_good_prof_timeline.jpg', dpi=300)

# -- plot nan density 
f, ax = plt.subplots()
aa = ax.pcolor(adcp_grid, dep_levs, nan_count, vmin=0, vmax=100)
nan_pass = nan_count.copy()
nan_pass[nan_pass > 75] = np.nan
cn = ax.contour(adcp_grid, dep_levs, nan_pass, levels=[10, 20, 30, 40, 50, 60, 70], colors='k', linewidths=0.5)
ax.clabel(cn, cn.levels, inline=True, fontsize=7)
ax.set_xlabel('Along-Transect Distance [km]')
ax.set_ylabel('Depth [m]')
ax.set_title('NaN Density (%)')
cb = plt.colorbar(aa)
cb.set_label('percent nan data')
ax.invert_yaxis()
plot_pro(ax)     
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_nan_density.jpg', dpi=300)
 
# -- plot cross section (three desired depths)      
# print(np.sum(good_p[lev, :], axis=1))
f, ax = plt.subplots(3, 1, sharex=True, figsize = (12, 7))
for m in range(3):
    for i in range(num_profs):
        if seg_out_count[lev[m], i] < sf:  #  np.sum(np.isnan(adcp_u_grid[lev[m], :, i])) < 50:
            # print('[' + str(m) + ', ' + str(i) + ']')
            ax[m].plot(dist_save[i], adcp_u[i][:, lev[m]], linewidth=0.8)
            ax[m].plot(adcp_grid, adcp_u_grid_cart[lev[m], :, i], color='k', linestyle='--', linewidth=0.7)
    ax[m].set_title('Zonal Velocity at ' + str(dep_levs[lev[m]]) + 'm (' + str(np.int(viable[lev[m]])) + ' out of ' + str(num_profs) + ' tracks)')  
    ax[m].set_ylabel('Velocity [m/s]')    
    ax[m].set_ylim([-2, 3])
    ax[m].grid()
ax[m].grid()    
ax[m].set_xlabel('Along-Transect Distance [km]')
plot_pro(ax[m])
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_velocity_3_depths.jpg', dpi=300)
    
# print(np.sum(good_p[lev, :], axis=1))
f, ax = plt.subplots(3, 1, sharex=True, figsize = (12, 7))
for m in range(3):
    for i in range(num_profs):
        if seg_out_count[lev[m], i] < sf:  #  np.sum(np.isnan(adcp_u_grid[lev[m], :, i])) < 50:
            # print('[' + str(m) + ', ' + str(i) + ']')
            ax[m].plot(dist_save[i], adcp_v[i][:, lev[m]], linewidth=0.8)
            ax[m].plot(adcp_grid, adcp_v_grid_cart[lev[m], :, i], color='k', linestyle='--', linewidth=0.7)
    ax[m].set_title('Meridional Velocity at ' + str(dep_levs[lev[m]]) + 'm (' + str(np.int(viable[lev[m]])) + ' out of ' + str(num_profs) + ' tracks)')  
    ax[m].set_ylabel('Velocity [m/s]')    
    ax[m].set_ylim([-2, 3])
    ax[m].grid()
ax[m].grid()    
ax[m].set_xlabel('Along-Transect Distance [km]')
plot_pro(ax[m])
if savee > 0:
    f.savefig('/Users/jakesteinberg/Documents/CPT/oleander/plots/' + this_year + '_v_velocity_3_depths.jpg', dpi=300)    

    
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# crude method for distance estimation 
    # x = 1852 * 60 * np.cos(np.deg2rad(lat_m)) * (adcp_lon[i] - lon_s)
    # y = 1852 * 60 * (adcp_lat[i] - lat_s)
    # dist = np.transpose(np.sqrt(x**2 + y**2) / 1000)[0]
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

# # -- thermosalinograph
# tsg_lon = x2['oleander_tsg']['lon'][0] + 360
# tsg_lat = x2['oleander_tsg']['lat'][0]
# tsg_time = x2['oleander_tsg']['time'][0] # (6 Jan 2001 -- 26 Feb 2019), determined from matlab...python needs offset
# tsg_salt = x2['oleander_tsg']['salt'][0]
# tsg_temp = x2['oleander_tsg']['temp'][0]
# rel_ind = []
# for i in range(len(tsg_time)):
#     t_s_i = datetime.datetime.fromordinal(np.int(tsg_time[i][0][0]) - 365)
#     if (np.int(t_s_i.year) == this_year):
#         rel_ind.append(i)
#         # print(datetime.datetime.fromordinal(np.int(tsg_time[i][0][0])))
#
# this_cruise = 2
# this_tsg_time = np.argsort(tsg_time[rel_ind[this_cruise]])
# # print length in time of this 'cruise'
# print(datetime.datetime.fromordinal(np.int(tsg_time[rel_ind[this_cruise]][0][0]) - 365))
# print(datetime.datetime.fromordinal(np.int(tsg_time[rel_ind[this_cruise]][0][-1]) - 365))
# tsg_d1 = datetime.datetime.fromordinal(np.int(tsg_time[rel_ind[this_cruise]][0][0]) - 365)
# tsg_d2 = datetime.datetime.fromordinal(np.int(tsg_time[rel_ind[this_cruise]][0][-1]) - 365)
# if tsg_d1 > tsg_d2:
#     tsg_min = tsg_d2
#     tsg_max = tsg_d1
# else:
#     tsg_min = tsg_d1
#     tsg_max = tsg_d2
# adcp_in = np.where((adcp_days >= (np.int(tsg_min.strftime('%j'))-2)) & (adcp_days <= (np.int(tsg_max.strftime('%j'))+0)))[0]
#
# # f, ax1 = plt.subplots()
# # ax1.scatter(adcp_lon[adcp_in], adcp_lat[adcp_in], s=5, color='b')
# # ax1.scatter(tsg_lon[rel_ind[this_cruise]], tsg_lat[rel_ind[this_cruise]], s=0.5, color='r')
# # ax1.set_xlim([285, 296])
# # plot_pro(ax1)
#
# # tsg T/S variabilty along a single crossing
# ref_lat = 35
# tsg_x = (tsg_lon[rel_ind[this_cruise]][0] - np.nanmin(tsg_lon[rel_ind[this_cruise]][0])) * 1852 * 60 * np.cos(np.deg2rad(ref_lat))
# tsg_y = (tsg_lat[rel_ind[this_cruise]][0] - np.nanmax(tsg_lat[rel_ind[this_cruise]][0])) * 1852 * 60
# tsg_dist = np.sqrt(tsg_x**2 + tsg_y**2)
