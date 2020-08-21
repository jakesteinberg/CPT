import scipy.ndimage as si
from scipy import interpolate
from scipy import integrate
import numpy as np
from scipy.fftpack import fft
from tqdm.notebook import tqdm
# from tqdm import tqdm
from math import radians, degrees, sin, cos, asin, acos, sqrt


# --- output in kilometers --- 
def Haversine(lat1,lon1,lat2,lon2, **kwarg):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, 
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points 
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d,4)
    
    
# --- interpolate --- 
def interpolate_nans(data, grid, cutoff):  # SLA 
    num_tracks = np.shape(data)[0]
    # -- search for good data
    # look at each track at each depth and inventory the number of nans, but also the number of nan segments and the segment lengths 
    seg_out = {}
    seg_out_count = {}
    good_indices_0 = []
    good_indices = np.zeros(num_tracks)
    # print('interpolating by track')
    for i in range(num_tracks):  # loop over tracks
        this_track = data[i, :]  # CALL ACTUAL DATA HERE 
        bad = np.where(np.isnan(this_track))[0]  # nan indices 
        seg = []
        if ((len(bad) >= 2) & (len(bad) < len(this_track))):
            breaky = np.where(np.diff(bad) > 1)[0] + 1  # look for breaks in list of nans 
            if len(breaky) > 0:      
                seg.append([bad[0], bad[breaky[0] - 1]])
                for b in range(len(breaky) - 1):
                    seg.append([bad[breaky[b]], bad[breaky[b + 1] - 1]])
                # last index 
                if bad[breaky[-1]] == bad[-1]:
                    seg.append([bad[breaky[-1]], bad[breaky[-1]]])    
                else:
                    seg.append([bad[breaky[-1]], bad[breaky[-1]]]) 
            elif (len(breaky) == 0) & (bad[0] > 0): 
                seg = [bad[0], bad[-1]]    
            elif (len(breaky) == 0) & (bad[0] == 0):
                seg = [bad[0], bad[-1]]   
    
        elif len(bad) == 1:
            seg = [bad[0], bad[0]]
        elif len(bad) == len(this_track):
            seg = len(data[0, :])  # all are nan's 
        else:
            seg = 0  # none are nans     
        seg_out[i] = seg    
        # this is a dictionary with coordinates (1, profile) identifying nan segments and their length 
                
        # inspect seg_out to see which are good and which might meet some defined criteria 
        if (seg != 0) & (seg != len(data[0, :])):
            spacer = np.nan * np.ones(len(seg))
            if len(np.shape(seg)) > 1:
                for b2 in range(len(seg)):
                    spacer[b2] = seg[b2][1] - seg[b2][0]
                seg_out_count[i] = spacer  # np.nanmax(spacer)  
            else:
                seg_out_count[i] = seg[1] - seg[0]
        elif seg == 0:
            seg_out_count[i] = np.nan          
        
    # interpolate
    interpolated_signal = data.copy()
    for i in range(num_tracks): # interpolate only transects that meet nan_seg criteria 
        this_u = data[i, :]
        interp_sig = interpolated_signal[i, :].copy()
        these_segments = seg_out[i]
        if these_segments == 0:
            # there are no nans 
            continue 
        if np.sum(np.isnan(this_u)) == len(this_u):
            continue
        else:
            seggy = len(these_segments)
            for j in range(seggy):
                if len(np.isfinite(np.shape(seg_out_count[i]))) == 1:
                    if these_segments[j][0] == (len(this_u) - 1):
                        continue
                    if seg_out_count[i][j] <= cutoff:
                        this_seg_s = these_segments[j][0] - 1
                        this_seg_e = these_segments[j][-1] + 1
                        interp_sig[this_seg_s:this_seg_e+1] = np.interp(grid[this_seg_s:this_seg_e+1], [grid[this_seg_s], grid[this_seg_e]], [this_u[this_seg_s], this_u[this_seg_e]])
                elif len(seg_out_count) == 1:
                    interp_sig[these_segments[0]:these_segments[-1]+1] = np.interp(grid[these_segments[0]:these_segments[-1]+1], [grid[these_segments[0]], grid[these_segments[-1]]], [this_u[these_segments[0]], this_u[these_segments[-1]]])
            
        interpolated_signal[i, :] = interp_sig
            
    return interpolated_signal    

# interpolate and smooth [MxN] array of sla (I apply this code in my parsing function directly such that I can list the output for binning process later on)    
def smooth_grid_tracks(cutoff, dist, sla, track_record, lon_record, lat_record, sigma):
    # for each track 
    sla_s = []
    for i in tqdm(range(len(track_record))):
        this_dist = dist[i]
        this_sla = sla[i]
        # interp 
        this_interp_sla = interpolate_nans(this_sla, this_dist, cutoff)
        # smooth for each pass 
        smoothed_sla_gauss = np.nan * np.ones(np.shape(this_sla))
        for j in range(np.shape(this_sla)[0]):     
            smoothed_sla_gauss[j, :] = si.gaussian_filter(this_interp_sla[j, :], sigma, order=0)
        sla_s.append(smoothed_sla_gauss)
              
    return(this_interp_sla, sla_s)


# parse_grid_tracks uses track and cycle information to separate, organize, and interpolate nans --> prep for smoothing 
def parse_grid_tracks(tracks, df2_s, d_grid_step, interp_cutoff):
    dist = []
    adt = []
    sla = []
    sla_int = []
    lon_record = []
    lat_record = []
    time_record = []
    track_record = []
    count = 0 
    # -- loop over each track (listed by number )
    for m in tqdm(range(len(tracks))):
        # -- subset the dataframe to deal with only one track at a time
        sat_track_i = df2_s[df2_s['track'] == tracks[m]]    
        these_cycles = np.unique(sat_track_i['cycle'])  # extract numbered list of each cycle 
    
        # -- compute distance along arc just once (first index) to determine the length of d_grid
        # - first cycle (index = 0)
        this_cycle = sat_track_i[sat_track_i['cycle'] == these_cycles[0]]
        
        # -- if the length of the track is too short skip track 
        if len(this_cycle) < 3:
            continue 
        
        # -- take first cycle and compute distance along track 
        this_dist = Haversine(this_cycle['latitude'][0], this_cycle['longitude'][0], \
                              this_cycle['latitude'], this_cycle['longitude']) 
        lon_start = this_cycle['longitude'][0]
        lat_start = this_cycle['latitude'][0]
        d_grid = np.arange(0, np.nanmax(this_dist), d_grid_step)
        lon_grid = np.interp(d_grid, this_dist, this_cycle['longitude'].values)
        lat_grid = np.interp(d_grid, this_dist, this_cycle['latitude'].values)  
        this_time = this_cycle['sla_filtered'].index
        this_time_c = (this_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
        # -- prep for looping over each cycle of this_track
        # - initialize arrays 
        this_mdt_grid = np.nan * np.ones((len(these_cycles), len(d_grid)))
        this_sla_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_lon_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_lat_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_time_grid = np.nan * np.ones(len(these_cycles)) 
        this_time_grid[0] = np.nanmean(this_time_c)
        # -- loop over each cycle 
        for c in range(len(these_cycles)):
            this_cycle = sat_track_i[sat_track_i['cycle'] == these_cycles[c]]
            # this_dist = Haversine(this_cycle['latitude'][0], this_cycle['longitude'][0], \
            #                       this_cycle['latitude'], this_cycle['longitude'])
            this_dist = Haversine(lat_start, lon_start, this_cycle['latitude'], this_cycle['longitude'])
            
            sla_grid_pass1  = np.interp(d_grid, this_dist, np.array(this_cycle['sla_filtered']))
        
            # deal with land and interpolatings 
            land = np.where(np.diff(np.array(this_dist)) > 10)[0]  # these are indices idenify gaps in the data 
            # interpolate across grid, but retain info as to which distances are covered, fill with nans
            this_dist2 = np.array(this_dist)
            for l in range(len(land)):
                land_i = np.where((d_grid >= this_dist2[land[l]]) & (d_grid <= this_dist2[land[l]+1]))[0]
                sla_grid_pass1[land_i] = np.nan
            # remove interpolated stretches across land that are repeats of the same data 
            sla_grid_pass1[np.where(np.abs(np.diff(sla_grid_pass1)) < 0.0001)[0]] = np.nan 
            this_sla_grid[c, :] = sla_grid_pass1 
            this_mdt_grid[c, :] = np.interp(d_grid, this_dist, np.array(this_cycle['mdt']))
            this_time = this_cycle['sla_filtered'].index
            this_time_c = (this_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            this_time_grid[c] = np.nanmean(this_time_c)
        
        # -- interpolate 
        this_interp_sla = interpolate_nans(this_sla_grid, d_grid, interp_cutoff)
        this_interp_mdt = interpolate_nans(this_mdt_grid, d_grid, interp_cutoff)
        # -- add mdt + sla to obtain adt 
        this_adt = this_interp_sla + this_interp_mdt 
        
        # identify and nan out land (or data gaps)
        lon_grid[np.isnan(np.nanmean(this_interp_sla, axis=0))] = np.nan  # this_sla_grid
        lat_grid[np.isnan(np.nanmean(this_interp_sla, axis=0))] = np.nan 
        
        # save for each track arrays of fields [cycle X Distance] (each array is an element in a list)
        adt.append(this_adt)
        sla.append(this_sla_grid)
        sla_int.append(this_interp_sla)
        dist.append(d_grid) 
        lon_record.append(lon_grid)
        lat_record.append(lat_grid)
        time_record.append(this_time_grid)
        track_record.append(tracks[m])
         
        # save all as list to index over when binning for maps 
        if count < 1:
            # -- to run if I want sla, mdt, adt by time increment 
            # time_t = np.tile(this_time_grid, (1, len(these_cycles)))
            # lon_t = np.tile(lon_grid, (1, len(these_cycles)))
            # lat_t = np.tile(lat_grid, (1, len(these_cycles)))  
            # sla_t = this_sla_grid.flatten()        
            lon_t = lon_grid.copy()
            lat_t = lat_grid.copy()
            track_t = tracks[m] * np.ones(len(d_grid))
            # -- mean sla variance (average over all cycles for each track) (assumes zero mean)
            # sla_t = np.nanmean(this_sla_grid**2, axis=0)  
            # -- mean adt 
            # adt_t = np.nanmean(this_adt, axis=0)  # adt
        else:
            # -- to run if I want sla, mdt, adt by time increment
            # time_t = np.append(time_t, np.tile(this_time_grid, (1, len(these_cycles))))
            # lon_t = np.append(lon_t, np.tile(lon_grid, (1, len(these_cycles))))
            # lat_t = np.append(lat_t, np.tile(lat_grid, (1, len(these_cycles))))
            # sla_t = np.append(sla_t, this_sla_grid.flatten())
            lon_t = np.append(lon_t, lon_grid.copy())
            lat_t = np.append(lat_t, lat_grid.copy())
            track_t = np.append(track_t, tracks[m] * np.ones(len(d_grid)))            
            # sla_t = np.append(sla_t, np.nanmean(this_sla_grid**2, axis=0))
            # adt_t = np.append(adt_t, np.nanmean(this_adt, axis=0))             
        count = count + 1

    return lon_t, lat_t, track_t, adt, sla, sla_int, dist, lon_record, lat_record, time_record, track_record


def coarsen(dist, lon_record, lat_record, coarsening_factor, sig_in):  
    coarse_sig_out = []
    coarse_grid_out = []
    coarse_lon_out = []
    coarse_lat_out = []
    for mm in tqdm(range(len(dist))):  # loop over each track 
        this_dist = dist[mm]
        this_lon = lon_record[mm]
        this_lat = lat_record[mm]
        smooth_sig = sig_in[mm]
        
        coarse_grid = np.arange(this_dist[0], this_dist[-1], (this_dist[1] - this_dist[0]) * coarsening_factor)
        coarse_i = np.nan * np.ones((np.shape(smooth_sig)[0], len(coarse_grid) - 1))
        coarse_lon = np.nan * np.ones(len(coarse_grid) - 1)
        coarse_lat = np.nan * np.ones(len(coarse_grid) - 1)
        
        if len(coarse_grid) > 3:
            coarse_grid_c = coarse_grid[0:-1] + (coarse_grid[1] - coarse_grid[0])/2  # bin center  
        else:
            coarse_grid_c = coarse_grid[0:-1]  

        if len(coarse_grid) > coarsening_factor:           
            # average all points in bins with width equal to coarser grid 
            for j in range(1, len(coarse_grid)):
                coarse_i[:, j - 1] = np.nanmean(smooth_sig[:, (this_dist > coarse_grid[j - 1]) & (this_dist < coarse_grid[j])], axis=1)  
                coarse_lon[j - 1] = np.nanmean(this_lon[(this_dist > coarse_grid[j - 1]) & (this_dist < coarse_grid[j])])
                coarse_lat[j - 1] = np.nanmean(this_lat[(this_dist > coarse_grid[j - 1]) & (this_dist < coarse_grid[j])])
        
        coarse_grid_out.append(coarse_grid_c)
        coarse_lon_out.append(coarse_lon)
        coarse_lat_out.append(coarse_lat)
        coarse_sig_out.append(coarse_i)
                    
    return coarse_grid_out, coarse_lon_out, coarse_lat_out, coarse_sig_out


def specsharp(grid_spacing, coarse_fac, nyquist_wavenumber):     
    # -- get filter weights for a given filter width, n grid cells
    def getWeights(n):  
        w = np.zeros(n+1) # initialize weights
        for i in range(n):
            integrand = lambda k: 2*(F(k)-1)*(np.cos((i+1)*k)-1)
            w[i] = integrate.quad(integrand,0,np.pi)[0]
        A = 2*np.pi*(np.eye(n) + 2)
        w[1:] = np.linalg.solve(A,w[0:n])
        w[0] = 1 - 2*np.sum(w[1:])
        return w
    
    x = coarse_fac                  # coarsening factor (actual 'width' is a function of grid spacing)
    # nyquist_wavenumber = smallest resolvable scale on the 'new' grid 
    F = interpolate.PchipInterpolator(np.array([0, 1/x, nyquist_wavenumber/x, nyquist_wavenumber]), np.array([1, 1, 0, 0]))
    print('Filter Half-Width = ')    
            
    weight_prev = getWeights(2)
    # loop over filter widths until weights converge
    for j in range(3, 50):  # 40 might not be enough? 
        this_weight = getWeights(j)
        # difference between these weights and last iterations (looking for convergence)
        wd = np.sum(np.abs(this_weight[0:3] - weight_prev[0:3]) / np.abs(weight_prev[0:3]))  
        if wd < 0.001:  # convergence threshold 
            jj = j
            print(str(j - 1) + ' ' + str(getWeights(j - 1)[0:4]))
            print(str(j) + ' ' + str(getWeights(j)[0:4]))
            print('converged //')
            break
        weight_prev = this_weight   
            
    filter_kernel = np.concatenate((np.flip(getWeights(jj))[0:-1], getWeights(jj)))
    print('------------------------------------------------------------')
    print('for a coarsening factor of ' + str(coarse_fac) + ', recommend:')
    print('-- filter width of ' + str(2*jj + 1) + ' grid cells (here = ' + str(grid_spacing * (2*jj + 1)) + ' km)')
    print('------------------------------------------------------------')
    return filter_kernel, jj


def smooth_tracks(dist, adt, sla, lon_record, lat_record, time_record, track_record, coarsening_factor, filter_choice, nyquist_wavelength):
    hor_grid_spacing = dist[0][1] - dist[0][0]
    
    if filter_choice == 'sharp':
        # compute filter kernel and weights 
        filter_kernel, jj = specsharp(hor_grid_spacing, coarsening_factor, nyquist_wavelength)
    if filter_choice == 'gaussian':
        sigma = coarsening_factor 
        
    sla_smooth = []
    adt_smooth = []
    count = 0
    for m in tqdm(range(len(track_record))):
        # -- load in data for this track 
        this_adt = adt[m]  # interpolated field
        this_sla = sla[m]  # interpolated field   
        lon_grid = lon_record[m]
        lat_grid = lat_record[m]
        d_grid = dist[m]
        these_cycles = np.arange(0, np.shape(this_sla)[0])
                  
        if len(d_grid) < 10:
            print('track ' + str(m) + ', too short') 
            sla_smooth.append(np.nan * np.ones(np.shape(this_sla)))
            adt_smooth.append(np.nan * np.ones(np.shape(this_sla)))
            continue
        
        # -- smooth HERE for each cycle 
        if filter_choice == 'gaussian':
            smoothed_sla = np.nan * np.ones(np.shape(this_sla))
            smoothed_adt = np.nan * np.ones(np.shape(this_sla))
            for j in range(np.shape(this_sla)[0]):     
                smoothed_sla[j, :] = si.gaussian_filter(this_sla[j, :], sigma, order=0)
                smoothed_adt[j, :] = si.gaussian_filter(this_adt[j, :], sigma, order=0)
            
        if filter_choice == 'sharp':    
            smoothed_sla = sharp_smooth(filter_kernel, this_sla)
            smoothed_adt = sharp_smooth(filter_kernel, this_adt)
    
        sla_smooth.append(smoothed_sla)
        adt_smooth.append(smoothed_adt)
    
    return sla_smooth, adt_smooth


def sharp_smooth(filter_kernel, signal0):            
    n = np.int((len(filter_kernel) - 1)/2) # filter half-width
    filter_width = len(filter_kernel)
    # -- loop over transect or pass
    smooth_sig = np.nan*np.ones(np.shape(signal0))
    for p in range(np.shape(signal0)[0]):
        this_sig = signal0[p, :].copy()
        for j in range(len(this_sig)):  # loop over each grid point and smooth.
            # if at left or right edge (filter is only partial) skip 
            if j < n:  # edge0
                # sig_partial = np.concatenate((np.zeros(n - j), this_sig[0:(j + n + 1)]))
                # if np.sum(np.isnan(sig_partial)) < 1:
                #     smooth_sig[p, j] = np.nansum(filter_kernel * sig_partial)
                continue    
            elif j >= (len(this_sig) - n):  # edge1
                # sig_partial = np.concatenate((this_sig[(j - n):], np.zeros(filter_width - len(this_sig[(j - n):]))))
                # if np.sum(np.isnan(sig_partial)) < 1:
                #     smooth_sig[p, j] = np.nansum(filter_kernel * sig_partial)
                continue
            else:
                if np.sum(np.isnan(this_sig[(j - n):(j + n + 1)])) < 1:  # check that there are no nans in this filter use at j 
                    smooth_sig[p, j] = np.nansum(filter_kernel * this_sig[(j - n):(j + n + 1)])
        smooth_sig[p, np.isnan(this_sig)] = np.nan
    return smooth_sig


# horizontal wavenumber spectra 
def spectra_slopes(track_record, dist, sla_int, k, L, dx, meso, spec_win, spec_win_ind, taper, taper_len, single, spec_inc):
    meso_ind = np.where((1/(k) >= meso[0]) & (1/(k) <= meso[-1]))[0]
    
    if taper:
        # edge taper (if desired) (grid points to taper over at each end)
        tape_len0 = taper_len  # length on each end 
        tape_len = tape_len0*2 - 1
        nn = np.arange(0, tape_len + 1) - tape_len/2
        taper_i = np.exp(-(1/2)*(2.5*nn/(tape_len/2))**2)
    
    if single:
        this_x = dist.copy()
        this_sla = sla_int.copy()
        # check that track distance is longer than 2 * spec_win
        if np.nanmax(this_x) > (spec_win * 2.5):
            x_mod = np.arange(np.where(this_x==spec_win)[0][0], len(this_x) - np.where(this_x==spec_win)[0][0] - 1)
            # -- determine array sizes for fft output (only once)      
            # grid_len = this_x[(x_mod[0] - spec_win_ind):(x_mod[0] + spec_win_ind + 1)]
            # L = np.int(len(grid_len)) - 1
            # k = np.arange(0, L/2, 1)/L/dx*2*np.pi
            # -- define fft_out and loop over each window of length spec_win and compute spectra 
            fft_out = np.nan * np.ones((np.shape(this_sla)[0], np.shape(this_sla)[1], len(k))) 
            meso_slope = np.nan * np.ones((np.shape(this_sla)[0], np.shape(this_sla)[1], 2))
            # -- loop over each cycle in track mm 
            for i in range(np.shape(this_sla)[0]):
                this_si = this_sla[i, :].copy()
                # loop over each x increment to compute spectra centered at grid point xx
                for xx in range(0, len(x_mod), spec_inc):  
                    this_grid = this_x[(x_mod[xx] - spec_win_ind):(x_mod[xx] + spec_win_ind + 1)].copy()
                    this_sig = this_si[(x_mod[xx] - spec_win_ind):(x_mod[xx] + spec_win_ind + 1)].copy()
                    if np.sum(np.isnan(this_sig)) < 1:
                        this_sig_anom = this_sig - np.nanmean(this_sig)
                        if taper:
                            this_sig_anom[0:tape_len0] = this_sig_anom[0:tape_len0] * taper_i[0:tape_len0]
                            this_sig_anom[-tape_len0:] = this_sig_anom[-tape_len0:] * taper_i[-tape_len0:]
                        
                        # -- take fft (have to multiply by grid spacing for proper variance to be calculated)
                        this_fft = fft(this_sig_anom, L) * dx
                        # fft_out[i, x_mod[xx], :] = this_fft[1:(np.int(np.floor(L/2)) + 2)] * np.conj(this_fft[1:(np.int(np.floor(L/2)) + 2)]) * dx * 2 * L
                        fft_out[i, x_mod[xx], :] = 2 * (k[1] - k[0]) * np.abs(this_fft[0:(np.int(np.floor(L/2)))])**2
                        # -- estimate slope over mesoscale wavenumber band
                        meso_p = np.polyfit(np.log10(k[meso_ind]), np.log10(fft_out[i, x_mod[xx], meso_ind]), 1)
                        meso_slope[i, x_mod[xx], :] = meso_p    
        else:
            print('track is too short')
            
        return(meso_slope, fft_out)     
    
    else:
        meso_slope_out = []
        meso_spectra_out = []
        for mm in tqdm(range(len(sla_int))):
            this_x = dist[mm]
            this_sla = sla_int[mm].copy()
            # check that track distance is longer than 2 * spec_win
            if np.nanmax(this_x) > (spec_win * 2.5):
                # x_mod = np.arange(np.where(this_x==spec_win)[0][0], len(this_x) - np.where(this_x==spec_win)[0][0] - 1)
                x_mod = np.arange(np.where(this_x>=spec_win)[0][0], len(this_x) - np.where(this_x>=spec_win)[0][0] - 1)
                # -- determine array sizes for fft output (only once)      
                # grid_len = this_x[(x_mod[0] - spec_win_ind):(x_mod[0] + spec_win_ind + 1)]
                # L = np.int(len(grid_len)) - 1
                # k = np.arange(0, L/2, 1)/L/dx*2*np.pi
                # -- define fft_out and loop over each window of length spec_win and compute spectra 
                fft_out = np.nan * np.ones((np.shape(this_sla)[0], np.shape(this_sla)[1], len(k))) 
                meso_slope = np.nan * np.ones((np.shape(this_sla)[0], np.shape(this_sla)[1], 2))
                # -- loop over each cycle in track mm 
                for i in range(np.shape(this_sla)[0]):
                    this_si = this_sla[i, :].copy()
                    # loop over each x increment to compute spectra centered at grid point xx
                    for xx in range(0, len(x_mod), spec_inc):  
                        this_grid = this_x[(x_mod[xx] - spec_win_ind):(x_mod[xx] + spec_win_ind + 1)].copy()
                        this_sig = this_si[(x_mod[xx] - spec_win_ind):(x_mod[xx] + spec_win_ind + 1)].copy()
                        if np.sum(np.isnan(this_sig)) < 1:
                            this_sig_anom = this_sig - np.nanmean(this_sig)
                            if taper:
                                this_sig_anom[0:tape_len0] = this_sig_anom[0:tape_len0] * taper_i[0:tape_len0]
                                this_sig_anom[-tape_len0:] = this_sig_anom[-tape_len0:] * taper_i[-tape_len0:]
                            # -- take fft     
                            this_fft = fft(this_sig_anom, L) * dx
                            # fft_out[i, x_mod[xx], :] = this_fft[0:np.int(np.floor(L/2))] * np.conj(this_fft[0:np.int(np.floor(L/2))]) * dx * 2
                            fft_out[i, x_mod[xx], :] = 2 * (k[1] - k[0]) * np.abs(this_fft[0:(np.int(np.floor(L/2)))])**2
                            # -- estimate slope over mesoscale wavenumber band
                            meso_p = np.polyfit(np.log10(k[meso_ind]), np.log10(fft_out[i, x_mod[xx], meso_ind]), 1)
                            meso_slope[i, x_mod[xx], :] = meso_p
                meso_slope_out.append(meso_slope)
                meso_spectra_out.append(fft_out)
            else:
                print('track ' + str(track_record[mm]) + ' too short to compute wavenumber spectra within mesoscale range')
                meso_slope_out.append(np.nan * np.ones(np.shape(this_sla)))
                meso_spectra_out.append(np.nan * np.ones((np.shape(this_sla)[0], np.shape(this_sla)[1], len(k))))
    
        return(meso_slope_out, meso_spectra_out)
    
    
# cross track velocity   
def velocity(adt, sla, adt_smooth, sla_smooth, lon_record, lat_record, time_record, track_record):
    
    vel = []
    vel_tot = []
    vel_tot_smooth = []
    tot_grad = []
    count = 0
    for m in tqdm(range(len(track_record))):
        # -- load in data for this track 
        this_adt = adt[m]  # interpolated field
        this_sla = sla[m]  # interpolated field   
        smoothed_adt = adt_smooth[m]  # interpolated field
        smoothed_sla = sla_smooth[m]  # interpolated field   
        lon_grid = lon_record[m]
        lat_grid = lat_record[m]
        d_grid = dist[m]
        these_cycles = np.arange(0, np.shape(this_sla)[0])
                  
        if len(d_grid) < 10:
            print('track ' + str(m) + ', too short') 
            tot_grad.append(np.nan * np.ones(np.shape(this_sla)))
            vel.append(np.nan * np.ones(np.shape(this_sla)))
            vel_tot.append(np.nan * np.ones(np.shape(this_sla)))
            vel_tot_smooth.append(np.nan * np.ones(np.shape(this_sla)))
            continue
            
        # -- gradient (of interpolated field)
        # (pol_rad = 6378.137km) (eq_rad = 6356.752km) 
        f_loc = 2*(7.27*10**(-5))*np.sin(np.deg2rad(lat_grid))    
        
        # estimate gradient from Arbic 2012 
        sla_grad = np.gradient(this_sla, d_grid*1000.0, axis=1)
        adt_grad = np.gradient(this_adt, d_grid*1000.0, axis=1)
        adt_smooth_grad = np.gradient(smoothed_adt, d_grid*1000.0, axis=1)
        for cdm in range(4, 4 + len(sla_grad[0, 4:-3])):
            # -- gradients from a 7 point stencil 
            sla_grad[:, cdm] = (this_sla[:, cdm+3] - 9*this_sla[:, cdm+2] + 45*this_sla[:, cdm+1] \
                                - 45*this_sla[:, cdm-1] + 9*this_sla[:, cdm-2] - this_sla[:, cdm-3]) / (60*(hor_grid_spacing*1000.0))
            adt_grad[:, cdm] = (this_adt[:, cdm+3] - 9*this_adt[:, cdm+2] + 45*this_adt[:, cdm+1] \
                                - 45*this_adt[:, cdm-1] + 9*this_adt[:, cdm-2] - this_adt[:, cdm-3]) / (60*(hor_grid_spacing*1000.0))
            # smoothed gradient 
            adt_smooth_grad[:, cdm] = (smoothed_adt[:, cdm+3] - 9*smoothed_adt[:, cdm+2] + 45*smoothed_adt[:, cdm+1] \
                                - 45*smoothed_adt[:, cdm-1] + 9*smoothed_adt[:, cdm-2] - smoothed_adt[:, cdm-3]) / (60*(hor_grid_spacing*1000.0))    
            
            # -- gradients from a 5 point stencil 
            # adt_grad[:, cdm] = (-this_adt[:, cdm+2] + 8*this_adt[:, cdm+1] - 8*this_adt[:, cdm-1] + this_adt[:, cdm-2]) / (12*(hor_grid_spacing*1000.0))
            # adt_smooth_grad[:, cdm] = (-smoothed_adt_gauss[:, cdm+2] + 8*smoothed_adt_gauss[:, cdm+1] \
            #        - 8*smoothed_adt_gauss[:, cdm-1] + smoothed_adt_gauss[:, cdm-2]) / (12*(hor_grid_spacing*1000.0))

        # compute velocity via geostrophic balance 
        this_vel = (9.81/np.tile(f_loc[None, :], (len(these_cycles), 1))) * sla_grad  # np.gradient(this_interp_sla, d_grid*1000.0, axis=1)
        this_vel_tot = (9.81/np.tile(f_loc[None, :], (len(these_cycles), 1))) * adt_grad  # np.gradient(this_adt, d_grid*1000.0, axis=1)
        this_vel_tot_s = (9.81/f_loc) * adt_smooth_grad  # np.gradient(smoothed_adt_gauss, d_grid*1000.0, axis=1)
        
        # near equator attempt beta plane correction from Lagerloef 1999 
        close_eq1 = np.where(np.abs(lat_grid) < 2.5)[0]
        if len(close_eq1) > 4:  # if there are points close to equator, make sure there are enough to compute a gradient
            beta = 2*(7.27*10**(-5))*np.cos(np.deg2rad(lat_grid[close_eq1]))/(6356752)
            y = 1852 * 60 * (lat_grid[close_eq1] - 0)  # 6356752*lat_grid[close_eq1]  
            # weights transitioning from beta plane to f plane 
            wb = np.exp(-(np.abs(lat_grid[close_eq1])/2.2)**2)
            wf = 1 - wb           
            L = 111000
            theta = y/L
            
            # uf = (9.81/np.tile(f_loc[close_eq1][None, :], (len(these_cycles), 1))) * adt_grad[:, close_eq1]
            ub = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1)))) * np.gradient(adt_grad[:, close_eq1], y, axis=1)
            uf_smooth = (9.81/np.tile(f_loc[close_eq1][None, :], (len(these_cycles), 1))) * adt_smooth_grad[:, close_eq1]
            ub_smooth = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1)))) * np.gradient(adt_smooth_grad[:, close_eq1], y, axis=1)
            
            uf = (9.81/(np.tile(f_loc[close_eq1][None, :], (len(these_cycles), 1)))) * adt_grad[:, close_eq1]   # np.tile(1/theta[None, :], (len(these_cycles), 1))
            # ub1 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*y)) * adt_grad[:, close_eq1] # * np.tile(theta[None, :], (len(these_cycles), 1))
            # ub2 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*L)) * \
            #     adt_grad[:, close_eq1] * np.tile(theta[None, :]**2, (len(these_cycles), 1))
            # ub3 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*L)) * \
            #     adt_grad[:, close_eq1] * np.tile(theta[None, :]**3, (len(these_cycles), 1))
            # ub = ub1  # ub1 + ub2 + ub3
            ug = np.tile(wb[None, :], (len(these_cycles), 1))*ub + np.tile(wf[None, :], (len(these_cycles), 1))*uf
            this_vel_tot[:, close_eq1] = ug 
            # print(lat_grid[close_eq1])
            # print(y)
            # print(uf_smooth[0, :])
            # print(ub_smooth[0, :])
            ug_smooth = np.tile(wb[None, :], (len(these_cycles), 1))*ub_smooth + np.tile(wf[None, :], (len(these_cycles), 1))*uf_smooth
            this_vel_tot_s[:, close_eq1] = ug_smooth
        
        # -- save for each track arrays of fields [cycle X Distance] (each array is an element in a list)
        tot_grad.append(adt_grad)
        vel.append(this_vel)
        vel_tot.append(this_vel_tot)
        vel_tot_smooth.append(this_vel_tot_s)
               
    return tot_grad, vel, vel_tot, vel_tot_smooth