import scipy.ndimage as si
from scipy import interpolate
from scipy import integrate
import numpy as np
from scipy.fftpack import fft
from tqdm.notebook import tqdm
from math import radians, degrees, sin, cos, asin, acos, sqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# ***This library includes*** 
# - Haversine              (great circle distance between points)
# - nan_helper             (ID nan indices in 1d array)
# - interp_nans            (interpolate nans ID'd in nan_helper with cutoff criteria)
# - parse_grid_tracks      *(takes dataset and parses measurements into 2d arrays for each track [cycle X distance])* 
# - filterSpec             generalized laplacian/biharmonic filter (Taper or Gaussian)
# - Laplacian1D            computes laplacian for fixed grid step 
# - Filter                 calls filterSpec and Laplacian1D to generate/apply desried filter  (sharp, gaussian, boxcar)
# - smooth_tracks_deg      filters to local delta longitude (in units of degree, i.e. 1/4)
# - smooth_tracks_Ld       filters to local deformation radius scale
# - velocity               cross-track geostrophic vel (*note: ignores and does not flip sign in n. vs. s. hemisphere)

# *** Secondardy (older/unused) Functions ***
# - specsharp              create sharp filter kernel, outputs actual filter weights
# - smooth_tracks          master smoothing function, calls specsharp
# - coarsen                (take smoothed signal and coarsen to filter scale)
# - spectra_slopes         (estimate wavenumber spectra and linear slope over defined mesoscale wavenumber band)

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
    return d  # round(d,4)
   
    
# -----------------------------------------------------------------------------------------    
# --- interpolate --- * only works for equi-spaced data --- 
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


# -----------------------------------------------------------------------------------------
def interp_nans(y, nans, x, cutoff):
    # Calculate length of nan segments and select to interpolate if shorter than cutoff
    test = np.where(nans == True)[0]
    nan_seg = []
    # loop over nan indices 
    for i in range(len(test)):
        this_nan_i = test[i]
        # check that this_nan isn't already accounted for in previous element of nan_seg
        if (len(nan_seg) > 0):
            if (nan_seg[-1][-1] >= this_nan_i):
                continue
        # check that first nan isn't last value in array
        if this_nan_i == (len(y) - 1):
            nan_seg.append([this_nan_i, this_nan_i])
            break
        # find next nan in list and determine how long the nan segment is
        if (this_nan_i + 1) < (len(y)):            # check that next index is available 
            # if next value after first nan is finite, end nan segment
            if np.isfinite(y[this_nan_i + 1]):
                nan_seg.append([this_nan_i, this_nan_i])
                continue
            if np.isnan(y[this_nan_i + 1]):            # is next value a nan? 
                # check if last element 
                if (this_nan_i + 1) == (len(y) - 1):
                    nan_seg.append([this_nan_i, this_nan_i + 1])
                    break
                # see how many nans are in this segment
                for j in range(1, 100):                      # assume we don't have nan segments longer than 100 points
                    if this_nan_i + j == (len(y) - 1):       # check if adding j index gets us to the end of the array
                        nan_seg.append([this_nan_i, this_nan_i + j])
                        break                   
                    next_pot_nan = y[this_nan_i + j]                 
                    if np.isfinite(next_pot_nan):      # if next value is finite break look 
                        nan_seg.append([this_nan_i, this_nan_i + j - 1])
                        break 
            
    # print(nan_seg)     
    nan_seg_length = []
    nans_to_interp = nans.copy()
    for i in range(len(nan_seg)):
        nan_seg_length.append(nan_seg[i][-1] - nan_seg[i][0] + 1)
        if (nan_seg[i][-1] - nan_seg[i][0] + 1) > cutoff:
            nans_to_interp[nan_seg[i][0]:nan_seg[i][-1] + 1] = False

    y_interp = y.copy()
    if (np.sum(nans_to_interp) > 0) & (np.sum(nans_to_interp) < len(y)):
        y_interp[nans_to_interp] = np.interp(x(nans_to_interp), x(~nans_to_interp), y[~nans_to_interp])

    return y_interp  # nan_seg, nans_to_interp

# -----------------------------------------------------------------------------------------
# -- parse_grid_tracks uses track and cycle information to separate, organize, and interpolate nans --> prep for smoothing 
def parse_grid_tracks(tracks, df2_s, d_grid_step, interp_cutoff, f_v_uf):
    dist = []
    adt = []
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
        # define grid to interpolate to (varies in length with length of track, but all have same step)
        d_grid = np.arange(0, np.nanmax(this_dist), d_grid_step)
        lon_grid = np.interp(d_grid, this_dist, this_cycle['longitude'].values)
        lat_grid = np.interp(d_grid, this_dist, this_cycle['latitude'].values)  
        if f_v_uf:
            this_time = this_cycle['sla_filtered'].index
        else:
            this_time = this_cycle['sla_unfiltered'].index
        this_time_c = (this_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
        # -- prep for looping over each cycle of this_track
        # - initialize arrays 
        this_mdt_grid = np.nan * np.ones((len(these_cycles), len(d_grid)))
        this_sla_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_lon_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_lat_grid = np.nan * np.ones((len(these_cycles), len(d_grid))) 
        this_time_grid = np.nan * np.ones(len(these_cycles)) 
        this_time_grid[0] = np.nanmean(this_time_c)
        land_mask = 0
        # -- loop over each cycle 
        for c in range(len(these_cycles)):
            this_cycle = sat_track_i[sat_track_i['cycle'] == these_cycles[c]]
            this_dist = Haversine(lat_start, lon_start, this_cycle['latitude'], this_cycle['longitude'])
            
            # choice of filtered or unfiltered sla 
            if f_v_uf:
                this_sla = np.array(this_cycle['sla_filtered'])
            else:
                this_sla = np.array(this_cycle['sla_unfiltered'])
            
            # maybe add mdt to sla such that output is adt (can remove mean later)
            this_ssh = this_sla  # + np.array(this_cycle['mdt'])   
            # interpolate to regular distance grid 
            sla_grid_pass1 = np.interp(d_grid, this_dist, this_ssh) 
        
            if land_mask < 1:
                # deal with land and interpolatings (goal is to only run this once and apply to each cycle in each track)
                land = np.where(np.diff(np.array(this_dist)) > 10)[0]  # these are indices idenify gaps in the data 
                # interpolate across grid, but retain info as to which distances are covered, fill with nans
                this_dist2 = np.array(this_dist)
                for l in range(len(land)):
                    land_i = np.where((d_grid >= this_dist2[land[l]]) & (d_grid <= this_dist2[land[l]+1]))[0]
                    sla_grid_pass1[land_i] = np.nan
                nan_locs = np.where(np.isnan(sla_grid_pass1))[0]
                land_mask = 2
            else:
                sla_grid_pass1[nan_locs] = np.nan
                
            # remove interpolated stretches across land that are repeats of the same data 
            sla_grid_pass1[np.where(np.abs(np.diff(sla_grid_pass1)) < 0.0001)[0]] = np.nan 
            
            # prep for output 
            this_sla_grid[c, :] = sla_grid_pass1 
            this_mdt_grid[c, :] = np.interp(d_grid, this_dist, np.array(this_cycle['mdt']))
            
            if f_v_uf:
                this_time = this_cycle['sla_filtered'].index
            else:
                this_time = this_cycle['sla_unfiltered'].index
            this_time_c = (this_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            this_time_grid[c] = np.nanmean(this_time_c)
        
        # -- interpolate nans (where segment length is less than cutoff length )
        this_interp_sla = np.nan * np.ones(np.shape(this_sla_grid))
        # this_interp_mdt = np.nan * np.ones(np.shape(this_mdt_grid))
        for i in range(np.shape(this_sla_grid)[0]):
            if np.sum(np.isnan(this_sla_grid[i, :])) > 0:
                nans, x = nan_helper(this_sla_grid[i, :])
                this_interp_sla[i, :] = interp_nans(this_sla_grid[i, :], nans, x, interp_cutoff)
            else:
                this_interp_sla[i, :] = this_sla_grid[i, :].copy()
            # if np.sum(np.isnan(this_mdt_grid[i, :])) > 0:
            #     nans, x = nan_helper(this_mdt_grid[i, :])
            #     this_interp_mdt[i, :] = interp_nans(this_mdt_grid[i, :], nans, x, interp_cutoff)
            # else:
            #     this_interp_mdt[i, :] = this_mdt_grid[i, :].copy()
    
        # -- add mdt + sla to obtain adt 
        # check if this_interp_sla is really already adt 
        this_adt = this_interp_sla + this_mdt_grid 
        
        # identify and nan out land (or data gaps)
        lon_grid[np.isnan(np.nanmean(this_interp_sla, axis=0))] = np.nan  # this_sla_grid
        lat_grid[np.isnan(np.nanmean(this_interp_sla, axis=0))] = np.nan 
        
        # save for each track arrays of fields [cycle X Distance] (each array is an element in a list)
        adt.append(this_adt)
        # sla.append(this_sla_grid)
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
        else:
            # -- to run if I want sla, mdt, adt by time increment
            # time_t = np.append(time_t, np.tile(this_time_grid, (1, len(these_cycles))))
            # lon_t = np.append(lon_t, np.tile(lon_grid, (1, len(these_cycles))))
            # lat_t = np.append(lat_t, np.tile(lat_grid, (1, len(these_cycles))))
            # sla_t = np.append(sla_t, this_sla_grid.flatten())
            lon_t = np.append(lon_t, lon_grid.copy())
            lat_t = np.append(lat_t, lat_grid.copy())
            track_t = np.append(track_t, tracks[m] * np.ones(len(d_grid)))                         
        count = count + 1

    return lon_t, lat_t, track_t, adt, sla_int, dist, lon_record, lat_record, time_record, track_record

# -----------------------------------------------------------------------------------------
# NEW FILTERING METHOD
def filterSpec(N,dxMin,Lf,plot_filter,shape="Gaussian", X=np.pi):
    """
    Inputs: 
    N is the number of total steps in the filter
    dxMin is the smallest grid spacing - should have same units as Lf
    Lf is the filter scale, which has different meaning depending on filter shape
    shape can currently be one of two things:
        Gaussian: The target filter has kernel ~ e^{-|x/Lf|^2}
        Taper: The target filter has target grid scale Lf. Smaller scales are zeroed out. 
               Scales larger than pi*Lf/2 are left as-is. In between is a smooth transition.
    X is the width of the transition region in the "Taper" filter; per the CPT Bar&Prime doc the default is pi.
    Note that the above are properties of the *target* filter, which are not the same as the actual filter.
    
    Outputs:
    NL is the number of Laplacian steps
    sL is s_i for the Laplacian steps; units of sL are one over the units of dxMin and Lf, squared
    NB is the number of Biharmonic steps
    sB is s_i for the Biharmonic steps; units of sB are one over the units of dxMin and Lf, squared
    """
    # Code only works for N>2
    if N <= 2:
        print("Code requires N>2")
        return 
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi/2)*(2*np.eye(N-1) - np.diag(np.ones(N-3),2) - np.diag(np.ones(N-3),-2))
    M[0,0] = 3*np.pi/2
    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid. 
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    # sMax = 2*(np.pi/dxMin)**2
    sMax = 1*(np.pi/dxMin)**2
    # Set up target filter
    if shape == "Gaussian":
        F = lambda t: np.exp(-(sMax*(t+1)/2)*(Lf/2)**2)
    elif shape == "Taper":
        F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(np.pi/(X*Lf))**2 -1,(2/sMax)*(np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
    else:
        print("Please input a valid shape")
        return
    # Compute inner products of Galerkin basis with target
    b = np.zeros(N-1)
    points, weights = np.polynomial.chebyshev.chebgauss(N+1)
    for i in range(N-1):
        tmp = np.zeros(N+1)
        tmp[i] = 1
        tmp[i+2] = -1
        phi = np.polynomial.chebyshev.chebval(points,tmp)
        b[i] = np.sum(weights*phi*(F(points)-((1-points)/2 + F(1)*(points+1)/2)))
    # Get polynomial coefficients in Galerkin basis
    cHat = np.linalg.solve(M,b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(N+1)
    p[0] = cHat[0] + (1+F(1))/2
    p[1] = cHat[1] - (1-F(1))/2
    for i in range(2,N-1):
        p[i] = cHat[i] - cHat[i-2]
    p[N-1] = -cHat[N-3]
    p[N] = -cHat[N-2]
    # Now plot the target filter and the approximate filter
    #x = np.linspace(-1,1,251)
    x = np.linspace(-1,1,10000)
    k = np.sqrt((sMax/2)*(x+1))
    
    # --- 
    if plot_filter:
        #fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
        f, ax = plt.subplots()
        #ax1 = plt.subplot(1,2,1)
        params = {'legend.fontsize': 'x-large',
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
        pylab.rcParams.update(params)
        plt.plot(k,F(x),'g',label='target filter',linewidth=4)
        plt.plot(k,np.polynomial.chebyshev.chebval(x,p),'m',label='approximation',linewidth=4)
        #plt.xticks(np.arange(5), ('0', r'$1/\Delta x$', r'$2/\Delta x$',r'$3/\Delta x$', r'$4/\Delta x$'))
        plt.axvline(1/Lf,color='k',linewidth=2)
        plt.axvline(np.pi/Lf,color='k',linewidth=2)
        #plt.text(1/Lf, 1.15, r'$\frac{1}{2}$',fontsize=20)
        #plt.text(np.pi/Lf, 1.15, r'$\frac{\pi}{2}$',fontsize=20)
        left, right = plt.xlim()
        plt.xlim(left=0)
        plt.xlim(right=2)
        bottom,top = plt.ylim()
        plt.ylim(bottom=-0.1)
        plt.ylim(top=1.1)
        plt.xlabel('k', fontsize=18)
        plt.grid(True)
        plt.legend()
        #plt.legend([p1, p2], ['Line Up', 'Line Down'])
        #ax2 = plt.subplot(1,2,2)
        #ax2.plot(k,F(x)-np.polynomial.chebyshev.chebval(x,p),linewidth=3)
        # plt.savefig('figures/filtershape_%s%i_dxMin%i_Lf%i.png' % (shape,N,dxMin,Lf),dpi=400,bbox_inches='tight',pad_inches=0)
    # -------
    
    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)
    # convert back to s in [0,sMax]
    s = (sMax/2)*(r+1)
    # Separate out the real and complex roots
    NL = np.size(s[np.where(np.abs(np.imag(r)) < 1E-12)]) 
    sL = np.real(s[np.where(np.abs(np.imag(r)) < 1E-12)])
    NB = (N - NL)//2
    sB_re,indices = np.unique(np.real(s[np.where(np.abs(np.imag(r)) > 1E-12)]),return_index=True)
    sB_im = np.imag(s[np.where(np.abs(np.imag(r)) > 1E-12)])[indices]
    sB = sB_re + sB_im*1j
    return NL,sL,NB,sB


# -----------------------------------------------------------------------------------------
# older as of jan 2021, use filterSpec() above 
def filterSpec0(N, dxMin, Lf, show_p, shape, X=np.pi):
    """
    Inputs: 
    N is the number of total steps in the filter
    dxMin is the smallest grid spacing - should have same units as Lf
    Lf is the filter scale, which has different meaning depending on filter shape
    shape can currently be one of two things:
        Gaussian: The target filter has kernel ~ e^{-|x/Lf|^2}
        Taper: The target filter has target grid scale Lf. Smaller scales are zeroed out. 
               Scales larger than pi*Lf/2 are left as-is. In between is a smooth transition.
    Note that the above are properties of the *target* filter, which are not the same as the actual filter.
    
    Outputs:
    NL is the number of Laplacian steps
    sL is s_i for the Laplacian steps; units of sL are one over the units of dxMin and Lf, squared
    NB is the number of Biharmonic steps
    sB is s_i for the Biharmonic steps; units of sB are one over the units of dxMin and Lf, squared
    """
    # Code only works for N>2
    if N <= 2:
        print("Code requires N>2")
        return 
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi/2)*(2*np.eye(N-1) - np.diag(np.ones(N-3),2) - np.diag(np.ones(N-3),-2))
    M[0,0] = 3*np.pi/2
    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid. 
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    sMax = 2*(np.pi/dxMin)**2
    # Set up target filter
    if shape == "Gaussian":
        F = lambda t: np.exp(-(sMax*(t+1)/2)*(Lf/2)**2)
    elif shape == "Taper":
        # F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(2/Lf)**2 -1,(2/sMax)*(np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
        F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(np.pi/(X*Lf))**2 -1,(2/sMax)*(np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
        # 2nd entry = (2/sMax)*(np.pi/(X*Lf))**2 -1  X = width of transition (~np.pi)
        # 3rd entry = nyquist wavelength on grid I'm filtering to
    else:
        print("Please input a valid shape")
        return
    # Compute inner products of Galerkin basis with target
    b = np.zeros(N-1)
    points, weights = np.polynomial.chebyshev.chebgauss(N+1)
    for i in range(N-1):
        tmp = np.zeros(N+1)
        tmp[i] = 1
        tmp[i+2] = -1
        phi = np.polynomial.chebyshev.chebval(points,tmp)
        b[i] = np.sum(weights*phi*(F(points)-((1-points)/2 + F(1)*(points+1)/2)))
    # Get polynomial coefficients in Galerkin basis
    cHat = np.linalg.solve(M,b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(N+1)
    p[0] = cHat[0] + (1+F(1))/2
    p[1] = cHat[1] - (1-F(1))/2
    for i in range(2,N-1):
        p[i] = cHat[i] - cHat[i-2]
    p[N-1] = -cHat[N-3]
    p[N] = -cHat[N-2]
    # Now plot the target filter and the approximate filter
    x = np.linspace(-1,1,251)
    k = np.sqrt((sMax/2)*(x+1))
    
    if show_p > 0:
        f, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 2))
        ax1.plot(k,F(x),k,np.polynomial.chebyshev.chebval(x,p))
        ax2.plot(k,F(x)-np.polynomial.chebyshev.chebval(x,p))
        ax1.set_title('approx. filter')
        ax1.set_xlabel('step')
        ax2.set_title('target - approx. error')
        ax2.set_xlabel('step')
        ax2.set_ylim([-0.25, 0.25])
        plt.show()
        
    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)
    # convert back to s in [0,sMax]
    s = (sMax/2)*(r+1)
    # Separate out the real and complex roots
    NL = np.size(s[np.where(np.abs(np.imag(r)) < 1E-12)]) 
    sL = np.real(s[np.where(np.abs(np.imag(r)) < 1E-12)])
    NB = (N - NL)//2
    sB_re,indices = np.unique(np.real(s[np.where(np.abs(np.imag(r)) > 1E-12)]),return_index=True)
    sB_im = np.imag(s[np.where(np.abs(np.imag(r)) > 1E-12)])[indices]
    sB = sB_re + sB_im*1j
    return NL,sL,NB,sB


# -----------------------------------------------------------------------------------------
# newer variant (test out and probably replace filterSpec 
def filterSpec1(dxMin,Lf,d=2,shape="Gaussian",X=np.pi,N=-1,plot_filter=1):
    """
    Inputs: 
    dxMin is the smallest grid spacing - should have same units as Lf
    Lf is the filter scale, which has different meaning depending on filter shape
    d is the dimension of the grid where the filter will be applied 
    shape can currently be one of two things:
        Gaussian: The target filter has kernel ~ e^{-6*|x/Lf|^2}
        Taper: k>=2*pi/Lf are zeroed out, k<=2*pi/(X*Lf) are left as-is, smooth transition in between.
        The std dev of the Gaussian is Lf/sqrt{12}.
    X is the width of the transition region in the "Taper" filter; per the CPT Bar&Prime doc the default is pi.
    Note that the above are properties of the *target* filter, which are not the same as the actual filter.
    
    Outputs:
    NL is the number of Laplacian steps
    sL is s_i for the Laplacian steps; units of sL are one over the units of dxMin and Lf, squared
    NB is the number of Biharmonic steps
    sB is s_i for the Biharmonic steps; units of sB are one over the units of dxMin and Lf, squared
    """
    if N == -1:
        if shape == "Gaussian":
            if d == 1:
                N = np.ceil(1.3*Lf/dxMin).astype(int)
            else: # d==2
                N = np.ceil(1.8*Lf/dxMin).astype(int)
        else: # Taper
            if d == 1:
                # N = np.ceil(4.5*Lf/dxMin).astype(int)  # what ian selected 
                N = np.ceil(4.5*Lf/dxMin).astype(int)
            else: # d==2
                N = np.ceil(6.4*Lf/dxMin).astype(int)
        print("Using default N, N = " + str(N) + " If d>2 or X is not pi then results might not be accurate.")
    # Code only works for N>2
    if N <= 2:
        print("Code requires N>2. If you're using default N, then Lf is too small compared to dxMin")
        return 
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi/2)*(2*np.eye(N-1) - np.diag(np.ones(N-3),2) - np.diag(np.ones(N-3),-2))
    M[0,0] = 3*np.pi/2
    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid. 
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    sMax = d*(np.pi/dxMin)**2
    # Set up target filter
    if shape == "Gaussian":
        F = lambda t: np.exp(-(sMax*(t+1)/2)*Lf**2/24)
    elif shape == "Taper":
        # F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(2*np.pi/(X*Lf))**2 -1,(2/sMax)*(2*np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
        transition_width = X
        filter_scale = Lf
        FK = interpolate.PchipInterpolator(np.array([0, 2 * np.pi / (transition_width * filter_scale), 2 * np.pi / filter_scale, 2 * np.sqrt(sMax)]), np.array([1, 1, 0, 0]))
        F = lambda t: FK(np.sqrt((t + 1) * (sMax / 2)))
    else:
        print("Please input a valid shape: Gaussian or Taper")
        return
    # Compute inner products of Galerkin basis with target
    b = np.zeros(N-1)
    points, weights = np.polynomial.chebyshev.chebgauss(N+1)
    for i in range(N-1):
        tmp = np.zeros(N+1)
        tmp[i] = 1
        tmp[i+2] = -1
        phi = np.polynomial.chebyshev.chebval(points,tmp)
        b[i] = np.sum(weights*phi*(F(points)-((1-points)/2 + F(1)*(points+1)/2)))
    # Get polynomial coefficients in Galerkin basis
    cHat = np.linalg.solve(M,b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(N+1)
    p[0] = cHat[0] + (1+F(1))/2
    p[1] = cHat[1] - (1-F(1))/2
    for i in range(2,N-1):
        p[i] = cHat[i] - cHat[i-2]
    p[N-1] = -cHat[N-3]
    p[N] = -cHat[N-2]
    # Now plot the target filter and the approximate filter
    #x = np.linspace(-1,1,251)
    x = np.linspace(-1,1,10000)
    k = np.sqrt((sMax/2)*(x+1))
    #params = {'legend.fontsize': 'x-large',
    #     'axes.labelsize': 'x-large',
    #     'axes.titlesize':'x-large',
    #     'xtick.labelsize':'x-large',
    #     'ytick.labelsize':'x-large'}
    #pylab.rcParams.update(params)
    
    if plot_filter:
        # f, ax = plt.subplots(1,1,figsize=(7,5))
        plt.plot(k,F(x),'g',label='target filter',linewidth=4)
        if shape=="Gaussian":
            plt.axvline(2*np.pi/(np.sqrt(12)*Lf),color='m',linewidth=2, label=' Gaussian std', linestyle='--')
            plt.plot(k,np.polynomial.chebyshev.chebval(x,p),'m',label='Gaussian approximation',linewidth=3)
        else:
            plt.plot(k,np.polynomial.chebyshev.chebval(x,p),'b',label='Taper approximation',linewidth=3)
            plt.axvline(2*np.pi/(X*Lf),color='b',linewidth=1)
            plt.axvline(2*np.pi/Lf,color='b',linewidth=1, label='Taper cutoff',linestyle='--')
        left, right = plt.xlim()
        plt.xlim(left=0, right=2)
        bottom,top = plt.ylim()
        plt.ylim(bottom=-0.2, top=1.1)
        plt.xlabel(r'k [$\sqrt{\frac{1}{2}\frac{\pi^2}{dx^2}(x+1)}$]', fontsize=15)
        plt.title('Lf = ' + str(Lf), fontsize=15)
        plt.grid(True)
           
    
    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)
    # convert back to s in [0,sMax]
    s = (sMax/2)*(r+1)
    # Separate out the real and complex roots
    NL = np.size(s[np.where(np.abs(np.imag(r)) < 1E-12)]) 
    sL = np.real(s[np.where(np.abs(np.imag(r)) < 1E-12)])
    NB = (N - NL)//2
    sB_re,indices = np.unique(np.real(s[np.where(np.abs(np.imag(r)) > 1E-12)]),return_index=True)
    sB_im = np.imag(s[np.where(np.abs(np.imag(r)) > 1E-12)])[indices]
    sB = sB_re + sB_im*1j
    return p,NL,sL,NB,sB


# -----------------------------------------------------------------------------------------
def Laplacian1D(field,landMask,dx):
    """
    Computes a Cartesian Laplacian of field. Assumes dy=constant, dx varies in y direction
    Inputs:
    field is a 1D array (x) whose Laplacian is computed
    landMask: 1D array, same size as field: 0 if cell is not on land, 1 if it is on land.
    dx is a 1D array, size size as 2nd dimension of field
    Output:
    Laplacian of field.
    """
    Nx = np.size(field,0)
    # Ny = np.size(field,1) # I suppose these could be inputs
    notLand = 1 - landMask
    # first compute Laplacian in y direction. "Right" is north and "Left" is south for this block
    fluxRight = np.zeros(Nx)
    fluxRight[0:Nx-1] = notLand[1:Nx]*(field[1:Nx] - field[0:Nx-1]) # Set flux to zero if on land
    # fluxRight[:,Ny-1] = notLand[:,0]*(field[:,0]-field[:,Ny-1]) # Periodic unless there's land in the way
    
    fluxLeft = np.zeros(Nx)
    fluxLeft[1:Nx] = notLand[0:Nx-1]*(field[1:Nx] - field[0:Nx-1]) # Set flux to zero if on land
    # fluxLeft[:,0] = notLand[:,Ny-1]*(field[:,0]-field[:,Ny-1]) # Periodic unless there's land in the way
    OUT = (1/(dx**2))*(fluxRight - fluxLeft)
    # Now compute Laplacian in x direction and add it back in
    # fluxRight = 0*fluxRight # re-set to zero just to be safe
    # fluxLeft = 0*fluxLeft # re-set to zero just to be safe
    # fluxRight[0:Nx-1,:] = notLand[1:Nx,:]*(field[1:Nx,:] - field[0:Nx-1,:]) # Set flux to zero if on land
    # fluxRight[Nx-1,:] = notLand[0,:]*(field[0,:]-field[Nx-1,:]) # Periodic unless there's land in the way
    # fluxLeft[1:Nx,:] = notLand[0:Nx-1,:]*(field[1:Nx,:] - field[0:Nx-1,:]) # Set flux to zero if on land
    # fluxLeft[0,:] = notLand[Nx-1,:]*(field[0,:]-field[Nx-1,:]) # Periodic unless there's land in the way
    # OUT = OUT + (1/(dx**2))*(fluxRight - fluxLeft)
    return OUT*notLand

# -----------------------------------------------------------------------------------------
# FILTER DATA with filterSpec and Laplacian1D
def Filter(N, filter_type, field, dx, coarsening_factor, *args, **kwargs):
    
    if filter_type == 'boxcar':
        sla_filt_out = []
        for m in range(len(field)):  # tqdm(range(len(field))):  # loop over each track
            this_sla = field[m]
            b_filt = (1/(coarsening_factor))*np.ones(coarsening_factor)
            sla_filt = np.nan * np.ones(np.shape(this_sla))
            for i in range(np.shape(this_sla)[0]):
                sla_filt[i, :] = np.convolve(this_sla[i, :], b_filt, mode='same')     
            sla_filt_out.append(sla_filt)    
    else:
        plot_filter = kwargs.get('plot_filter', 0)
        NL,sL,NB,sB = filterSpec(N, dx, coarsening_factor, plot_filter, filter_type, X=np.pi)
        sla_filt_out = []
        # each track
        for c in range(len(field)):  # tqdm(range(len(field))):
            sla_filt = np.nan * np.ones(np.shape(field[c]))
            land = np.where(np.isnan(field[c][0, :]))[0]
            landMask = np.zeros(np.shape(field[c])[1])
            landMask[land] = 1
            # each cycle
            for m in range(np.shape(field[c])[0]):
                data = field[c][m, :].copy()
                # tempL_out = np.nan * np.ones((NL, np.shape(field)[0], np.shape(field)[1]))
                for i in range(NL):
                    tempL = Laplacian1D(data,landMask,dx)
                    # tempL_out[i, :, :] = tempL.copy()
                    data = data + (1/sL[i])*tempL # Update filtered field
                for i in range(NB):
                    tempL = Laplacian1D(data, landMask, dx)
                    tempB = Laplacian1D(tempL, landMask, dx)
                    data = data + (2*np.real(sB[i])/(np.abs(sB[i])**2))*tempL + (1/(np.abs(sB[i])**2))*tempB
                sla_filt[m, :] = data
            sla_filt_out.append(sla_filt)
            
    return(sla_filt_out)

# -----------------------------------------------------------------------------------------
# -- ALT FILTERING FUNCTION 1
# define function to filter using a filter of variable length 
# (filter scale = local distance equal to a desired grid step in longitude i.e. 1/4 degree)
# filter USED is GAUSSIAN 
def smooth_tracks_deg(dist, sla, lon_record, lat_record, resolution, sigma):
    # resolution = desired grid scale to filter to (i.e. 1/4 degree)
    # sigma = gaussian standard deviation 
    sla_filtered = []
    for m in tqdm(range(len(sla))):
        this_sla = sla[m]
        this_lon = lon_record[m]
        this_lat = lat_record[m]
        this_dist = dist[m]     
        this_lon_step = 1852 * 60 * np.cos(np.deg2rad(this_lat)) * (resolution)
        sla_filt = np.nan * np.ones(np.shape(this_sla))
        for i in range(11, np.shape(this_sla)[1] - 11):  # loop across all space
            if np.isnan(this_lon_step[i]):
                continue
            this_local_grid = np.arange(-this_lon_step[i]*4, this_lon_step[i]*5, this_lon_step[i])   # create local grid 
            for k in range(np.shape(this_sla)[0]):                                                   # loop in time (across each cycle)
                sla_on_local_lon_grid = np.interp(this_local_grid, (this_dist[i-10:i+11]*1000) - this_dist[i]*1000, this_sla[k, i-10:i+11])
                sla_filt[k, i] = si.gaussian_filter(sla_on_local_lon_grid, sigma, order=0)[10]       # extract middle value [index=10]       
        sla_filtered.append(sla_filt)
    return sla_filtered

# -----------------------------------------------------------------------------------------
# -- ALT FILTERING FUNCTION 2
# define a function to filter where local filter width is defined using the local deformation radius
# filter used is GAUSSIAN
def smooth_tracks_Ld(dist, sla, lon_record, lat_record, resolution, c98):
    # resolution = horizontal grid spacing 
    # c98 = array of deformation radii 
    # filter scale = local deformation radius / resolution = filter width in number of grid points
    sla_filtered = []
    for m in tqdm(range(len(sla))):  # loop over each track
        this_sla = sla[m]
        this_lon = lon_record[m]
        this_lat = lat_record[m]      
        # at each location along track, find local deformation radius 
        sla_filt = np.nan * np.ones(np.shape(this_sla))
        for i in range(10, np.shape(this_sla)[1] - 11):  # loop across all space for each track
            # print(this_lon[i])
            if np.isnan(this_lon[i]):
                continue
            # find deformation radius 
            c_in = np.where((c98[:, 0] > this_lat[i]-0.75) & (c98[:, 0] < this_lat[i]+0.75) & \
                     (c98[:, 1] > this_lon[i]-0.75) & (c98[:, 1] < this_lon[i]+0.75))[0]
            if len(c_in) >= 1:
                this_Ld = np.nanmean(c98[c_in, 3])
                for k in range(np.shape(this_sla)[0]):  # loop in time 
                    # apply filter across a subset of points and then take middle (relevant value) [10th index]
                    sla_filt[k, i] = si.gaussian_filter(this_sla[k, i-10:i+11], this_Ld/resolution, order=0)[10]
        sla_filtered.append(sla_filt)
    return sla_filtered

# -----------------------------------------------------------------------------------------
# -- CROSS-TRACK GEOSTROPHIC VELOCITY (from sla or adt)
# --------------------------------------------------------------------
def velocity(dist, sla, lon_record, lat_record, track_record, stencil_width):   
    transition_lat = 5  # latitude to smoothly transition to beta-plane from local f-plane
    vel = []
    vel_f = []
    grad = []
    count = 0
    for m in tqdm(range(len(track_record))):
        # -- load in data for this track 
        this_sla = sla[m]            # interpolated field, sla is just a place holder (confusing I know)
        lon_grid = lon_record[m]
        lat_grid = lat_record[m]
        d_grid = dist[m]
        grid_space = d_grid[1] - d_grid[0]
        these_cycles = np.arange(0, np.shape(this_sla)[0])                 
        if len(d_grid) < 10:
            print('track ' + str(m) + ', too short') 
            grad.append(np.nan * np.ones(np.shape(this_sla)))
            vel.append(np.nan * np.ones(np.shape(this_sla)))
            continue
            
        # -- gradient ([Arbic 2012]) (pol_rad = 6378.137km) (eq_rad = 6356.752km) 
        f_loc = 2*(7.27*10**(-5))*np.sin(np.deg2rad(lat_grid))    
        sla_grad = np.gradient(this_sla, d_grid*1000.0, axis=1)
        for cdm in range(4, 4 + len(sla_grad[0, 4:-3])):
            # -- gradients from a 7 point stencil 
            if stencil_width == 7:   
                sla_grad[:, cdm] = (this_sla[:, cdm+3] - 9*this_sla[:, cdm+2] + 45*this_sla[:, cdm+1] \
                                    - 45*this_sla[:, cdm-1] + 9*this_sla[:, cdm-2] - this_sla[:, cdm-3]) / (60*(grid_space*1000.0))   
            # -- gradients from a 5 point stencil
            elif stencil_width == 5:   
                sla_grad[:, cdm] = (-this_sla[:, cdm+2] + 8*this_sla[:, cdm+1] - 8*this_sla[:, cdm-1] \
                                    + this_sla[:, cdm-2]) / (12*(hor_grid_spacing*1000.0))
            elif stencil_width == 3: 
                sla_grad[:, cdm] = (this_sla[:, cdm+1] - this_sla[:, cdm-1]) / (2*(hor_grid_spacing*1000.0))  
            else:
                print('select either 3,5,7 for gradient stencil_width')
                
        # velocity via geostrophic balance 
        this_vel = (9.81/np.tile(f_loc[None, :], (len(these_cycles), 1))) * sla_grad 
        this_vel_f = this_vel.copy()
        
        # -- near equator attempt beta plane correction from [Lagerloef 1999] 
        close_eq1 = np.where(np.abs(lat_grid) < transition_lat)[0]
        if len(close_eq1) > 10:  # if there are points close to equator, make sure there are enough to compute a gradient
            beta = 2*(7.27*10**(-5))*np.cos(np.deg2rad(lat_grid[close_eq1]))/(6356752)
            y = 1852 * 60 * (lat_grid[close_eq1] - 0)  # 6356752*lat_grid[close_eq1]  
            # -- weights transitioning from beta plane to f plane 
            wb = np.exp(-(np.abs(lat_grid[close_eq1])/2.2)**2)
            wf = 1 - wb           
            # L = 111000, theta = y/L
            
            # -- geostrophic balance 
            uf = (9.81/(np.tile(f_loc[close_eq1][None, :], (len(these_cycles), 1)))) * sla_grad[:, close_eq1]  
            # -- approximate the along-track distance of d_eta/dx
            # d_sla_grad_dx = np.zeros(np.shape(sla_grad[:, close_eq1]))
            # d_sla_grad_dx[:, 1:-1] = (sla_grad[:, close_eq1[2:]] - sla_grad[:, close_eq1[0:-2]])/(y[2:] - y[0:-2])
            # ub = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1)))) * d_sla_grad_dx            
            ub = (9.81/(y*np.tile(beta[None, :], (len(these_cycles), 1)))) * sla_grad[:, close_eq1]
            # ub = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1)))) * np.gradient(sla_grad[:, close_eq1], y, axis=1)  # improper gradient estimate
            # -- attempt at asympototic solution
            # ub1 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*y)) * adt_grad[:, close_eq1] # * np.tile(theta[None, :], (len(these_cycles), 1))
            # ub2 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*L)) * \
            #     adt_grad[:, close_eq1] * np.tile(theta[None, :]**2, (len(these_cycles), 1))
            # ub3 = (9.81/(np.tile(beta[None, :], (len(these_cycles), 1))*L)) * \
            #     adt_grad[:, close_eq1] * np.tile(theta[None, :]**3, (len(these_cycles), 1))
            # ub = ub1  # ub1 + ub2 + ub3
            
            # -- combine uf, wb each scaled by weights 
            ug = np.tile(wb[None, :], (len(these_cycles), 1))*ub + np.tile(wf[None, :], (len(these_cycles), 1))*uf
            this_vel[:, close_eq1] = ug 
            
            # -- debugging / inspecting actual values 
            # if m == 23:
            #     print(beta)
            #     print(lat_grid[close_eq1])
            #     # print(d_sla_grad_dx[5, :])
            #     print(ub[5, :])
            #     print(ug[5, :] - uf[5, :])
            #     print(ug[5, :])
        
        # -- save for each track arrays of fields [cycle X Distance] (each array is an element in a list)
        grad.append(sla_grad)
        vel.append(this_vel)
        vel_f.append(this_vel_f)
               
    return grad, vel, vel_f


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# SECONDARY FUNCTIONS 
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# create filter kernel, function of
# - grid step (grid_spacing) (I'm using units of km) 
# - coarsening factor (x) (grid_spacing * coarse_fac = desired grid step) 
# - and nyquist wavenumber
#      - relative to desired grid step (or new coarsened grid), what is our smallest resolvable scale on new grid
#      - true nyquist wavenumber = grid_spacing * nyquist_wavenumber 
def specsharp(grid_spacing, x, nyquist_wavelength):     
    # -- find filter weights for a given filter width, n grid cells
    def getWeights(n):  
        w = np.zeros(n+1) # initialize weights
        for i in range(n):
            integrand = lambda k: 2*(F(k)-1)*(np.cos((i+1)*k)-1)
            w[i] = integrate.quad(integrand,0,np.pi)[0]
        A = 2*np.pi*(np.eye(n) + 2)
        w[1:] = np.linalg.solve(A,w[0:n])
        w[0] = 1 - 2*np.sum(w[1:])
        return w
    
    # F = interpolate.PchipInterpolator(np.array([0, 1/x, nyquist_wavelength/x, nyquist_wavelength]), np.array([1, 1, 0, 0]))
    F = interpolate.PchipInterpolator(np.array([0, np.pi/(nyquist_wavelength*x), np.pi/x, np.pi]), np.array([1, 1, 0, 0]))
    print('Filter Half-Width = ')    
            
    weight_prev = getWeights(2)
    # loop over filter widths until weights converge
    for j in range(3, 80):  # 70 iterations should be enough unless coarsening scale is really large 
        this_weight = getWeights(j)
        # difference between these weights and last iterations (looking for convergence over first 4 weights)
        wd = np.sum(np.abs(this_weight[0:3] - weight_prev[0:3]) / np.abs(weight_prev[0:3]))  
        if wd < 0.002:  # convergence threshold (arbitrary threshold...I'd rather be too conservative) 
            jj = j      # jj = have filter length (in number of grid cells) 
            print(str(j - 1) + ' ' + str(getWeights(j - 1)[0:4]))
            print(str(j) + ' ' + str(getWeights(j)[0:4]))
            print('converged //')
            break
        weight_prev = this_weight   
            
    filter_kernel = np.concatenate((np.flip(getWeights(jj))[0:-1], getWeights(jj)))
    print('------------------------------------------------------------')
    print('for a coarsening factor of ' + str(x) + ', recommend:')
    print('-- filter width of ' + str(2*jj + 1) + ' grid cells (here = ' + str(grid_spacing * (2*jj + 1)) + ' km)')
    print('------------------------------------------------------------')
    return filter_kernel, jj


# -----------------------------------------------
# smoothing function that calls filtering function (depending on choice of filter) 
# does actual smoothing 
def smooth_tracks(dist, data, track_record, coarsening_factor, filter_choice, filter_kernel, nyquist_wavelength, space_time):
    
    # if filter_choice == 'sharp':
        # * already have our filter kernel as input 
        # compute filter kernel and weights 
        # filter_kernel, jj = specsharp(hor_grid_spacing, coarsening_factor, nyquist_wavelength)
    if filter_choice == 'gaussian':
        sigma = coarsening_factor 
        
    data_smooth = []
    count = 0
    for m in tqdm(range(len(track_record))):
        # -- load in data for this track 
        if space_time < 1:
            this_data = data[m]  # interpolated field [cycle X distance]
        else:
            this_data = np.transpose(data[m])

        d_grid = dist[m]
        these_cycles = np.arange(0, np.shape(this_data)[0])
                  
        if len(d_grid) < 10:
            print('track ' + str(m) + ', too short') 
            data_smooth.append(np.nan * np.ones(np.shape(this_data)))
            continue
        
        # -- smooth HERE for each cycle 
        if filter_choice == 'gaussian':
            smoothed_data = np.nan * np.ones(np.shape(this_data))
            for j in range(np.shape(this_data)[0]):     
                smoothed_data[j, :] = si.gaussian_filter(this_data[j, :], sigma, order=0)
            
        if filter_choice == 'sharp':    
            # smoothed_sla = sharp_smooth(filter_kernel, this_sla)
            # smoothed_adt = sharp_smooth(filter_kernel, this_adt)
            smoothed_data = np.nan * np.ones(np.shape(this_data))
            for i in range(np.shape(this_data)[0]):
                data_convolve = np.convolve(filter_kernel, this_data[i, :])
                smoothed_data[i, :] = data_convolve[np.int(np.floor(len(filter_kernel)/2)):-np.int(np.floor(len(filter_kernel)/2))]
                smoothed_data[i, 0:np.int(np.floor(len(filter_kernel)/2))] = np.nan
                smoothed_data[i, -np.int(np.floor(len(filter_kernel)/2)):] = np.nan
        
        if space_time < 1:
            data_smooth.append(smoothed_data)
        else:
            data_smooth.append(np.transpose(smoothed_data))
    
    return data_smooth


# -----------------------------------------------------------------------------------------
# coarsen data after it has been smoothed (part two of coarse graining process)
# inputs:
# - dist = list of horizontal grids for each satellite track 
# - lon_record, lat_record = corresponding latitude and longitude points of every grid point
# - coarsening factor = multiplier of initial grid scale (i.e. dist[mm][1] - dist[mm][0])
# - sig_in = signal (where for example sig_in[mm] has units [cycle_number, grid] 
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
                coarse_i[:, j - 1] = np.nanmean(smooth_sig[:, (this_dist > coarse_grid[j - 1]) &\
                                                           (this_dist < coarse_grid[j])], axis=1)  
                coarse_lon[j - 1] = np.nanmean(this_lon[(this_dist > coarse_grid[j - 1]) & (this_dist < coarse_grid[j])])
                coarse_lat[j - 1] = np.nanmean(this_lat[(this_dist > coarse_grid[j - 1]) & (this_dist < coarse_grid[j])])
        
        coarse_grid_out.append(coarse_grid_c)
        coarse_lon_out.append(coarse_lon)
        coarse_lat_out.append(coarse_lat)
        coarse_sig_out.append(coarse_i)
                    
    return coarse_grid_out, coarse_lon_out, coarse_lat_out, coarse_sig_out

# -----------------------------------------------------------------------------------------
# -- horizontal wavenumber spectra --
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


# OLD FUNCTIONS 
# -----------------------------------------------------------------------------------------
# take filter_kernel created by specsharp() and convolve it with signal 
# - REPLACED by np.convolve...but needed if we define a filter in degrees rather than fixed km width 
# def sharp_smooth(filter_kernel, signal0):            
#     n = np.int((len(filter_kernel) - 1)/2) # filter half-width
#     filter_width = len(filter_kernel)
#     smooth_sig = np.nan*np.ones(np.shape(signal0))
#     for p in range(np.shape(signal0)[0]):     # -- loop over each cycle of each track 
#         this_sig = signal0[p, :].copy()
#         for j in range(len(this_sig)):        # -- loop over each grid point and smooth. 
#             if j < n:                         # edge0 (ignore and don't smoothe edge)
#                 continue
#             elif j >= (len(this_sig) - n):    # edge1 (ignore and don't smoothe edge)
#                 continue
#             else:
#                 if np.sum(np.isnan(this_sig[(j - n):(j + n + 1)])) < 1:  # check that there are no nans in signal to be filtered 
#                     smooth_sig[p, j] = np.nansum(filter_kernel * this_sig[(j - n):(j + n + 1)])
#         smooth_sig[p, np.isnan(this_sig)] = np.nan
#     return smooth_sig
# -----------------------------------------------------------------------------------------
# def interpolate_nans(data, grid, cutoff):  # SLA 
#     num_tracks = np.shape(data)[0]
#     # -- search for good data
#     # look at each track at each depth and inventory the number of nans, but also the number of nan segments and the segment lengths 
#     seg_out = {}
#     seg_out_count = {}
#     good_indices_0 = []
#     good_indices = np.zeros(num_tracks)
#     # print('interpolating by track')
#     for i in range(num_tracks):  # loop over tracks
#         this_track = data[i, :]  # CALL ACTUAL DATA HERE 
#         bad = np.where(np.isnan(this_track))[0]  # nan indices 
#         seg = []
#         if ((len(bad) >= 2) & (len(bad) < len(this_track))):
#             breaky = np.where(np.diff(bad) > 1)[0] + 1  # look for breaks in list of nans 
#             if len(breaky) > 0:      
#                 seg.append([bad[0], bad[breaky[0] - 1]])
#                 for b in range(len(breaky) - 1):
#                     seg.append([bad[breaky[b]], bad[breaky[b + 1] - 1]])
#                 # last index 
#                 if bad[breaky[-1]] == bad[-1]:
#                     seg.append([bad[breaky[-1]], bad[breaky[-1]]])    
#                 else:
#                     seg.append([bad[breaky[-1]], bad[breaky[-1]]]) 
#             elif (len(breaky) == 0) & (bad[0] > 0): 
#                 seg = [bad[0], bad[-1]]    
#             elif (len(breaky) == 0) & (bad[0] == 0):
#                 seg = [bad[0], bad[-1]]   
    
#         elif len(bad) == 1:
#             seg = [bad[0], bad[0]]
#         elif len(bad) == len(this_track):
#             seg = len(data[0, :])  # all are nan's 
#         else:
#             seg = 0  # none are nans     
#         seg_out[i] = seg    
#         # this is a dictionary with coordinates (1, profile) identifying nan segments and their length 
                
#         # inspect seg_out to see which are good and which might meet some defined criteria 
#         if (seg != 0) & (seg != len(data[0, :])):
#             spacer = np.nan * np.ones(len(seg))
#             if len(np.shape(seg)) > 1:
#                 for b2 in range(len(seg)):
#                     spacer[b2] = seg[b2][1] - seg[b2][0]
#                 seg_out_count[i] = spacer  # np.nanmax(spacer)  
#             else:
#                 seg_out_count[i] = seg[1] - seg[0]
#         elif seg == 0:
#             seg_out_count[i] = np.nan          
        
#     # interpolate
#     interpolated_signal = data.copy()
#     for i in range(num_tracks): # interpolate only transects that meet nan_seg criteria 
#         this_u = data[i, :]
#         interp_sig = interpolated_signal[i, :].copy()
#         these_segments = seg_out[i]
#         if these_segments == 0:
#             # there are no nans 
#             continue 
#         if np.sum(np.isnan(this_u)) == len(this_u):
#             continue
#         else:
#             seggy = len(these_segments)
#             for j in range(seggy):
#                 if len(np.isfinite(np.shape(seg_out_count[i]))) == 1:
#                     if these_segments[j][0] == (len(this_u) - 1):
#                         continue
#                     if seg_out_count[i][j] <= cutoff:
#                         this_seg_s = these_segments[j][0] - 1
#                         this_seg_e = these_segments[j][-1] + 1
#                         interp_sig[this_seg_s:this_seg_e+1] = np.interp(grid[this_seg_s:this_seg_e+1], [grid[this_seg_s], grid[this_seg_e]], [this_u[this_seg_s], this_u[this_seg_e]])
#                 elif len(seg_out_count) == 1:
#                     interp_sig[these_segments[0]:these_segments[-1]+1] = np.interp(grid[these_segments[0]:these_segments[-1]+1], [grid[these_segments[0]], grid[these_segments[-1]]], [this_u[these_segments[0]], this_u[these_segments[-1]]])
            
#         interpolated_signal[i, :] = interp_sig
            
#     return interpolated_signal    
# -----------------------------------------------------------------------------------------

# # -- interpolate and smooth [MxN] array of sla (I apply this code in my parsing function directly such that I can list the output for binning process later on)    
# def smooth_grid_tracks(cutoff, dist, sla, track_record, lon_record, lat_record, sigma):
#     # for each track 
#     sla_s = []
#     for i in tqdm(range(len(track_record))):
#         this_dist = dist[i]
#         this_sla = sla[i]
#         # interp 
#         this_interp_sla = interpolate_nans(this_sla, this_dist, cutoff)
#         # smooth for each pass 
#         smoothed_sla_gauss = np.nan * np.ones(np.shape(this_sla))
#         for j in range(np.shape(this_sla)[0]):     
#             smoothed_sla_gauss[j, :] = si.gaussian_filter(this_interp_sla[j, :], sigma, order=0)
#         sla_s.append(smoothed_sla_gauss)
              
#     return(this_interp_sla, sla_s)