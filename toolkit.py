# tools 
import numpy as np
import matplotlib.pyplot as plt


def plot_pro(ax):
    ax.grid()
    plt.show(block=False)
    plt.pause(0.1)
    return ()


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])


def unq_searchsorted(A, B):
    # Get unique elements of A and B and the indices based on the uniqueness
    unqA, idx1 = np.unique(A, return_inverse=True)
    unqB, idx2 = np.unique(B, return_inverse=True)
    # Create mask equivalent to np.in1d(A,B) and np.in1d(B,A) for unique elements
    mask1 = (np.searchsorted(unqB, unqA, 'right') - np.searchsorted(unqB, unqA, 'left')) == 1
    mask2 = (np.searchsorted(unqA, unqB, 'right') - np.searchsorted(unqA, unqB, 'left')) == 1
    # Map back to all non-unique indices to get equivalent of np.in1d(A,B), 
    # np.in1d(B,A) results for non-unique elements
    return mask1[idx1], mask2[idx2]


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def is_number(a):
    # will be True also for 'NaN'
    try:
        number = float(a)
        return True
    except ValueError:
        return False


# nan-seg interp (interpret nan's found in an array...does not deal with a nan segment that ends an array)
def nanseg_interp(xx, y):
    n = len(y)
    iv = np.where(np.isnan(y))[0]  # index of NaN values in y
    diffiv = np.diff(iv)
    nb = np.size(np.where(diffiv > 1)[0]) + 1  # number of blocks of NaNs to be interpolated
    yi = y.copy()

    if len(iv) < 1:
        b = 23
    else:
        if iv[0] == 0:
            ing = np.where(np.isfinite(y))[0][0]
            yi[0:ing] = y[ing]
            nb = nb - 1

        for jj in range(nb):
            ilg = np.where(np.isnan(yi))[0][0] - 1  # index of last y value before first NaN
            if np.sum(np.isfinite(yi[(ilg + 1):n])) > 0:
                ing = np.where(np.isfinite(yi[(ilg + 1):n]))[0][0] + ilg + 1
                yi[(ilg + 1):ing] = np.interp(xx[(ilg + 1):ing], [xx[ilg], xx[ing]], [y[ilg], y[ing]])
    return yi


def data_covariance(den_anom, x, y, time, dt, ds, Ls, Lt):
    den_var = np.var(den_anom)

    # time lag
    l = 0
    t_lag = np.nan * np.zeros(len(time) * len(time))
    for i in range(len(time)):
        for j in range(len(time)):
            t_lag[l] = np.abs(time[i] - time[j])
            l = l + 1
    del_t = t_lag.copy()
    # dist lag
    k = 0
    del_x = np.nan * np.zeros(len(x) * len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            del_x[k] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            k = k + 1
    # den anom product
    m = 0
    prod1 = np.nan * np.zeros(len(x) * len(x))
    for i in range(len(den_anom)):
        for j in range(len(den_anom)):
            prod1[m] = den_anom[i] * den_anom[j]
            m = m + 1

    # look at den_anom^2 as a function of distance 
    x1 = np.reshape(del_t, (len(x), len(x)))
    x2 = np.reshape(del_x, (len(x), len(x)))
    x3 = np.reshape(prod1, (len(x), len(x)))

    # create a grid of dt & ds
    # average prod(ucts) within each [dt,ds] bin
    XX = np.arange(0, 270, dt)
    YY = np.arange(0, 100000, ds)
    avgprod1 = np.zeros((len(YY), len(XX)))

    for h in range(1, len(YY)):
        for r in range(1, len(XX)):
            in1 = np.where((del_t >= XX[r - 1]) & (del_t < XX[r]))[0]
            in2 = np.where((del_x >= YY[h - 1]) & (del_x < YY[h]))[0]
            in3 = np.intersect1d(in1, in2)
            avgprod1[h - 1, r - 1] = np.nanmean(prod1[in3])

    ## look at distribution of data in delt, dist space
    CLIM = [np.nanmin(avgprod1), np.nanmax(avgprod1)]
    XXX = XX + .5 * (XX[1] - XX[0])
    YYY = YY + .5 * (YY[1] - YY[0])

    [xx, yy] = np.meshgrid(XXX, YYY)
    errsq = 0
    varr = den_var - errsq
    cov_est = varr * np.exp(((-(yy ** 2)) / Ls ** 2 - xx / Lt))

    fig, ax = plt.subplots()
    ax.pcolor(XXX, YYY, avgprod1, vmin=CLIM[0], vmax=CLIM[1])
    ax.contour(XXX, YYY, cov_est, 10, vmin=CLIM[0], vmax=CLIM[1])
    plot_pro(ax)

    return den_var, cov_est


def trend_fit(lon, lat, data):
    A = np.transpose([lon, lat, lon / lon])
    b = data
    C = np.linalg.lstsq(A, b)
    return C[0][0], C[0][1], C[0][2]


def trifilt(x, n):
    def x_triang(n):
        # W = TRIANG(N) returns the N-point triangular window.
        if np.remainder(n, 2):
            # It's an odd length sequence
            w1 = 2 * (np.arange(1, (n + 2) / 2)) / (n + 1)
            w2 = np.fliplr(w1[None, 0:-1])
            w = np.concatenate((np.transpose(w1[None, :]), np.transpose(w2)), axis=0)
        else:
            # It's even
            w1 = 2 * (np.arange(1, (n + 1) / 2)) / n
            w2 = np.fliplr(w1[None, 0:-1])
            w = np.concatenate((np.transpose(w1[None, :]), np.transpose(w2)), axis=0)
        return w
        # xf is x filtered with a triangular filter of half-width n

    # xf has the same length as x so that features in xf and x
    # line up.  Endpoints are corrected by filter area so that
    # effective filter area is half at the endpoints and progressively
    # increases to unity 2*n points from the ends of the series x
    m = len(x)
    g = x_triang(2 * n - 1) / n
    y = np.convolve(x, g[:, 0])
    s = len(y)
    xf = y[np.int((s - m) / 2 + 1):np.int((s - m) / 2 + m + 1)]
    uu = np.ones((m, 1))
    vv = np.convolve(uu[:, 0], g[:, 0])
    uu = vv[np.int((s - m) / 2 + 1):np.int((s - m) / 2 + m + 1)]
    xf = xf / uu
    return xf

# for power spectrum (finding break in slope, linear trends in log space)
# --- fminsearch to find break in slope that best matches k-5/3 k -3
def spectrum_fit(ipoint_0, x, pe):
    x = np.log10(x)
    pe = np.log10(pe)
    ipoint = np.log10(ipoint_0)
    l_b = np.nanmin(x)
    r_b = np.nanmax(x)
    x_grid = np.arange(l_b, r_b, 0.01)
    pe_grid = np.interp(x_grid, x, pe)
    # first_over = np.where(x_grid > ipoint)[0][0]
    first_over = np.where(x_grid > ipoint)[0]
    if len(first_over) > 0:
        first_over = first_over[0]
        s1 = -5/3
        b1 = pe_grid[first_over - 1] - s1 * x_grid[first_over - 1]
        fit_53 = np.polyval(np.array([s1, b1]), x_grid[0:first_over])
        s2 = -3
        b2 = pe_grid[first_over] - s2 * x_grid[first_over]
        fit_3 = np.polyval(np.array([s2, b2]), x_grid[first_over:])
        fit = np.concatenate((fit_53, fit_3))
        fit_back = np.interp(x, x_grid, fit)
        #  This is the target function that needs to be minimized
        fsq = ((10 ** fit_back) - (10 ** pe))**2

        return fsq.sum()
    else:
        return np.nan


