import numpy as np
import sfs


def tdoa_gcc(p, fs=None):
    """Computes the Time Differences of Arrival between several sensors
    by using cross-correlation. The reference is in the first column of p.

    Parameter
    ---------
    p: (s, n_r) ndarray - sound field
        s: number of samples, observed time period
        n_r: number of receivers
    fs: int, optional
        Sampling frequency in Hertz.

    Returns
    -------
    (n_r-1,) ndarray
        time difference of arrival in s between the receivers

    """
    if fs is None:
        fs = sfs.defs.fs
    cc = np.empty((2*np.shape(p)[0]-1, np.shape(p)[1]-1))
    c1 = 0  # counter variable
    while c1 <= np.shape(p)[1]-2:
        cc[:, c1] = np.correlate(p[:, c1+1], p[:, 0], 'full')
        c1 += 1
    tdoa = (np.argmax(cc, axis=0)-(np.shape(p)[0]-1))/fs
    return tdoa


def tdoa_phat(p, fs=None):
    """Computes the Time Differences of Arrival using the Generalized Cross-
    Correlation Phase Transform. Reference is the first column.

    Parameter
    ---------
    p: (s, n_r) ndarray - sound field
        s: number of samples, observed time period
        n_r: number of receivers
    fs: int, optional
        Sampling frequency in Hertz.

    Returns
    -------
    (n_r-1,) ndarray
        time difference of arrival in s between the receivers

    """
    if fs is None:
        fs = sfs.defs.fs
    L = int(np.ceil(np.log2(2*p.shape[0]-1)))
    N = np.power(2, L)
    #N = 2*p.shape[0]-1
    M = p.shape[1]-1
    # replicate FFT of reference sensor's signal
    G = np.array([np.fft.fft(p[:, 0], N)]*M)
    G = np.transpose(G)
    F = np.fft.fft(p[:, 1:], N, axis=0)
    Y = F*np.conj(G)
    w = np.abs(Y)
    w[w < 1e-9] = 1e-9  # prevent division by zero
    w = 1/w  # weighting
    Y = Y*w
    y = np.fft.ifft(Y, axis=0)
    gcc_phat = np.real(np.fft.fftshift(y, axes=0))
    #tdoa = (np.argmax(gcc_phat, axis=0)-(p.shape[0]-1))/fs
    tdoa = (np.argmax(gcc_phat, axis=0)-(N/2))/fs
    return tdoa


def multilateration(xr, tdoa, c=None):
    """Computes the analytic solution to the multilateration equations.

    Parameter
    ---------
    xr: (n_xr, 3) array_like - receiver positions in cartesian coordiantes
        n_xr: number of receivers
    tdoa: (n_xr-1,) array_like
        time differences of arrival
        receiver in first row of xr is the reference
    c: float - sound velocity

    Returns
    -------
    (3,1) array - calculated sound source position
    """
    if c is None:
        c = sfs.defs.c
    rd = tdoa*c  # range difference
    r = np.linalg.norm(xr, axis=1)**2
    A = np.empty((np.shape(xr)[0]-1, 4))
    B = np.empty((np.shape(xr)[0]-1, 1))
    c1 = 0  # counter variable
    while c1 < np.shape(xr)[0]-1:
        A[c1, :] = np.array([xr[c1+1, 0] - xr[0, 0], xr[c1+1, 1] - xr[0, 1],
                             xr[c1+1, 2] - xr[0, 2], rd[c1]])
        B[c1, 0] = 0.5*(r[c1+1] - rd[c1]**2 - r[0])
        c1 += 1
    A = np.matrix(A)
    inv_A = np.linalg.pinv(A)
    B = np.matrix(B)
    xs = inv_A*B
    """delete last column, because no use of distance between source and
    reference sensor"""
    xs = xs[:-1]
    return np.asarray(xs)


def multilateration_processing(xr, tdoa_shift, fs=None, c=None):
    """Computes a potential sound source location by multilateration. Every
    receiver position is used as reference receiver.

    Parameter
    ---------
    xr: (n_xr, 3) array - cartesian coordinates of the receivers
        n_xr: number of receivers
    tdoa_shift: (n_xr, n_xr - 1) array - every xr was used as reference
    fs: int - sampling frequency in Hz
    c: float - sound velocity

    Returns
    -------
    (n_xr, 3) sound source location estimate
    """
    if fs is None:
        fs = sfs.defs.fs
    if c is None:
        c = sfs.defs.c
    xr_shift = np.empty([xr.shape[0], xr.shape[1]])
    #p_shift = np.empty([p_xr.shape[0], p_xr.shape[1]])
    xs = np.empty((xr.shape[0], 3))  # 3 estimates
    c1 = 0
    while c1 < xr.shape[0]:
        xr_shift = np.concatenate((xr[c1:, :], xr[:c1, :]), axis=0)
        #p_shift = np.concatenate((p_xr[:, c1:], p_xr[:, :c1]), axis=1)
        #tdoa_shift[c1, :] = tdoa_phat(p_shift, fs)
        xs[c1, :] = np.reshape(multilateration(xr_shift, tdoa_shift[c1, :], c),
                               (3,))
        c1 += 1
    return xs


def straightray(xr, tdoa):
    """Sound source location, sound velocity and time of flight from sound
    source to reference receiver estimation by straight ray processing. Needs
    at least 6 receiver positions xr and 5 tdoa.

    Parameter
    ---------
    xr: (n_xr, 3) array - receivers cartesian coordinates
        n_xr: number of receivers
    tdoa: (5, ) array_like - time differences of arrival
        xr[0, :] is reference

    Returns
    -------
    (5,1) array - estimates of cartesian coordinates of source location, sound
    velocity and time of flight
    """
    xr = xr[:, :3]
    # squared absolute of a position vector
    r = np.linalg.norm(xr[:, :3], axis=1)
    B = 0.5*(r[1:]**2-r[0]**2)
    B = np.matrix(np.reshape(B, (xr.shape[0]-1, 1)))
    A = xr[1:, :]-xr[0, :]
    A = np.concatenate((A, np.reshape(tdoa**2, (tdoa.shape[0], 1)),
                        np.reshape(tdoa, (tdoa.shape[0], 1))), axis=1)
    inv_A = np.linalg.pinv(A)
    xs = inv_A*B
    #xs[3] = np.sqrt(2*xs[3])  # sound velocity
    #xs[4] = xs[4]/xs[3]**2  # reference time of flight
    return np.asarray(xs)


def straightray_processing(xr, tdoa_shift, fs=None, c=None):
    """Computes the potential location of a sound source by using straight ray
    processing. every receiver position xr is used as reference reciever.
    literature: Ray-based acoustic localization of cavitation in a highly re-
    verberant environment, N. A. Chang and D. R. Dowling, The Journal of the
    Acoustical Society of America, Vol. 125, June 2009

    Parameter
    ---------
    xr: (n_xr, 3) array - cartesian coordinates of the receivers
    tdoa_shift: (n_xr, n_xr - 1) array - every xr was used as reference
        n_xr: number of receivers
    fs: int - sampling frequency in Hz
    c: float - sound velocity

    Returns
    -------
    xs: (n_xr, 3) array - estimates of 3 cartesian coordinates
    """
    if fs is None:
        fs = sfs.defs.fs
    if c is None:
        c = sfs.defs.c
    x = np.empty((xr.shape[0], 5))  # 5 estimates
    """to use every receiver as reference xr, p_xr and tdoa have to be shifted,
    because first column is the reference"""
    xr_shift = np.empty([xr.shape[0], xr.shape[1]])
    #tdoa_shift = np.empty((xr.shape[0], xr.shape[0]-1))
    #p_shift = np.empty([p_xr.shape[0], p_xr.shape[1]])

    c1 = 0  # counter
    while c1 < xr.shape[0]:
        xr_shift = np.concatenate((xr[c1:, :], xr[:c1, :]), axis=0)
        #p_shift = np.concatenate((p_xr[:, c1:], p_xr[:, :c1]), axis=1)
        #tdoa_shift[c1, :] = tdoa_phat(p_shift, fs)
        x[c1, :] = np.reshape(straightray(xr_shift, tdoa_shift[c1, :]), (5,))
        c1 += 1
    # delete last column because no use of time of flight to reference
    x = x[:, :-2]
    return x


def SX(xr, tdoa, c=None):
    """Sound source location estimation by spherical intersection. Needs at
    least 4 receiver positions and 3 tdoa.

    Parameter
    ---------
    xr: (n_xr, 3) array- receivers cartesian coordinates
        n_xr: number of receivers
    tdoa: (n-xr-1, ) array_like - time differences of arrival
        reference is xr[0, :]
    c: float - sound velocity

    Returns
    -------
    (3,1) array - sound source location estimate
    """
    if c is None:
        c = sfs.defs.c
    xr = xr[:, :3]
    xr_lc = xr[1:, :] - xr[0, :]  # locale coordinate system
    xr_len = np.linalg.norm(xr_lc, axis=1)  # length of position vector
    rd = tdoa*c  # range differences
    delta = np.matrix(np.reshape(xr_len**2-rd**2, (xr.shape[0]-1, 1)))
    delta_T = np.transpose(delta)
    d = np.matrix(np.reshape(rd, (tdoa.shape[0], 1)))
    d_T = np.transpose(d)
    S = np.matrix(xr_lc)
    SW = np.linalg.pinv(S)
    SW_T = np.transpose(SW)
    # equation coefficients
    ac = 4 - 4*d_T*SW_T*SW*d
    bc = 4*d_T*SW_T*SW*delta
    cc = -delta_T*SW_T*SW*delta
    dc = bc**2-4*ac*cc
    if dc == 0:
        Rs = float(-bc/(2*ac))
        xs = np.reshape(0.5*SW*(delta-2*Rs*d), (1, 3))
    elif dc > 0:
        Rs1 = float((-bc+np.sqrt(dc))/(2*ac))
        Rs2 = float((-bc-np.sqrt(dc))/(2*ac))
        xs1 = 0.5*SW*(delta-2*Rs1*d)
        xs2 = 0.5*SW*(delta-2*Rs2*d)
        xs = np.reshape([xs1, xs2], (2, 3))
    else:
        return np.ones((2,3))*np.nan
        #raise ValueError('no positive sphere radius')
    xs = xs + xr[0, :]
    return np.asarray(xs)


def SX_processing(xr, tdoa_shift, fs=None, c=None):
    """Sound source location estimation by spherical intersection. Needs at
    least 4 receiver positions xr and 3 tdoa. Every receiver is used as
    reference.
    literature: Passive Source Localization Employing Intersecting Spherical
    Surfaces from Time-of-Arrival Differences, H. C. Schau and A. Z. Robinson,
    IEEE TRANSACTIONS ON ACOUSTICS, SPEECH, AND SIGNAL PROCESSING, VOL. 35,
    NO. 8, AUGUST 1987

    Parameter
    ---------
    xr: (n_xr, 3) array - cartesian coordinates of the receivers
    tdoa_shift: (n_xr, n_xr - 1) array - every xr was used as reference
    fs: int - sampling frequency in Hz
    c: float - sound velocity

    Returns
    -------
    (n_xs,1) array - sound source location estimate
        0 < n_xs < 2*n_xr
    """
    if fs is None:
        fs = sfs.defs.fs
    if c is None:
        c = sfs.defs.c
    xs = np.empty((2*xr.shape[0], 3))
    #tdoa = np.empty(xr.shape[0]-1)
    c1 = 0
    c2 = 0  # counter number of returned sources
    while c1 < xr.shape[0]:
        #p_shift = np.concatenate((p_xr[:, c1:], p_xr[:, :c1]), axis=1)
        xr_shift = np.concatenate((xr[c1:, :], xr[:c1, :]), axis=0)
        #tdoa = tdoa_phat(p_shift, fs)
        interim = SX(xr_shift, tdoa_shift[c1, :], c)
        if interim.shape[0] == 1:
            xs[c2, :] = interim
            c2 += 1
        elif interim.shape[0] == 2:
            xs[c2:c2+2, :] = interim
            c2 += 2
        c1 += 1
    return np.asarray(xs[:c2, :])


def SI(xr, tdoa, c=None):
    """Sound source location estimation by spherical interpolation. Needs at
    least 4 receiver positions xr and 3 tdoa.
    literature: Closed-Form Least Squares Source Location Estimation from Range
    -Differences Measurements, J. O. Smith and J. S. Abel, IEEE TRANSACTIONS ON
    ACOUSTICS, SPEECH, AND SIGNAL PROCESSING, VOL. 35, NO. 8, AUGUST 1987

    Parameter
    ---------
    xr: (n_xr, 3) array- receivers cartesian coordinates
        n_xr: number of receivers
    tdoa: (n-xr-1, ) array_like - time differences of arrival
        reference is xr[0, :]
    c: float - sound velocity

    Returns
    -------
    (3,1) array - sound source location estimate
    """
    if c is None:
        c = sfs.defs.c
    xr = xr[:, :3]
    xr_lc = xr[1:, :] - xr[0, :]  # local coordinate system
    xr_len = np.linalg.norm(xr_lc, axis=1)  # length of position vector
    rd = tdoa*c  # range differences
    delta = np.matrix(np.reshape(xr_len**2-rd**2, (xr.shape[0]-1, 1)))
    d = np.matrix(np.reshape(rd, (tdoa.shape[0], 1)))
    d_T = np.transpose(d)
    S = np.matrix(xr_lc)
    SW = np.linalg.pinv(S)
    PS = S*SW
    PSO = np.identity(xr.shape[0]-1)-PS
    Rs = float((d_T*PSO*delta)/(2*d_T*PSO*d))
    if Rs < 0:
        return np.asarray([[np.nan]*3])
        #raise ValueError('negative sphere radius')
    xs = np.array(0.5*SW*(delta-2*Rs*d))
    xs = xs.T + xr[0, :]  # globale coordinate system
    return np.asarray(xs.T)


def SI_processing(xr, tdoa_shift, fs=None, c=None):
    """Sound source location estimation by spherical interpolation. Every
    receiver position is used reference.
    literature: Closed-Form Least Squares Source Location Estimation from Range
    -Differences Measurements, J. O. Smith and J. S. Abel, IEEE TRANSACTIONS ON
    ACOUSTICS, SPEECH, AND SIGNAL PROCESSING, VOL. 35, NO. 8, AUGUST 1987

    Parameter
    ---------
    xr: (n_xr, 3) array- receivers cartesian coordinates
        n_xr: number of receivers
    tdoa_shift: (n_xr, n_xr - 1) array - every xr was used as reference
    dim: (6,) array_like - limits of the cuboid
        [x_min, x_max, y_min, ymax, z_min, z_max]
    fs: int - sampling frequency in Hz
    c: float - sound velocity

    Returns
    -------
    (n_xr,1) array - sound source location estimate
    """
    if c is None:
        c = sfs.defs.c
    if fs is None:
        fs = sfs.defs.fs
    xs = np.empty((xr.shape[0], 3))
    #tdoa = np.empty((xr.shape[0], xr.shape[0]-1))
    c1 = 0
    while c1 < xr.shape[0]:
        #p_shift = np.concatenate((p_xr[:, c1:], p_xr[:, :c1]), axis=1)
        xr_shift = np.concatenate((xr[c1:, :], xr[:c1, :]), axis=0)
        #tdoa[c1, :] = tdoa_phat(p_shift, fs)
        xs[c1, :] = np.reshape(SI(xr_shift, tdoa_shift[c1, :], c), (3, ))
        c1 += 1
    return xs


def delete_outlier(xs, dim=None):
    """Deletes the values of xs which don't lie in the cuboid.
    length = x_max - x_min
    width = y_max - y_min
    height = z_max- z_min

    Parameter
    ---------
    xs: (n_xs, 3) array - potential sound source positions
        n_xs - number of potential sound sources
        3 - catresian coordinates
    dim: (6,) array_like - limits of the cuboid
        [x_min, x_max, y_min, ymax, z_min, z_max]

    Returns
    -------
    xs_within: array - sound sources within the cuboid
    """
    if dim is None:
        dim = [-1.3, 1.3, -0.3, 0.3, -0.3, 0.3]
    xs_within = xs[(xs[:, 0] > dim[0]) & (xs[:, 0] < dim[1])]
    xs_within = xs_within[(xs_within[:, 1] > dim[2]) &
                          (xs_within[:, 1] < dim[3])]
    xs_within = xs_within[(xs_within[:, 2] > dim[4]) &
                          (xs_within[:, 2] < dim[5])]
    return xs_within
