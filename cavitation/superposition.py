# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 20:20:24 2016

@author: David
"""
import numpy as np
import sfs


def mirror_source(dim, ssp, g, nr, rc, pp):
    """Computes the position of 5 or 6 image sources for rectangular rooms.
    No wall is hitted twice in succession. The coordinate origin coincides with
    the center point of the cuboid and the surfaces are parallel to the planes
    of the coordiante system.

    Parameter
    ---------
    dim: (3,) cuboid dimension
        [0]: lenght, x-direction
        [1]: width, y-direction
        [2]: height, z-direction
    ssp: (3,) sound source position
        [0]: x-coordinate
        [1]: y-coordinate
        [2]: z-coordinate
    nr: number of the last reflecting wall, '0' no wall hitted
    g: gain factor
        takes the rc after multiple reflections into account
    rc: (6,) reflection coefficient of the 6 surfaces
        order: see Returns, last colmumn
    pp: propagation path

    Returns
    -------
    (5,6) or (6,6) ndarray
        first 3 columns: x, y, z coordinates of image sources
        4th column: gain factor, considering the reflection coefficients of
            all surfaces in the propagation path
        5th column: number of the last reflecting wall
            1: wall at x = -l/2
            2: wall at x = +l/2
            3: wall at y = -w/2
            4: wall at y = +w/2
            5: wall at z = -h/2
            6: wall at z = +h/2
        last column: propagation path
            sequence of the hitted surfaces

    """
    c = 0
    if nr == 0:
        sq = np.zeros((6, 6))
    else:
        sq = np.zeros((5, 6))
    if nr != 1:
        sq[c, :] = np.array([-dim[0]/2+(-dim[0]/2-ssp[0]), ssp[1], ssp[2],
                            g*rc[0], 1, pp*10+1])
        c = c+1
    if nr != 2:
        sq[c, :] = np.array([+dim[0]/2+(+dim[0]/2-ssp[0]), ssp[1], ssp[2],
                            g*rc[1], 2, pp*10+2])
        c += 1
    if nr != 3:
        sq[c, :] = np.array([ssp[0], -dim[1]/2+(-dim[1]/2-ssp[1]), ssp[2],
                            g*rc[2], 3, pp*10+3])
        c += 1
    if nr != 4:
        sq[c, :] = np.array([ssp[0], +dim[1]/2+(+dim[1]/2-ssp[1]), ssp[2],
                            g*rc[3], 4, pp*10+4])
        c += 1
    if nr != 5:
        sq[c, :] = np.array([ssp[0], ssp[1], -dim[2]/2+(-dim[2]/2-ssp[2]),
                            g*rc[4], 5, pp*10+5])
        c += 1
    if nr != 6:
        sq[c, :] = np.array([ssp[0], ssp[1], +dim[2]/2+(+dim[2]/2-ssp[2]),
                            g*rc[5], 6, pp*10+6])
        c += 1
    return np.around(sq, 5)  # rounding to avoid numerical problems


def number_of_sources(n):
    """Computes the sum of the first n elements of a geometric progression with
    a_0 = 6, q = 5
    Formelsammlung höhere Mathematik, W. Göhler, edition 17, p. 32

    """
    a_0 = 6
    q = 5
    s_n = a_0*(q**n-1)/(q-1)
    return(int(s_n))


def delete_sources(image_sources):
    """Deletes image source positions which have a gain factor of 0

    Parameter
    ---------
    image_sources: array that contains the image source positions
        4th row corresponds to the gain factors

    Returns
    -------
    ndarray with image sources positions

    """
    index = np.where(image_sources[:, 3] == 0.0)
    active_sources = np.delete(image_sources, index, 0)
    return(active_sources)


def unique_sources(image_sources):
    """Deletes image source positions which have a gain factor of 0 and
    position which are present multiple times.

    Parameter
    ---------
    image_sources: array that contains the image source positions
        4th row corresponds to the gain factors

    Returns
    -------
    ndarray with image sources positions

    """
    index = np.where(image_sources[:, 3] == 0.0)  # search for inactive sources
    active_sources = np.delete(image_sources, index, 0)

    sorted_id = np.lexsort(np.transpose(active_sources[:, :3]), axis=0)
    # sorting active sources by the values of their coordinates, starting with
    # z-coordinate
    # sources with the same coordinate are listed one below the other
    sorted_sources = active_sources[sorted_id, :]
    # search for adjacent source positions that have the same coordinates
    index = np.where(np.any(np.diff(sorted_sources[:, :3], axis=0), axis=1) ==
                     False)
	 # shifting by 1 to delete the correct row, see np.diff()
    index = np.array(index)+1
    unique_sources = np.delete(sorted_sources, index, axis=0)
    resorted_id = np.lexsort(np.transpose(unique_sources[:, :]), axis=-1)
    resorted_sources = unique_sources[resorted_id, :]
    return resorted_sources


def image_sources(dim, xs, order, rc):
    """Computes image sources by mirroring up to the given order for a cuboid.
    The coordinate origin coincides with the center point of the cuboid and the
    surfaces are parallel to the planes of the coordinate system.
    The number of image sources depends on the order.

    Parameter
    ---------
    dim: (3,) cuboid dimension
        [0]: lenght, x-direction
        [1]: width, y-direction
        [2]: height, z-direction
    xs: (3,) sound source position in cartesian coordinates
    order: determined number of image source
    rc: (6,) reflection coefficient of the 6 surfaces
        sequence consequential Returns

    Returns
    -------
    (number_of_sources(order)+1, 6) ndarray
        first 3 columns: x, y, z coordinates of image sources
        4th column: gain factor, considering the reflection coefficients of
            all surfaces in the propagation path
        5th columns: number of the last reflecting wall
            0: no wall hitted
            1: wall at x = -l/2
            2: wall at x = +l/2
            3: wall at y = -w/2
            4: wall at y = +w/2
            5: wall at z = -h/2
            6: wall at z = +h/2
        last column: propagation path
            sequence of the number of the hitted surfaces
        first row: coordinates of the sound source

    """
    sources = np.zeros((number_of_sources(order)+1, 6))
    """gain factor of sound source = 1
    number of the last hitted wall = 0
    propagation path = 0, because 0 wall hitted"""
    sources[0, :] = [xs[0], xs[1], xs[2], 1, 0, 0]

    c = 0  # counter to iterate
    r = 1  # variable to write data in the corresponding row
    while c <= number_of_sources(order - 1):
        sq = mirror_source(dim, [sources[c, 0], sources[c, 1],
                           sources[c, 2]], sources[c, 3], sources[c, 4], rc,
                           sources[c, 5])
        sources[r:r+sq.shape[0], :] = sq
        c += 1
        r += sq.shape[0]
    return(sources)


def superpose_point_sources(xs, signal, dt, grid, fs=None, c=None,
                            kind='linear'):
    """Computes the soundfield of several point sources by superposition.

    Parameter
    ---------
    xs: (n,4) ndarray
        position of the point sources
        n: number of sources
        4:  cartesian coordinates in order x, y, z and gain factor
    signal : (N,) array_like
        Excitation signal.
    dt: float
        observed period of time
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    fs: int, optional
        Sampling frequency in Hertz.
    c : float, optional
        Speed of sound.
    kind: str
        kind of interpolation used, default is linear

    Returns
    -------
    ndarray
        Scalar sound pressure field evaluated at positions given by *grid*.
        row: soundfield at a discrete time
        column: soundfield at a discrete point

    """
    xs = xs[:, :4]
    signal = sfs.util.asarray_1d(signal)
    nt = int(dt*fs+1)  # required samples for given time period
    t = np.linspace(0, dt, nt)
    if fs is None:
        fs = sfs.defs.fs
    if c is None:
        c = sfs.defs.c
    p = np.zeros((nt, np.shape(grid[0])[0], np.shape(xs)[0]))
    c1 = 0  # counter variable
    while c1 <= np.shape(xs)[0]-1:
        c2 = 0  # counter variable
        while c2 <= nt - 1:
            p[c2, :, c1] = sfs.time.source.point(xs[c1, :3], xs[c1, 3]*signal,
                                                 t[c2], grid, fs, c, kind)
            c2 += 1
        c1 += 1
    # superposition of the sound fields
    return (np.sum(p, axis=2))


def compute_max_distance(xs, h):
    """Computes the distance between points. xs and h are arrays that contain
    several points. Therefore every row of xs is replicated n_h times and every
    column of h is replicated n_xs times.

    Parameter
    ---------
    xs: (n_xs, 3) array
        n_xs: number of points
        3: cartesian ccordinates
    h: (n_h, 3) array
        see above

    Returns
    -------
    (n_xs*n_h,) array

    """
    h1 = np.repeat(xs[:], np.shape(h)[0], axis=0)
    h2 = np.repeat([h], np.shape(xs)[0], axis=0)
    h2 = np.reshape(h2, (np.shape(xs)[0]*np.shape(h)[0], 3))
    return max(np.linalg.norm(h1-h2, axis=1))
