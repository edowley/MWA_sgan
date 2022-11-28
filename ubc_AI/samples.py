import numpy as np

def normalize(data):
    '''data:input array of 1-3 dimentions
       to be normalized.
       Remember to return the normalized data. 
       The input will not be changed.
    '''
    if type(data) in [list]:
        result = []
        for a in data:
            result.append(normalize(a))
        return result
    else:
        if data.ndim > 1:
            N_row = data.shape[0]
            shape = data.shape
            return np.array([normalize(data[i,...]) for i in range(N_row)])
        else:
            mean = np.median(data)
            var = np.std(data)
            if var > 0:
                data = (data-mean)/var
            else:
                data = (data-mean)
        return data

from scipy import ndimage, array, mgrid

def downsample(a, n, align=0):
    '''a: input array of 1-3 dimensions
       n: downsample to n bins
       optional:
       align : if non-zero, downsample grid (coords) 
               will have a bin at same location as 'align'
               ( typically max(sum profile) )
               useful for plots vs. phase
         
    '''
    if type(a) in [list]:
        result = []
        for b in a:
            result.append(downsample(b))
        return result
    else:
        shape = a.shape
        D = len(shape)
        if D == 1:
            coords = mgrid[0:1-1./n:1j*n]
        elif D == 2:
            d1,d2 = shape
            if align: 
                #original phase bins
                x2 = mgrid[0:1.-1./d2:1j*d2]
                #downsampled phase bins
                crd = mgrid[0:1-1./n:1j*n]
                crd += x2[align]
                crd = (crd % 1)
                crd.sort()
                offset = crd[0]*d2
                coords = mgrid[0:d1-1:1j*n, offset:d2-float(d2)/n+offset:1j*n]
            else:
                coords = mgrid[0:d1-1:1j*n, 0:d2-1:1j*n]
        elif D == 3:
            d1,d2,d3 = shape
            coords = mgrid[0:d1-1:1j*n, 0:d2-1:1j*n, 0:d3-1:1j*n]
        else:
            raise "too many dimentions %s " % D
        def map_to_index(x,bounds,N):
            xmin, xmax= bounds
            return (x - xmin)/(xmax-xmin)*N
        if D == 1:
            m = len(a)
            x = mgrid[0:1-1./m:1j*m]
            if align:
                #ensure new grid lands on max(a)
                coords += x[align]
                coords = coords % 1
                coords.sort()
            return np.interp(coords, x, a)
        elif D == 2:
            newf = ndimage.map_coordinates(a, coords, cval=np.median(a))
            return newf
        else:
            newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)
            return newf