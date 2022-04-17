"""
cookb_signalsmooth.py

from: http://scipy.org/Cookbook/SignalSmooth
"""

import numpy as np

class UnmaskAndPad(object):
    ''' decorator class to preprocess arrays for smoothing '''
    
    def __init__(self, smoother):
        ''' store the smoothing operation we are going to apply '''
        self.smoother = smoother

    def __call__(self, data, pad_value=0, **kwargs):
        ''' unmask and pad data, execute smoother, and restore mask '''
        
        if not isinstance(data,np.ndarray):
            raise TypeError(data)
        
        # remove mask
        if isinstance(data, np.ma.masked_array):
            mask = data.mask; fill_value = data._fill_value
            data = data.filled(pad_value) # not actually inplace
        else:
            mask = None
        # remove NaN
        if np.issubdtype(data.dtype, np.inexact):
            nan_mask = np.isnan(data)
            data[nan_mask] = pad_value
            if np.isinf(data).any():
                raise NotImplementedError("Non-finite values except NaN are currently not handled in smoothing.")
        else:
            nan_mask = None
        
        # apply smoother
        data = self.smoother(data, **kwargs)
        
        # restore NaN
        if nan_mask is not None:
            data[nan_mask] = np.NaN
        # restore mask
        if mask is not None:
            data = np.ma.masked_array(data, mask=mask)
            data._fill_value = fill_value
            
        # return
        return data              
      

@UnmaskAndPad
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the appropriate size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be of equal size or bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[ 2*x[0]-x[window_len//2:0:-1], x, 2*x[-1]-x[-2:-(window_len//2)-2:-1] ]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len//2:-(window_len//2)]


#*********** part2: 2d

from scipy import signal

def twoDim_kern(size, window, sizey=None):
    """ Returns a normalized 2D kernel array for convolutions """    
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)    
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]    
    if window=='gauss':
        g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    elif window=='flat':
        g = np.ones((size,sizey))
    elif window=='hanning':
        g1d_x = np.hanning(size)
        g1d_y = np.hanning(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='hamming':
        g1d_x = np.hamming(size)
        g1d_y = np.hamming(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='bartlett':
        g1d_x = np.bartlett(size)
        g1d_y = np.bartlett(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='blackman':    
        g1d_x = np.blackman(size)
        g1d_y = np.blackman(sizey)        
        Temp = np.outer(g1d_x,g1d_y)
        Temp[np.abs(Temp)<1e-15] = 0        
        g = np.sqrt(Temp) 
        # NOTE: For the blackman window some elements have tiny negative values which
        #   become problematic when taking the square root. So I've added the above
        #   code to fix this.
    return g/g.sum()

@UnmaskAndPad
def smooth_image(im, window='gauss', n=10, ny=None):
    """ blurs the image by convolving with a kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    n = int(n)
    if not ny:
        ny = n
    else:
        ny = int(ny)
    g = twoDim_kern(size=n,window=window,sizey=ny) 
    [mx,my] = im.shape
    ox = 2*(n//2)+mx
    oy = 2*(ny//2)+my
    S = np.zeros((ox,oy))
    S[n//2:-(n//2),ny//2:-(ny//2)] = im
    for i in np.arange(n//2,ox-(n//2)):
        S[i,:] = np.r_[ 2*im[i-(n//2),0]-im[i-(n//2),ny//2:0:-1], im[i-(n//2),:], 
            2*im[i-(n//2),-1]-im[i-(n//2),-2:-(ny//2)-2:-1] ]
    for j in np.arange(ny//2,oy-(ny//2)):
        S[:,j] = np.r_[ 2*im[0,j-(ny//2)]-im[n//2:0:-1,j-(ny//2)], 
            im[:,j-(ny//2)], 2*im[-1,j-(ny//2)]-im[-2:-(n//2)-2:-1,j-(ny//2)] ]
    TL_1 = np.zeros((n//2,ny//2))
    TR_1 = np.zeros((n//2,ny//2))
    BL_1 = np.zeros((n//2,ny//2))
    BR_1 = np.zeros((n//2,ny//2))
    for i in np.arange(ox-(n//2),ox):
        TL_1[i-ox+(n//2),:] = 2*S[i,ny//2]-S[i,2*(ny//2):ny//2:-1] 
        TR_1[i-ox+(n//2),:] = 2*S[i,-1-(ny//2)]-S[i,-2-(ny//2):-2*(ny//2)-2:-1] 
    for i in np.arange(n//2):
        BL_1[i,:] = 2*S[i,ny//2]-S[i,2*(ny//2):ny//2:-1]
        BR_1[i,:] = 2*S[i,-1-(ny//2)]-S[i,-2-(ny//2):-2*(ny//2)-2:-1] 
    TL_2 = np.zeros((n//2,ny//2))
    TR_2 = np.zeros((n//2,ny//2))
    BL_2 = np.zeros((n//2,ny//2))
    BR_2 = np.zeros((n//2,ny//2))
    for j in np.arange(oy-(ny//2),oy):
        BR_2[:,j-oy+(ny//2)] = 2*S[n//2,j]-S[2*(n//2):n//2:-1,j] 
        TR_2[:,j-oy+(ny//2)] = 2*S[-1-(n//2),j]-S[-2-(n//2):-2*(n//2)-2:-1,j] 
    for j in np.arange(ny//2):
        BL_2[:,j] = 2*S[n//2,j]-S[2*(n//2):n//2:-1,j]
        TL_2[:,j] = 2*S[-1-(n//2),j]-S[-2-(n//2):-2*(n//2)-2:-1,j]
    S[0:n//2,0:ny//2] = (BL_1+BL_2)*0.5
    S[ox-(n//2):ox,0:ny//2] = (TL_1+TL_2)*0.5
    S[0:n//2,oy-(ny//2):oy] = (BR_1+BR_2)*0.5
    S[ox-(n//2):ox,oy-(ny//2):oy] =(TR_1+TR_2)*0.5
    improc = signal.convolve(S,g,mode='same')
    #print(np.sum(np.abs(TL_1-TL_2)>1e-12)+np.sum(np.abs(TR_1-TR_2)>1e-12)+
    #    np.sum(np.abs(BL_1-BL_2)>1e-12)+np.sum(np.abs(BR_1-BR_2)>1e-12))
    return(improc[n//2:-(n//2),ny//2:-(ny//2)])


def smooth_demo():
    import matplotlib.pyplot as plt

    t = np.linspace(-4,4,100)
    x = np.sin(t)
    xn = x + np.random.randn(len(t)) * 0.1
    y = smooth(x)
    ws = 31

    plt.subplot(211)
    plt.plot(np.ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    #plt.hold(True)
    for w in windows[1:]:
        #eval('plt.plot('+w+'(ws) )')
        plt.plot(getattr(np, w)(ws))

    plt.axis([0,30,0,1.1])

    plt.legend(windows)
    plt.title("The smoothing windows")
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(smooth(xn,window_len=10,window=w))
    l = ['original signal', 'signal with noise']
    l.extend(windows)
    plt.legend(l)
    plt.title("Smoothing a noisy signal")
    #plt.show()


def smooth_image_demo():
    import matplotlib.pyplot as plt
    
    windows=['gauss', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman']    
    
    X, Y = np.mgrid[-70:70, -70:70]
    Z = np.cos((X**2+Y**2)/200.)+ np.random.normal(size=X.shape)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(Z)
    plt.title("The perturbed signal")
    
    for w in windows:
        [n,ny] = Z.shape
        g = twoDim_kern(size=31,window=w)
        Z2 = smooth_image(Z,window=w,n=5)
        plt.figure()
        plt.subplot(121)
        plt.imshow(g) 
        plt.colorbar(orientation="horizontal")
        plt.title("Weight function "+w)
        plt.subplot(122)
        plt.imshow(Z2)  
        plt.colorbar(orientation="horizontal")
        plt.title("Smoothed using window "+w)


if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    # part 1: 1d
    smooth_demo()
    
    # part 2: 2d
    smooth_image_demo()
    
