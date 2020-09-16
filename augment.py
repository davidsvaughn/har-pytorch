import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

'''####################################################################'''

def checknan(x):
    if (x != x).any():
        idx = np.where(np.isnan(x))[0]
        print(idx)
        raise ValueError('NaN encountered!')
    
def normalize(X, axis=0):
    Xh = X.max(axis) - X.min(axis)
    return (X - X.min(axis))/Xh
    
def scale_like(X, Y, axis=0):
    Xn = normalize(X, axis)
    Yh = Y.max(axis) - Y.min(axis)
    Xs = Xn * Yh + Y.min(axis)
    if (Xs != Xs).any():
        return X
    return Xs

# magnitude vector
def mag(X, axis=1):
    return np.sqrt(np.sum(np.square(X), axis=axis))

## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(dim, sigma=0.2, knot=4, tied=True, rng=None):
    rng = np.random if rng is None else rng
    xx = (np.ones((dim[1],1))*(np.arange(0,dim[0], (dim[0]-1)/(knot+1)))).transpose()
    yy = rng.normal(loc=1.0, scale=sigma, size=(knot+2, dim[1]))
    x_range = np.arange(dim[0])
    cs0 = CubicSpline(xx[:,0], yy[:,0])
    css = []
    for i in range(dim[1]):
        if tied:
            cs = cs0
        else:
            cs = CubicSpline(xx[:,i], yy[:,i])
        css.append(cs(x_range))
    return np.array(css).transpose()


''' TIME WARPING '''
# #### Hyperparameters :  sigma = STD of the random knots for generating curves
# #### knot = # of knots for the random curves (complexity of the curves)

sigma = 0.1
kf = 100 ## knot frequency

def DistortTimesteps(dim, sigma=0.2, kf=100, tied=True, rng=None):
    tt = GenerateRandomCurves(dim, sigma, dim[0]//kf, tied, rng=rng) # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(dim[0]-1)/tt_cum[-1,0],(dim[0]-1)/tt_cum[-1,1],(dim[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def apply_timewarp(X, tt_new):
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
    ## rescale ???
    X_new = scale_like(X_new, X)
    return X_new

def DA_TimeWarp(X, sigma=0.2, kf=100, tied=True, rng=None):
    x = X[0] if isinstance(X, (list, tuple)) else X
    tt_new = DistortTimesteps(x.shape, sigma=sigma, kf=kf, tied=tied, rng=rng)
    Xw = [apply_timewarp(x, tt_new) for x in X] if isinstance(X, (list, tuple)) else apply_timewarp(X, tt_new)
    return Xw


''' ROTATION '''

def apply_rotation(X, R):
    Xr = np.matmul(X, R)
    ## clip to [-1,1]
    t = np.array([1.,1.,1.])
    Xr = np.clip(Xr, -t, t)
    return Xr
    
def DA_Rotation(X, rng=None, f=1.0):
    x = X[0] if isinstance(X, (list, tuple)) else X
    rng = np.random if rng is None else rng
    axis = rng.uniform(low=-1, high=1, size=x.shape[1])
    angle = rng.uniform(low=-np.pi*f, high=np.pi*f)
    R = axangle2mat(axis, angle)
    Xr = [apply_rotation(x, R) for x in X] if isinstance(X, (list, tuple)) else apply_rotation(X, R)
    return Xr
    
'''####################################################################'''

''' TESTING '''

if __name__ == "__main__":

    A = np.load('fidget/A_sample.npy')
    G = np.load('fidget/G_sample.npy')
    
    X = A
    
    #X = X[:100]
    
    print(X.shape)
    N = X.shape[0]
    
    plt.figure(figsize=(10,3))
    plt.plot(A)
    plt.title("An example of 3-axis accel data")
    plt.axis([0,N,-1.5,1.5])
    plt.show()
    
    plt.figure(figsize=(10,3))
    plt.plot(G)
    plt.title("An example of 3-axis gyro data")
    plt.axis([0,N,-1.5,1.5])
    plt.show()
    
    '''
    print('\nPERM + ROT')
    ## Rotation + Permutation
    fig = plt.figure(figsize=(15,4))
    for ii in range(8):
        ax = fig.add_subplot(2,4,ii+1)
        ax.plot(DA_Rotation(DA_Permutation(X, nPerm=4)))
        ax.set_xlim([0,N])
        ax.set_ylim([-1.5,1.5])
    
    plt.show()
    '''
    
    print('\nTIME-WARP + ROTATION')
    ## Rotation + Permutation
    
    sigma = 0.2
    kf = 50
    
    ## Random curves around 1.0
    fig = plt.figure(figsize=(10,10))
    for ii in range(9):
        ax = fig.add_subplot(3,3,ii+1)
        ax.plot(DistortTimesteps(X.shape, sigma, kf))
        ax.set_xlim([0,N])
        ax.set_ylim([0,N])
        
    ##sys.exit()
    
    fig = plt.figure(figsize=(10,15))
    for ii in range(8):
        ax = fig.add_subplot(8,1,ii+1)
        a,g = A,G
        a,g = DA_TimeWarp([a,g], sigma, kf)
        #a,g = DA_Rotation([a,g])
        S = a
        #print([S.min(0), S.max(0)])
        #print((mag(S)-mag(X)).mean())
        ax.plot(S)
        ax.set_xlim([0,N])
        ax.set_ylim([-1.5,1.5])
    
    plt.show()
    
    ################################

## experiment with sigma, knots....
'''

sigma, kf = 0.2, 50
N = 500
dim = (N,3)
t = DistortTimesteps(dim, sigma, kf)[:,0]
plt.plot(t)
## max distortion rate...
print(abs(np.diff(t)-1).max())
## max distortion...
s = np.linspace(1,N-1,N)
print(abs(s-t).max())


'''
