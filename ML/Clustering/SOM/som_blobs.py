from distutils.ccompiler import new_compiler
from json.encoder import py_encode_basestring
from re import I
from tkinter import W
from tkinter.ttk import tclobjs_to_py
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from sklearn import datasets
import SOM 

'''
    Ref:
        python example:  
            https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
        pybind11 example:


'''

# Return the (g,h) index of the BMU in the grid
def find_BMU(SOM,x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
    
# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, learn_rate, radius_sq, 
                   BMU_coord, step=3):
    g, h = BMU_coord
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step+1)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step+1)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])   
    return SOM    

# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, learn_rate = .1, radius_sq = 1, 
             lr_decay = .1, radius_decay = .1, epochs = 10):    
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in tqdm(range(0, epochs)):
        np.random.shuffle(train_data)      
        for train_ex in train_data:
            g, h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex, 
                                 learn_rate, radius_sq, (g,h))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)            
    return SOM

def compareRuntime(som_, x_, lr, lr_decay, radius, radius_decay, step, epochs):

    seconds_in_day = 24 * 60 * 60
    ''' python method'''
    print ("SOM by python start !!!")
    t_start = datetime.datetime.now()
    som_py  = train_SOM(som_, x_, learn_rate=lr, radius_sq=radius, lr_decay=lr_decay, radius_decay=radius_decay, epochs=epochs) 
    t_end   = datetime.datetime.now()
    difference = t_end - t_start
    print ("Python Runtime")
    print (divmod(difference.days * seconds_in_day + difference.seconds
                + difference.microseconds * 1e-6, 60))
    t_py = divmod(difference.days * seconds_in_day + difference.seconds
                + difference.microseconds * 1e-6, 60)
    ''' pybind11 method'''
    print ("SOM by pybind11 start!!!")
    t_start = datetime.datetime.now()
    _lr       = lr  + 0.
    _radius   = radius + 0.
    for ei in tqdm(range(epochs)):
        np.random.shuffle(x_)
        som_c = SOM.clustering(som_, x_, _lr, _radius, step)
        _lr   = lr * np.exp(-ei * lr_decay)
        _radius = radius * np.exp(-ei * radius_decay)            
    t_end   = datetime.datetime.now()
    difference = t_end - t_start
    print ("pybind11 Runtime")
    print (divmod(difference.days * seconds_in_day + difference.seconds    
                + difference.microseconds * 1e-6, 60))
    t_c  = divmod(difference.days * seconds_in_day + difference.seconds
                + difference.microseconds * 1e-6, 60)
    return som_py, som_c, t_py, t_c

def plotSOMContourf(som):
    n_dim   = som.shape[0]
    xx, yy  = np.mgrid[0:n_dim, 0:n_dim]
    n_col   = som.shape[-1]  
    fig, axs = plt.subplots(n_col)
    for ni in range(n_col):
        ss = axs[ni].contourf(xx, yy, som[...,ni])
        axs[ni].set_title(f"SOM dim{ni}")
        if ni == n_col - 1:
            plt.colorbar(ss)
    plt.show()

def plotSOM3D(som):
    from mpl_toolkits.mplot3d.axes3d import get_test_data
    from mpl_toolkits.mplot3d import Axes3D
    n_dim = som.shape[0]
    n_col = som.shape[-1]
    fig = plt.figure(figsize=plt.figaspect(1/float(n_col)))

    xx, yy = np.mgrid[0:n_dim, 0:n_dim]
    for ni in range(n_col):
        ax = fig.add_subplot(1, n_col, ni+1, projection="3d") 
        ax.plot_wireframe(xx, yy, som[...,ni])
        ax.set_title(f"SOM dim{ni}")
        ax.set_ylabel("M")
        ax.set_xlabel("N")
    plt.show()

def plotData(X, xlabels):
    n_center = int(np.max(xlabels) + 1)
    for ni in range(n_center):
        _x = X[xlabels == ni]
        plt.scatter(_x[:,0], _x[:,1])
    plt.legend([f"center{i}" for i in range(n_center)])
    plt.show()

def main():
    n_samples = 2000
    n_centers = 6
    n_features = 2
    X, xlabels = datasets.make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features)
    X /= np.max(np.abs(X))
     # Dimensions of the SOM grid
    m = 10#int(n_centers**2)
    n = 10#int(n_centers**2)
    c = X.shape[-1]
    # Number of training examples
    # Initialize the SOM randomly
    _SOM = np.random.randint(0, 255, (m, n, c)).astype(float) / 255
    _SOM *= 2
    _SOM -= 1
    first_time = datetime.datetime.now()
    epochs  = 100
    lr      = 0.05
    lr_decay = 0.1
    radius  = 1
    step    = 3 
    radius_decay = 0.1

    plotData(X, xlabels)
    som_py, som_c, t_py, t_c = compareRuntime(_SOM, X.copy(), lr, lr_decay, radius, radius_decay, step, epochs)
    print(f"Runtime ratio (python/pybind11) = {t_py[-1] / t_c[-1]}")
    # plot SOM contourf for each channel
    plotSOMContourf(som_c)
    # plot SOM 3D for each channel
    plotSOM3D(som_c)
    
    # plot center projected into SOM map (M x N) 
    som_dim = som_c.shape[0]
    xx, yy  = np.mgrid[0:som_dim, 0:som_dim] 
    fm = 2
    fn = 3
    fig, axs = plt.subplots(fm, fn)
    i_img = 0
    for ni in range(n_centers):
        ff = xlabels == ni
        _x = X[ff]
        ij = SOM.find_BMU_image(som_c, _x)
        ii = i_img // fn 
        jj = i_img %  fn 
        axs[ii, jj].contourf(xx, yy, ij)
        axs[ii, jj].set_title(f"center{ni}")
        i_img += 1
#        plt.contourf(xx, yy, som_c[...,0])
    plt.show()

    '''
    for epoch in range(epochs):
        print (f"[epoch{epoch}]: start !!!")
        som.clustering(SOM, data_mnist_flat[0,:], 0.001, 1, 1)
    ij = som.find_BMU_image(SOM, data_mnist_flat)
    import pdb; pdb.set_trace()
    later_time = datetime.datetime.now()
    seconds_in_day = 24 * 60 * 60
    difference = later_time - first_time
    print (divmod(difference.days * seconds_in_day + difference.seconds, 60))
    first_time = datetime.datetime.now()
    som_py = train_SOM(som_bk, train_data, learn_rate = .1, radius_sq = 1, epochs=1)
    later_time = datetime.datetime.now()
    seconds_in_day = 24 * 60 * 60
    difference = later_time - first_time
    print (divmod(difference.days * seconds_in_day + difference.seconds 
                + difference.microseconds * 1e-6, 60))
    import pdb; pdb.set_trace()
    # Display both the training matrix and the SOM grid
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 3.5), 
        subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(train_data.reshape(50, 60, 3))
    ax[0].title.set_text('Training Data')
    ax[1].imshow(SOM.astype(int))
    ax[1].title.set_text('Randomly Initialized SOM Grid')   
    plt.show()
    '''

if __name__ == "__main__":
    main()