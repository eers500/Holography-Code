#%% rgb2gray
def rgb2gray(img):
    ## Convert rgb image to grayscale using Y' = 0.299R'+0.587G' + 0.114B'
    # Input:     img - RBG image
    # Output: img_gs - Grayscale image
    import numpy as np
    [ni, nj, nk] = img.shape
    img_gs = np.empty([ni, nj])
    for ii in range(0, ni):
        for jj in range(0, nj):
            img_gs[ii, jj] = 0.299 * img[ii, jj, 0] + 0.587 * img[ii, jj, 1] + 0.114 * img[ii, jj, 2]

    return img_gs


#%% square_image
def square_image(img):
    ## Make image square by adding rows or columns of the mean value of the image np.mean(img)
    # Input: img - grayscale image
    # Output: imgs - square image
    #         axis - axis where data is added
    #            d - number of rows/columns added
    import numpy as np

    [ni, nj] = img.shape
    dn = ni - nj
    d = abs(dn)
    if dn < 0:
        M = np.flip(img[ni - abs(dn):ni, :], 0)
        imgs = np.concatenate((img, M), axis=0)
        axis = 'i'
    elif dn > 0:
        M = np.flip(img[:, nj - abs(dn):nj], 1)
        imgs = np.concatenate((img, M), axis=1)
        axis = 'j'
    elif dn == 0:
        imgs = img
        axis = 'square'
    return imgs, axis, d


#%% bandpassFilter
def bandpassFilter(img, xs, xl):
    ## Bandpass filter
    # Input: img - Grayscale image array (2D)
    #        xl  - Large cutoff size (Pixels)
    #        xs  - Small cutoff size (Pixels)
    # Output: img_filt - filtered image
    import numpy as np

    # FFT the grayscale image
    imgfft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(imgfft)
    img_amp = abs(img_fft)
    del imgfft

    # Pre filter image information
    [ni, nj] = img_amp.shape
    MIS = ni

    # Create bandpass filter when BigAxis ==
    # LCO = np.empty([ni, nj])
    # SCO = np.empty([ni, nj])

    # for ii in range(ni):
    #     for jj in range(nj):
    #         LCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xl / MIS) ** 2)
    #         SCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xs / MIS) ** 2)
    # BP = SCO - LCO

    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni))
    
    LCO = np.exp(-((ii-MIS/2)**2 + (jj-MIS/2)**2) * (2*xl/MIS)**2)
    SCO = np.exp(-((ii-MIS/2)**2 + (jj-MIS/2)**2) * (2*xs/MIS)**2)
    BP =  SCO - LCO
    BPP = np.fft.ifftshift(BP)
    
    # Filter image
    filtered = BP * img_fft
    img_filt = np.fft.ifftshift(filtered)
    img_filt = np.fft.ifft2(img_filt)
    # img_filt = np.rot90(np.real(img_filt),2)

    return img_filt, BPP


#%% videoImport
def videoImport(video, N):
    ## Import video as stack of images in a 3D array
    #   Input:  video   - path to video file
    #               N   - frame number to import
    #   Output: imStack - 3D array of stacked images in 8-bit
    import cv2
    import numpy as np

    CAP = cv2.VideoCapture(video)
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('uint8'))
    # IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))

    I = 0
    SUCCESS = True

    if N == 0:
        # IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype='float16')
        IMG = np.empty((HEIGHT, WIDTH, 3))
        IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH), dtype='float32')

        while (I < NUM_FRAMES and SUCCESS):
            SUCCESS, IMG = CAP.read()
            # IM_STACK[I] = IMG[I, :, :, 1]
            IM_STACK[I] = IMG[:, :, 0]
            I += 1
            # print(('VI', I))

    elif N > 0:
        IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype='float32')
        IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))
        STACK = IM_STACK

        while (I < NUM_FRAMES and SUCCESS):
            SUCCESS, IMG[I] = CAP.read()
            STACK[I] = IMG[I, :, :, 1]
            if I == N:
                IM_STACK = IMG[I, :, :, 1]
                FRAMENUM = I
                print(('VI', I))
            I += 1
    CAP.release()

    if N == 0:
        IM_STACK = np.swapaxes(np.swapaxes(IM_STACK, 0, 2), 0, 1)

    return IM_STACK


#%% exportAVI
def exportAVI(filename, IM, NI, NJ, fps):
    ## Export 3D array to .AVI movie file
    #   Input:  IM - numpy 3D array
    #           NI - number of rows of array
    #           NJ - number of columns of array
    #          fps - frames per second of output file
    #   Output: .AVI file in working folder
    import os
    from cv2 import VideoWriter, VideoWriter_fourcc
    dir = os.getcwd()
    filenames = os.path.join(dir, filename)
    FOURCC = VideoWriter_fourcc(*'MJPG')
    VIDEO = VideoWriter(filenames, FOURCC, float(fps), (NJ, NI), 0)

    for i in range(IM.shape[2]):
        frame = IM[:, :, i]
        frame = np.uint8(255*frame/frame.max())
        #    frame = np.random.randint(0, 255, (NI, NJ,3)).astype('uint8')
        VIDEO.write(frame)

    VIDEO.release()

    print(filename, 'exported successfully')
    return


#%% rayleighSommerfeldPropagator
def rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS):
    ## Rayleigh-Sommerfeld Back Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        IMM - 3D array representing stack of images at different Z
    import numpy as np
    from functions import bandpassFilter

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN
    
    #    IN = I - I_MEDIAN,
    #     IN[IN < 0] = 0

    # Bandpass Filter
    _, BP = bandpassFilter(IN, 2, 30)
    E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameters     #Set as input parameters
    # N = 1.3226               # Index of refraction
    LAMBDA = LAMBDA       # HeNe
    FS = FS               # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    # SZ = 10
    Z = SZ*np.arange(1, NUMSTEPS)
    # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
    #    Z = SZ*np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    #P = np.empty_like(I_MEDIAN, dtype='complex64')

    #for i in range(NI):
    #    for j in range(NJ):
    #        P[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)

    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
    P = const*((ii-NI/2)**2 + (jj-NJ/2)**2)

    # P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z > 0):
        Q = np.conj(Q)

    # R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
    IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
        IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))
    #        print(('RS', k))
    return IZ


#%% medianImage
def medianImage(VID, numFrames):
    ## Median Image
    #   Input:   VID - 3D numpy array of video file
    #            numFrames - Number of frames to calculat median image
    #   Output: MEAN - 2D pixel mean array
    import numpy as np

    def spaced_elements(array, numElems):
        out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
        return out

    N = np.shape(VID)[2]
    id = spaced_elements(np.arange(N), numFrames)

    print('MI')
    MEAN = np.median(VID[:, :, id], axis=2)

    return MEAN


#%% zGradientStack
def zGradientStack(IM):
    # Z-Gradient Stack
    #   Inputs:   I - hologram (grayscale)
    #            IM - median image
    #             Z - numpy array defining defocusing distances
    #   Output: CONV - 3D array representing stack of images at different Z
    import numpy as np
    from scipy import ndimage
    from functions import rayleighSommerfeldPropagator, exportAVI

    #    I = mpimg.imread('131118-1.png')
    #    I_MEDIAN = mpimg.imread('AVG_131118-2.png')
    #    Z = 0.02*np.arange(1, 151)
    #     IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)

    #% Sobel-type kernel
    SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
    SZ1 = np.zeros_like(SZ0)
    SZ2 = -SZ0
    SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
    del SZ0, SZ1, SZ2

    # Convolution IM*SZ
    # IM = IM ** 2
    IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
    GS = ndimage.convolve(IMM, SZ, mode='mirror')
    GS = np.delete(GS, [0, np.shape(GS)[2] - 1], axis=2)
    del IMM

    #    exportAVI('gradientStack.avi',CONV, CONV.shape[0], CONV.shape[1], 24)
    #    exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
    return GS


#%% dataCursor1D
def dataCursor1D():
    # Data Cursor in plots
    import mpldatacursor
    mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                             formatter='x = {i}\ny = {y:.06g}'.format)
    return 0


#%% dataCursor2D
def dataCursor2D():
    # Data Cursor in 2D plots
    import mpldatacursor
    mpldatacursor.datacursor(display='multiple', hover=True, bbox=dict(alpha=1, fc='w'),
                             formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
    return 0


#%% dataCursor3D
def dataCursor3D():
    # Data Cursor in 3D plots
    import mpldatacursor
    mpldatacursor.datacursor(hover=False, bbox=dict(alpha=1, fc='w'),
                             formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
    return 0


#%% histequ
def histeq(im):
    ## Histogram equalization of a grayscale image
    import numpy as np
    from PIL import Image

    # get image histogram
    imhist, bins = np.histogram(im.flatten(), 256, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


#%% guiImport
def guiImport():
    # GUI for values
    import PySimpleGUI as sg

    layout = [
        [sg.Text('Select AVi File recording', size=(35, 1)), sg.In(), sg.FileBrowse()],
        [sg.Text('Select Median Image (optional)', size=(35, 1)), sg.In(), sg.FileBrowse()],
        [sg.Text('Frame number to use for calculations', size=(35, 1)), sg.InputText(default_text=1)],
        [sg.Text('Number of frames for median image', size=(35, 1)), sg.InputText(default_text=20)],
        [sg.Text('Refraction index of media (water = 1.3226)', size=(35, 1)), sg.InputText(default_text=1.3226)],
        [sg.Text('Wavelength in um (HeNe/0.642)', size=(35, 1)), sg.InputText(default_text=0.642)],
        [sg.Text('Sampling Frequency px/um (0.711) ', size=(35, 1)), sg.InputText(default_text=0.711)],
        [sg.Text('Step size (10)', size=(35, 1)), sg.InputText(default_text=10)],
        [sg.Text('Number os steps (150)', size=(35, 1)), sg.InputText(default_text=150)],
        [sg.Text('Gradient Stack Threshold (~0.1)', size=(35, 1)), sg.InputText(default_text=0.1)],
        [sg.Text('Magnification (10, 20, etc)', size=(35, 1)), sg.InputText(default_text=10)],
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('Hologramphy inputs', layout)
    event, values = window.Read()
    window.Close()

    return values


#%% Positions3D
# Particles positions in 3D
def positions3D(GS, peak_min_distance):
    import numpy as np
    from skimage.feature import peak_local_max
   
    ZP = np.max(GS, axis=-1)
    PKS = peak_local_max(ZP, min_distance=peak_min_distance)  # 30
    
    # import matplotlib.pyplot as plt
    # plt.imshow(ZP, cmap='gray')
    # plt.scatter(PKS[:,1], PKS[:,0], marker='o', facecolors='none', s=80, edgecolors='r')
    # plt.show()
    
    D1 = 8
    D2 = 8
    Z_SUM_XY = np.empty((GS.shape[2], len(PKS)))
    for ii in range(len(PKS)):
        idi = PKS[ii, 0]
        idj = PKS[ii, 1]
        A = GS[idi-D1:idi+D2:, idj-D1:idj+D2, :]                # How to treat borders?
        Z_SUM_XY[:, ii] = np.sum(A, axis=(0, 1))
    
    Z_SUM_XY_MAXS_FOLDED = np.empty((len(PKS), 1), dtype=object)
    for ii in range(len(PKS)):
        Z_SUM_XY_MAXS_FOLDED[ii, 0] = peak_local_max(Z_SUM_XY[:, ii], num_peaks=1)
        if Z_SUM_XY_MAXS_FOLDED[ii, 0].size == 0:
            Z_SUM_XY_MAXS_FOLDED[ii, 0] = np.array([[0]])
    

    Z_SUM_XY_MAXS = []
    for ii in range(len(Z_SUM_XY_MAXS_FOLDED)):
        if len(Z_SUM_XY_MAXS_FOLDED[ii, 0]) != 1:
            for jj in range(len(Z_SUM_XY_MAXS_FOLDED[ii, 0])):
                Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0][jj].item(), ii])
        else:
            Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0].item(), ii])
    
    Z_SUM_XY_MAXS = np.array(Z_SUM_XY_MAXS)

    # XYZ_POSITIONS = np.empty((len(Z_SUM_XY_MAXS), 2))
    # POSPOS = np.empty((len(Z_SUM_XY_MAXS), 2))
    
    # for ii in range(len(Z_SUM_XY_MAXS)):
    #     XYZ_POSITIONS[ii, 0] = PKS[Z_SUM_XY_MAXS[ii, 1], 0]
    #     XYZ_POSITIONS[ii, 1] = PKS[Z_SUM_XY_MAXS[ii, 1], 1]
        
    # for ii in range(len(PKS)):
    #     POSPOS[ii, 0] = PKS[ii, 0]
    #     POSPOS[ii, 1] = PKS[ii, 1]
        
    
    # XYZ_POSITIONS = np.hstack((XYZ_POSITIONS, Z_SUM_XY_MAXS[:, 0]))
    XYZ_POSITIONS = np.insert(PKS, 2, Z_SUM_XY_MAXS[:, 0], axis=-1)         # Actually [Y, X, Z]

    return XYZ_POSITIONS   


#%% plot3D
def plot3D(LOCS, title, fig, ax):
    # 3D Scatter Plot
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot

    # fig = pyplot.figure()
    # ax = Axes3D(fig)

    ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize='20')
    ax.set_xlabel('x (pixels)', fontsize='18')
    ax.set_ylabel('y (pixels)', fontsize='18')
    ax.set_zlabel('z (slices)', fontsize='18')
    pyplot.show()

    return

#%% imshow_sequence
def imshow_sequence(im, delay, run):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    while run:
        for i in range(np.shape(im)[2]):
            ax.cla()
            ax.imshow(im[:, :, i], cmap='gray')
            ax.set_title("frame {}".format(i))
            # Note that using time.sleep does *not* work here!
            plt.pause(delay)

    return

#%% fraunhoffer
# Fraunhofer diffraction
def fraunhofer(rho, wsize, zdist):
    """Fraunhofer diffraction"""
    import numpy as np
    from scipy.special import j1
    lam = 0.642     # lambda
    # z = 5
    return j1(np.pi*wsize*rho/(lam*zdist))/(wsize*rho/(lam*zdist))

#%% Imshow_slider
def imshow_slider(cube, axis, color_map):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    # s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    
    if axis == 0:
        im = cube[0, :, :]
    elif axis == 1:
        im = cube[:, 0, :]
    elif axis == 2:
        im = cube[:, :, 0]

    # display image
    l = ax.imshow(im, cmap=color_map)

    # define slider
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = cube[s].squeeze()
        l.set_data(im)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()
    
#%% modified_propagator
def modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS):
    ## Rayleigh-Sommerfeld Back Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        IMM - 3D array representing stack of images at different Z
    import numpy as np
    from functions import bandpassFilter, histeq

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN
    #    IN = I - I_MEDIAN,
    #     IN[IN < 0] = 0

    # Bandpass Filter
    _, BP = bandpassFilter(IN, 2, 30)
    E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameters     #Set as input parameters
    # N = 1.3226               # Index of refraction
    LAMBDA = LAMBDA       # HeNe
    FS = FS               # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    # SZ = 10
    Z = SZ*np.arange(0, NUMSTEPS)
    # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
    #    Z = SZ*np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    # Q = np.empty_like(I_MEDIAN, dtype='complex64')
    # for i in range(NI):
    #     for j in range(NJ):
    #         Q[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)

    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
    Q = const*((ii-NI/2)**2 + (jj-NJ/2)**2)

    # P = np.conj(P)
    Q = np.sqrt(1 - Q) - 1

    if all(Z > 0):
        Q = np.conj(Q)

    R = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
    GS = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
    # R1 = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
    # IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = 1j*K*Q*np.exp((1j*K*Z[k]*Q), dtype='complex64')
        # R1 = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
        GS[:, :, k] = np.abs(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))
        # IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R1)))
        
    # GS, _ = histeq(GS)
    # TH = 254.9
    # GS[GS < TH] = 0
    # GS = 255*((GS - TH) / np.max(GS - TH))
    # GS[GS < 250] = 0
    
    GS = GS - 1
    _, BINS = np.histogram(GS.flatten())
    GS[GS < BINS[9]] = 0

    return GS

#%% Smooth trajectories piecewise cubic spline
def smooth_curve(L, spline_degree, lim, sc):
    import numpy as np
    from scipy import interpolate
    # from mpl_toolkits.mplot3d import Axes3D
    from scipy import ndimage
    
    #%
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # L = LINKED[LINKED.PARTICLE == 6].values
        
    num_sample_pts = len(L)
    x_sample = L[:, 0]
    y_sample = L[:, 1]
    z_sample = L[:, 2]
    
    jump = np.sqrt(np.diff(x_sample)**2 + np.diff(y_sample)**2 + np.diff(z_sample)**2) 
    smooth_jump = ndimage.gaussian_filter1d(jump, 1, mode='wrap')  # window of size 5 is arbitrary
    limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
    xn, yn, zn = x_sample[:-1], y_sample[:-1], z_sample[:-1]
    # xn = xn[(jump > 0) & (smooth_jump < limit)]
    # yn = yn[(jump > 0) & (smooth_jump < limit)]
    # zn = zn[(jump > 0) & (smooth_jump < limit)]
    
    xn = xn[(jump > 0)]
    yn = yn[(jump > 0)]
    zn = zn[(jump > 0)]
    
    m = len(xn)
    smoothing_condition = (m-np.sqrt(m), m+np.sqrt(m))
    smoothing_condition = np.mean(smoothing_condition)
    # smoothing_condition = sc
    spline_degree = 3
    
    if len(xn) > 3:
        tck, u = interpolate.splprep([xn,yn,zn], s=smoothing_condition, k=spline_degree)
        x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0, 1, num_sample_pts)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        X = [x_fine, y_fine, z_fine]
    else:
        X = -1
        
    # fig = plt.figure(figsize=(7, 4.5))
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.set_facecolor('none')
    # ax1.plot(x_sample, y_sample, z_sample, 'o')
    # ax1.plot(x_fine, y_fine, z_fine, '-')
    # plt.show()
    
    #%        
    return X

#%% smoothing with CSAPS
def csaps_smoothing(L, smoothing_condition, filter_data):
    import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    from scipy import ndimage
    from csaps import csaps, CubicSmoothingSpline
    
    # L = LINKED[LINKED.PARTICLE == 2]
    t_sample = np.linspace(0, 1, len(L))
    x_sample = L.X.values
    y_sample = L.Y.values
    z_sample = L.Z.values
    data = [x_sample, y_sample, z_sample]
    t_interp = np.linspace(0, 1, 1*len(L))
    
    # smoothing_condition = 0.999999
    
    if filter_data == False:
        # Smooth sample data
        smooth_data = csaps(t_sample, data, t_interp, smooth=smoothing_condition)
        x_smooth = smooth_data[0, :]
        y_smooth = smooth_data[1, :]
        z_smooth = smooth_data[2, :]
        
        # # Smooth sample data with variable smoothing condition
        # xi, smooth_x = csaps(t_sample, x_sample, t_interp)
        # yi, smooth_y = csaps(t_sample, y_sample, t_interp)
        # zi, smooth_z = csaps(t_sample, z_sample, t_interp)
    elif filter_data == True:
        # Filter sample data
        jump = np.sqrt(np.diff(x_sample)**2 + np.diff(y_sample)**2 + np.diff(z_sample)**2) 
        smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
        limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
        xn, yn, zn = x_sample[:-1], y_sample[:-1], z_sample[:-1]
        xn = xn[(jump > 0) & (smooth_jump < limit)]
        yn = yn[(jump > 0) & (smooth_jump < limit)]
        zn = zn[(jump > 0) & (smooth_jump < limit)]
        tn = np.linspace(0, 1, len(zn))
        
        # Smooth filtered data
        datani_smooth = csaps(tn, [xn, yn, zn], tn, smooth=smoothing_condition)
        x_smooth = datani_smooth[0, :]
        y_smooth = datani_smooth[1, :]
        z_smooth = datani_smooth[2, :]
        
        # xni_smooth = datani_smooth[0, :]
        # yni_smooth = datani_smooth[1, :]
        # zni_smooth = datani_smooth[2, :]
        
        # # Smooth filtered sample data with variable smoothing condition
        # xni, smooth_xni = csaps(tn, xn, tn)
        # yni, smooth_yni = csaps(tn, yn, tn)
        # zni, smooth_zni = csaps(tn, zn, tn)
    
    # Plot data
    # fig = plt.figure(figsize=(7, 4.5))
    
    # ax1 = fig.add_subplot(221, projection='3d')
    # # ax11.set_facecolor('none')
    # ax1.plot(x_sample, y_sample, z_sample, 'ro', label='Sample Data')
    # ax1.plot(x_smooth, y_smooth, z_smooth, 'b-', label='Smoothed Sample Data')
    # ax1.legend(loc='upper left')
    
    # ax2 = fig.add_subplot(222, projection='3d')
    # ax2.plot(x_sample, y_sample, z_sample, 'ro', label='Sample Data')
    # ax2.plot(xi, yi, zi, 'b-', label='Smoothed (variable) Sample Data')
    # ax2.legend(loc='upper left')
    
    # # fig1 = plt.figure(figsize=(7, 4.5))
    # ax3 = fig.add_subplot(223, projection='3d')
    # # ax21.set_facecolor('none')
    # ax3.plot(xn, yn, zn, 'ro', label='Sample Data (Filtered)')
    # ax3.plot(xni_smooth, yni_smooth, zni_smooth, 'b-', label='Smoothed Sample Data (Filtered)')
    # ax3.legend(loc='upper left')
    
    # ax4 = fig.add_subplot(224, projection='3d')
    # ax4.plot(xn, yn, zn, 'ro', label='Sample Data (Filtered)')
    # ax4.plot(xni, yni, zni, 'b-', label='Smoothed (variable) Sample Data (Filtered)')
    # ax4.legend(loc='upper left') 
    # plt.show()
    
    return [x_smooth, y_smooth, z_smooth]
    
    
#%% Surface plot

def surf(array):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(array.shape[1])
    y = np.arange(array.shape[0])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, array, cmap='jet')
    plt.show()


    return
    
