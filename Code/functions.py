#%% Convert rgb image to grayscale using Y' = 0.299R'+0.587G' + 0.114B'
# Input:     img - RBG image
# Output: img_gs - Grayscale image
def rgb2gray(img):
    import numpy as np
    [ni,nj,nk] = img.shape
    img_gs = np.empty([ni, nj])
    for ii in range(0, ni):
        for jj in range(0, nj):
            img_gs[ii, jj] = 0.299*img[ii, jj, 0] + 0.587*img[ii, jj, 1] + 0.114*img[ii, jj, 2]
            
    return img_gs

#%% Make image square by adding rows or columns of the mean value of the image np.mean(img)
# Input: img - grayscale image
# Output: imgs - square image
#         axis - axis where data is added
#            d - number of rows/columns added
def square_image(img):
    import numpy as np

    [ni, nj] = img.shape
    dn = ni - nj
    d = abs(dn)
    if dn < 0:
            M = np.flip(img[ni-abs(dn):ni,:], 0)
            imgs = np.concatenate((img,M), axis = 0)
            axis = 'i'
    elif dn > 0:
            M = np.flip(img[:,nj-abs(dn):nj], 1)
            imgs = np.concatenate((img,M), axis = 1)
            axis = 'j'
    elif dn == 0:
            imgs = img
            axis = 'square'            
    return imgs, axis, d

#%% Bandpass filter
# Input: img - Grayscale image array (2D)
#        xl  - Large cutoff size (Pixels)
#        xs  - Small cutoff size (Pixels)
# Output: img_filt - filtered image
def bandpassFilter(img,xl,xs):
    import numpy as np
    
    # FFT the grayscale image
    imgfft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(imgfft)
    img_amp = abs(img_fft)
    del imgfft
    
    # Pre filter image information
    [ni,nj] = img_amp.shape
    MIS = ni
         
    # Create bandpass filter when BigAxis == 
    LCO = np.empty([ni,nj])
    SCO = np.empty([ni,nj])
    
    for ii in range(0,ni-1):
        for jj in range(0,nj-1):
            LCO[ii, jj] = np.exp(-((ii-MIS/2)**2+(jj-MIS/2)**2)*(2*xl/MIS)**2)
            SCO[ii, jj] = np.exp(-((ii-MIS/2)**2+(jj-MIS/2)**2)*(2*xs/MIS)**2)            
    BP = SCO - LCO  
    BPP = np.fft.ifftshift(BP)     
    # Filter image 
    filtered = BP*img_fft
    img_filt = np.fft.ifftshift(filtered) 
    img_filt= np.fft.fft2(img_filt)       
    img_filt = np.rot90(np.real(img_filt),2)
    
    return  img_filt, BPP

#%% Import video as stack of images in a 3D array
#   Input:  video   - path to video file
#   Output: imStack - 3D array of stacked images in 8-bit
def videoImport(video):
    import cv2
    import numpy as np
    
    CAP = cv2.VideoCapture(video)
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('uint8'))
    IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))
    
    I = 0
    SUCCESS = True
    
    while (I < NUM_FRAMES  and SUCCESS):
        SUCCESS, IMG[I] = CAP.read()
        IM_STACK[I] = IMG[I, :, :, 1]
        I += 1
        print(('VI', I))
    
    CAP.release()
    IM_STACK = np.swapaxes(np.swapaxes(IM_STACK, 0, 2), 0, 1)

    return IM_STACK

#%% Export 3D array to .AVI movie file
#   Input:  IM - numpy 3D array
#           NI - number of rows of array
#           NJ - number of columns of array
#          fps - frames per second of output file
#   Output: .AVI file in working folder 
def exportAVI(filename,IM, NI, NJ, fps):
    import os
    from cv2 import VideoWriter, VideoWriter_fourcc
    dir = os.getcwd()
    filename = os.path.join(dir,filename)
    FOURCC = VideoWriter_fourcc(*'MJPG')
    VIDEO = VideoWriter(filename, FOURCC, float(fps), (NJ, NI),0)
    
    for i in range(IM.shape[2]):
        frame = IM[:,:,i]
    #    frame = np.random.randint(0, 255, (NI, NJ,3)).astype('uint8')
        VIDEO.write(frame)
        
    VIDEO.release()
    return 'Video exported successfully'

#%% Rayleigh-Sommerfeld Back Propagator
#   Inputs:          I - hologram (grayscale)
#             I_MEDIAN - median image
#                    Z - numpy array defining defocusing distances
#   Output:        IMM - 3D array representing stack of images at different Z
def rayleighSommerfeldPropagator(I, I_MEDIAN, Z):  
    import math as m
    import numpy as np
    from functions import bandpassFilter
        
    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.average(I_MEDIAN)
    IN = I/I_MEDIAN
#    IN = I - I_MEDIAN
    IN[IN < 0] = 0
    
    # Bandpass Filter
    _, BP = bandpassFilter(IN, 2, 30)
    E = BP*np.fft.fft2(IN - 1)
    
    # Patameters     #Set as input parameters
    N = 1.3226               # Index of refraction
    LAMBDA = 0.642           # HeNe
    FS = 1.422               # Sampling Frequency px/um
    NI = np.shape(IN)[0]     # Number of rows
    NJ = np.shape(IN)[1]     # Nymber of columns
    K = 2*m.pi*N/LAMBDA      # Wavenumber
    
    # Rayleigh-Sommerfeld Arrays
    P = np.empty_like(I_MEDIAN, dtype=complex)
    for i in range(NI):
        for j in range(NJ):
            P[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)
            
    Q = np.sqrt(1-P)-1
    Q = np.conj(Q)    
    R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
    IZ = np.empty_like(R, dtype=float)
#    for k in range(Z.shape[0]):        
#        for i in range(NI):
#            for j in range(NJ):
#                R[i, j, k] = np.exp((-1j*K*Z[k])*Q[i, j])
#        IZ[:, :, k] = 1+np.fft.ifft2(BP*(np.fft.fft2(IN-1))*R[:, :, k])
#        print(('RS' ,k, i, j))
    for k in range(Z.shape[0]):
        R[:, :, k] = np.exp((-1j*K*Z[k]*Q))
        IZ[:, :, k] = 1 + np.real(np.fft.ifft2(E*R[:, :, k]))
#        print(('RS', k))
            
        #Converted to 8-bit
#        IZ8 = np.uint8((IZ - np.min(IZ))*255)
#        IZ8 = np.uint8(IZ8/np.max(IZ8)*255)
#        IZ8 = np.uint8(255*(IZ8/255)**2)
    
    return IZ

#%% Median Image
#   Input:   VID - 3D numpy array of video file
#   Output: MEAN - 2D pixel mean array
def medianImage(VID):
    import numpy as np
    from scipy import ndimage

    print('MI')
    MEAN = np.median(VID, axis=2)

    return MEAN

#%% Z-Gradient Stack
 #   Inputs:   I - hologram (grayscale)
#            IM - median image
#             Z - numpy array defining defocusing distances
#   Output: CONV - 3D array representing stack of images at different Z
#           
def zGradientStack(I, I_MEDIAN, Z):
    import numpy as np
    from scipy import ndimage
    from functions import rayleighSommerfeldPropagator, exportAVI
    
#    I = mpimg.imread('131118-1.png')
#    I_MEDIAN = mpimg.imread('AVG_131118-2.png')
#    Z = 0.02*np.arange(1, 151)
    IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
    
    #%% Sobel-type kernel
    SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
    SZ1 = np.zeros_like(SZ0)
    SZ2 = -SZ0
    SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
    del SZ0, SZ1, SZ2, I, I_MEDIAN, Z
    
    #%% Convolution IM*SZ    
    IMM = np.dstack((IM[:,:,0][:, :, np.newaxis], IM, IM[:,:,-1][:, :, np.newaxis]))
    GS = ndimage.convolve(IMM, SZ, mode='mirror')  
    GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)
    del IMM
    
    
#    exportAVI('gradientStack.avi',CONV, CONV.shape[0], CONV.shape[1], 24)
#    exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
    return GS, IM

