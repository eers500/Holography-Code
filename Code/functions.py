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
def Bandpass_Filter(img,xl,xs):
    import numpy as np
    # FFT the grayscale image
    imgfft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(imgfft)
    img_amp = abs(img_fft)
    del imgfft
    
    # Pre filter image information
    [ni,nj]=img_amp.shape
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
#   Output: imStack - 3D array of stacked images
def videoImport(video):
    import numpy as np
    import cv2
    import time
    
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    imStack = image[:,:,0] 
    t0 = time.time()
    for ii in range(1,num_frames):
            success,fr = vidcap.read()
            frame = fr[:,:,0]
            imStack = np.dstack((imStack,frame))
    t1 = time.time()
    time = t1-t0
    
    return imStack
