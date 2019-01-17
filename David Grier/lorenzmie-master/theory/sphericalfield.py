import numpy as np
from builtins import range

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != np.ndarray:
        print(char_x + ' must be an numpy array')
        return False
    else:
        return True

def sphericalfield(x, y, z, ab, lamb, cartesian=False, str_factor=False, 
                   convention='Bohren'):
    """
    Calculate the complex electric field (or electric field strength factor) 
    due to a Lorenz-Mie scatterer a height z [pixels] above the grid (sx, sy).

    Args:
        x: [npts] array of pixel coordinates [pixels]
        y: [npts] array of pixel coordinates [pixels]
        z: If field is required in a single plane, then
            z is the plane's distance from the sphere's center
            [pixels]. Otherwise it is an [npts] array of pixel coordinates 
            [pixels].
        ab: [2, nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
        lamb: wavelength of light in medium [pixels]

    Keywords:
        cartesian: if True, field is expressed as (x, y, z) else (r, theta, phi)
        str_factor: if True, returned field is the electric field strength 
            factor
    Returns:
        field: [3, npts] scattered electric field or field strength factor
    """

    # Check that inputs are numpy arrays
    for var, char_var in zip([x,y,ab], ['x', 'y', 'ab']):
        if not check_if_numpy(var, char_var):
            print('x, y and ab must be numpy arrays')
            return None

    z = np.array(z) # In case it is a float or integer.
    type_z = type(z)
    if type_z != np.ndarray and type_z != int and type_z != float:
        print('z must be a float, int or numpy array.')
        return None

    signZ = np.sign(z)

    # Check the inputs are the right size
    if x.shape != y.shape:
        print('x has shape {} while y has shape {}'.format(x.shape, y.shape))
        print('and yet their dimensions must match.')
        return None

    npts = len(x)
    nc = len(ab[:,0])-1     # number of terms required for convergence

    k = 2.0 * np.pi / lamb # wavenumber in medium [pixel^-1]

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    rho   = np.sqrt(x**2 + y**2)
    r     = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    phi   = np.arctan2(y, x)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi   = np.cos(phi)
    sinphi   = np.sin(phi)

    kr = k*r # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = np.sin(kr)
    coskr = np.cos(kr)

    '''
    Particles above the focal plane create diverging waves described by
    Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0. Those below the focal
    plane appear to be converging from the perspective of the camera. They are
    descrinbed by Eq. (4.14) for $h_n^{(2)}(kr)$, and have z < 0. We can select
    the appropriate case by applying the correct sign of the imaginary part of
    the starting functions...
    '''
    if convention == 'Bohren':
        factor = 1
    else:
        factor = -1
    xi_nm2 = coskr + factor * signZ*1.j*sinkr # \xi_{-1}(kr)
    xi_nm1 = sinkr - factor * signZ*1.j*coskr # \xi_0(kr)
    #xi_nm2 = coskr + 1.j*sinkr
    #xi_nm1 = sinkr - 1.j*coskr
    # ... angular functions (4.47), page 95
    pi_nm1 = 0.0                    # \pi_0(\cos\theta)
    pi_n   = 1.0                    # \pi_1(\cos\theta)

    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = np.zeros([3, npts],complex)
    Ne1n = np.zeros([3, npts],complex)

    # storage for scattered field
    Es = np.zeros([3, npts],complex)

    # Compute field by summing multipole contributions
    for n in range(1, nc+1):

        # upward recurrences ...
        # ... Legendre factor (4.47)
        # Method described by Wiscombe (1980)
        swisc = pi_n * costheta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

        # ... Riccati-Bessel function, page 478
        xi_n = (2.0*n - 1.0) * xi_nm1 / kr - xi_nm2    # \xi_n(kr)

        # vector spherical harmonics (4.50)
        #Mo1n[0, :] = 0               # no radial component
        Mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
        Mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

        dn = (n * xi_n)/kr - xi_nm1
        Ne1n[0, :] = n*(n + 1.0) * pi_n * xi_n # ... divided by cosphi sintheta/kr^2
        Ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
        Ne1n[2, :] = pi_n  * dn      # ... divided by sinphi/kr

        # prefactor, page 93
        En = 1.j**n * (2.0*n + 1.0) / n / (n + 1.0)

        # the scattered field in spherical coordinates (4.45)
        Es += En * (1.j * ab[n,0] * Ne1n - ab[n,1] * Mo1n)

        # upward recurrences ...
        # ... angular functions (4.47)
        # Method described by Wiscombe (1980)
        pi_nm1 = pi_n
        pi_n = swisc + (n + 1.0) * twisc / n

        # ... Riccati-Bessel function
        xi_nm2 = xi_nm1
        xi_nm1 = xi_n

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    if str_factor:
        # Compute the electric field strength factor by removing r-dependence.
        radialFactor = np.exp(-1.0j*kr) / k
    else:
        radialFactor = 1 / kr
    Es[0, :] *= cosphi * sintheta * radialFactor / kr
    Es[1, :] *= cosphi * radialFactor
    Es[2, :] *= sinphi * radialFactor

    # By default, the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and 
    # is linearly polarized along x
    if cartesian:
        Ec = np.zeros([3, npts], complex)
        Ec += Es

        Ec[0, :] =  Es[0, :] * sintheta * cosphi
        Ec[0, :] += Es[1, :] * costheta * cosphi
        Ec[0, :] -= Es[2, :] * sinphi

        Ec[1, :] =  Es[0, :] * sintheta * sinphi
        Ec[1, :] += Es[1, :] * costheta * sinphi
        Ec[1, :] += Es[2, :] * cosphi


        Ec[2, :] =  Es[0, :] * costheta - Es[1, :] * sintheta

        return Ec
    else:
        return Es

def test_angular_spectrum():
    '''Tests r dependence of angular spectrum.'''
    from sphere_coefficients import sphere_coefficients
    import matplotlib.pyplot as plt
    from spheredhm import spheredhm

    # Choose necessary constants.
    nx, ny = 201, 201
    mpp = 0.135
    lamb = 0.447
    a_p = 0.5 
    n_p = 1.5
    nm_obj = 1.3326
    nm_img = 1.000

    # Make a cartesian grid of points in the focal plane.
    sx = np.tile(np.arange(nx, dtype = float), ny)
    sy = np.repeat(np.arange(ny, dtype = float), nx)
    sx -= float(nx)/2.
    sy -= float(ny)/2.

    # Compute sphere coefficients.
    ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)

    # Compute the angular spectrum.
    lamb_m = lamb/np.real(nm_obj)/mpp # medium wavelength [pixel]
    z_p = 100.
    low_ang_spec = sphericalfield(sx, sy, z_p, ab, lamb_m, cartesian = False, str_factor = True)
    low_ang_spec = low_ang_spec.reshape(3, ny, nx)
    low_ang_spec = np.hstack(map(np.abs, low_ang_spec[0:3, :, :]))

    z_p *= 10
    mid_ang_spec = sphericalfield(sx*10, sy*10, z_p, ab, lamb_m, cartesian = False, str_factor = True)
    mid_ang_spec = mid_ang_spec.reshape(3, ny, nx)
    mid_ang_spec = np.hstack(map(np.abs, mid_ang_spec[0:3, :, :]))

    z_p *= 1000
    high_ang_spec = sphericalfield(sx*10000, sy*10000, z_p, ab, lamb_m, cartesian = False, str_factor = True)
    high_ang_spec = high_ang_spec.reshape(3, ny, nx)
    high_ang_spec = np.hstack(map(np.abs, high_ang_spec[0:3, :, :]))

    # Plot results.
    ang_stack = np.vstack([low_ang_spec, mid_ang_spec, high_ang_spec])
    #ang_stack = np.abs(ang_stack)
    plt.imshow(ang_stack, interpolation = None)
    plt.title('Angular Spectrum at small, medium and large r')
    plt.show()


if __name__ == '__main__':
    test_angular_spectrum()
