import nn
#from apogee.aspcap import ferre
#from apogee.aspcap import norm
#from apogee.utils import spectra
#from tools import plots
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pdb
from astropy.io import fits

def lsf_smooth(wrange=[3600,10400],sig=1.2):
    """ Smooth FERRE library files with MaStar LSF
    """

    # get median dispersion as a function of wavelength
    mastar=fits.open('mastar-goodspec-v2_7_1-trunk.fits')[1].data
    if sig == None : disp=np.median(mastar['DISP'],axis=0)
    else : disp=sig*.0001*np.log(10)*mastar['WAVE'][0]
    #plt.plot(mastar['WAVE'][0],disp)
    #plt.plot(mastar['WAVE'][0],sig)
    #pdb.set_trace()

    # loop over files
    for file in ['f_nsc1','f_nsc2','f_nsc3','f_nsc4','f_nsc5'] :

        # read files and trim wavelengths
        hdr=ferre.rdlibhead(file+'.hdr')
        wave=10.**(hdr['WAVE'][0]+np.arange(hdr['NPIX'])*hdr['WAVE'][1])
        i1=np.argmin(np.abs(wave-wrange[0]))
        i2=np.argmin(np.abs(wave-wrange[1]))
        deltav=hdr['WAVE'][1]*3.e5*np.log(10)
        print('deltav: ', deltav)

        # read spectra and unwrap dimensions
        lib=np.fromfile(file+'.unf',dtype=np.float32)
        dim=[]
        for i in range(hdr['N_OF_DIM']) : dim.append(hdr['N_P'][i])
        dim.append(hdr['NPIX'])
        lib=np.reshape(lib,dim)
        wave=wave[i1:i2]
        lib=lib[:,:,:,i1:i2]

        # add rotation dimension
        nrot=5
        vsinis = 1.+np.arange(nrot)*np.log10(2.5)
        newdim=[nrot]
        newdim.extend(list(np.shape(lib)))
        conv=np.zeros(newdim)
        # loop over the input wavelengths
        for irot,vsini in enumerate(vsinis) :
            rot=rotate(deltav,10.**vsini)
            for pix,w in enumerate(wave) :
                # get LSF sigma from nearest MaStar pixel
                diff = np.abs(w-mastar['wave'][0,:])
                idisp=np.argmin(diff)
                sig=disp[idisp]
                sig=1.5*.0001*np.log(10)*w
                dw=wave-w
                # create the kernel
                kernel=np.tile(np.convolve(np.exp(-dw**2/2/sig**2),rot,mode='same'),(1,1,1,1))
                #kernel=np.tile(rot,(1,1,1,1))
                #kernel=np.tile(np.exp(-dw**2/2/sig**2),(1,1,1,1))

                # add this pixel from all of the models into the output array
                gd=np.where(kernel > 0.001)[3]
                conv[irot,:,:,:,gd[0]:gd[-1]+1] += kernel[:,:,:,gd]*lib[:,:,:,np.tile(pix,len(gd))]/np.sqrt(2*np.pi)/sig
                print(w,10.**vsini,len(gd),idisp,sig,kernel.shape,gd[0],kernel[0,0,0,gd[0]],np.sum(kernel)/np.sqrt(2*np.pi)/sig)

        # now interpolated smoothed high res spectra to MaStar wavelengths 
        interp=scipy.interpolate.interp1d(spectra.airtovac(wave),conv,axis=4)
        new=interp(mastar['WAVE'][0,:])

        # fill headers and output FITS file
        hdulist=fits.HDUList()
        hdu=fits.PrimaryHDU(new)
        ndim = hdr['N_OF_DIM']
        hdu.header['NAXIS'] = ndim+1
        hdu.header['CRVAL1'] = np.log10(mastar['WAVE'][0,0])
        hdu.header['CDELT1'] = np.log10(mastar['WAVE'][0,1]/mastar['WAVE'][0,0])
        hdu.header['CTYPE1'] = 'LOG(WAVELENGTH)'
        for i in range(ndim,0,-1) :
            hdu.header['NAXIS{:d}'.format(ndim-i+2)] = hdr['N_P'][i-1]
            hdu.header['CRVAL{:d}'.format(ndim-i+2)] = hdr['LLIMITS'][i-1]
            hdu.header['CDELT{:d}'.format(ndim-i+2)] = hdr['STEPS'][i-1]
            hdu.header['CTYPE{:d}'.format(ndim-i+2)] = hdr['LABEL'][i-1].decode('UTF-8')
        hdu.header['NAXIS5'] = 5
        hdu.header['CRVAL5'] = 1.
        hdu.header['CDELT5'] = np.log10(2.5)
        hdu.header['CTYPE5'] = 'LOG(VSINI)'
        hdulist.append(hdu)
        hdulist.writeto(file+'_rot.fits',overwrite=True)

    pdb.set_trace()




def comp(lib,teff=5000,logg=5,mh=0,am=0,vmicro=np.log10(1.),clear=True) :

    mod=nn.get_model('stack600')
    pix = np.arange(4563)
    lab=[teff,logg,mh,am]
    spec = nn.spectrum(pix, *lab)


    wave=lib.header['CRVAL1']+np.arange(lib.header['NAXIS1'])*lib.header['CDELT1']
    ilogg = np.round((logg-lib.header['CRVAL2'])/lib.header['CDELT2']).astype(int)
    iteff = np.round((teff-lib.header['CRVAL3'])/lib.header['CDELT3']).astype(int)
    imh = np.round((mh-lib.header['CRVAL4'])/lib.header['CDELT4']).astype(int)
    print(ilogg,iteff,imh)

    #inds=np.round(([mh,am,vmicro,teff,logg]-hdr['LLIMITS'])/hdr['STEPS']).astype(int)
    #synth = lib[inds[0],inds[1],inds[2],inds[3],inds[4],600:-200]

    synth = lib.data[imh,iteff,ilogg,:]
    synth = synth / norm.cont(synth,synth,medfilt=400,order=0,poly=False)
    synth=synth[600:-200]

    fig,ax=plots.multi(1,2,hspace=0.001,sharex=True)
    if clear: 
        ax[0].cla()
        ax[1].cla()
    ax[0].plot(10.**wave[600:-200],spec)
    ax[0].plot(10.**wave[600:-200],synth)
    ax[1].plot(10.**wave[600:-200],spec/synth)
    plt.draw()



def rotate(deltav,vsini,epsilon=0.6) :
    """ rotation kernel from IDL Users library routine
    
    Args :
        deltav (float) : velocity spacing of pixels
        vsini (float ) : v sin i for profile
        epsilon (float ) : parameter for rotation kernel (default=0.6)

    Returns :
        normalized rotation profile
    """
    e1 = 2.0*(1.0 - epsilon)
    e2 = np.pi*epsilon/2.0
    e3 = np.pi*(1.0 - epsilon/3.0)

    npts = np.ceil(2*vsini/deltav)
    if npts%2 == 0 : npts = npts +1
    nwid = int(npts/2.)
    x = (np.arange(0,npts)- nwid)
    x = x*deltav/vsini
    x1 = abs(1.0 - x**2)
    kernel=(e1*np.sqrt(x1) + e2*x1)/e3
    return kernel/kernel.sum()


