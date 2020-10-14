from __future__ import division
import numpy as np
import pdb

import norm
import training
import matplotlib
try: matplotlib.use('Agg')
except : pass
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
try:
    from keras import models
    from keras import layers
    from keras import optimizers
    from keras import regularizers
except :
    print('keras not available!')

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, TableColumns, Column
import pickle
import copy
import os
import sys
import shutil
import time
try:
    import emcee
except:
    print('emcee not available!')
try: import corner
except: pass


nepochs=10000
nodes=20
reg=0.0005
batch_size=1000
verbose=0
reg=0.
nepochs=25000

nodes=20
nepochs=50000

nodes=300
nepochs=5

def get_model(file,aspcappix=False) :
    """ load model and set up for use
    """
    global NN_coeffs

    try :
        with open(file+'.pkl','rb') as f: 
            NN_coeffs = pickle.load(f)
    except:
        tmp = np.load(file+'.npz')
        NN_coeffs={}
        NN_coeffs['w_array_0'] = tmp["w_array_0"]
        NN_coeffs['w_array_1'] = tmp["w_array_1"]
        NN_coeffs['w_array_2'] = tmp["w_array_2"]
        NN_coeffs['b_array_0'] = tmp["b_array_0"]
        NN_coeffs['b_array_1'] = tmp["b_array_1"]
        NN_coeffs['b_array_2'] = tmp["b_array_2"]
        NN_coeffs['x_min'] = tmp["x_min"]
        NN_coeffs['x_max'] = tmp["x_max"]
        tmp.close()

    if aspcappix :
        tmp=fits.open(NN_coeffs['data_file']+'.fits')[2].data[0,:]
        gdpix=np.where(np.isfinite(tmp))[0]
        gridpix=set()
        for i in range(3) : gridpix = gridpix | set(range(aspcap.gridPix()[i][0],aspcap.gridPix()[i][1]))
        NN_coeffs['gdmodel'] = [i for i in range(len(gdpix)) if gdpix[i] in gridpix]

    return NN_coeffs


def func(pars,obs,obserr,order) :
    """ Return minimization quantity
    """
    scaled_labels = (np.array(pars)-NN_coeffs['x_min'])/(NN_coeffs['x_max']-NN_coeffs['x_min']) - 0.5
    tmp = np.dot(NN_coeffs['w_array_0'],scaled_labels)+NN_coeffs['b_array_0']
    nlayers=len(NN_coeffs['num_neurons'])
    for i in range(nlayers) :
        spec = np.dot(sigmoid(tmp),NN_coeffs['w_array_{:d}'.format(i+1)].T)+NN_coeffs['b_array_{:d}'.format(i+1)]
        tmp = spec

    try : spec=spec[NN_coeffs['gdmodel']]
    except: pass

    if order > 0 :
        cont = norm.cont(spec,obserr,poly=True,order=order,chips=True,apstar=False)
        spec /=cont

    return ((obs-spec)**2/obserr**2).sum()

def spectrum(x,*pars) :
    """ Return full spectrum given input list of pixels, parameters
    """
    scaled_labels = (np.array(pars)-NN_coeffs['x_min'])/(NN_coeffs['x_max']-NN_coeffs['x_min']) - 0.5
    #pdb.set_trace()
    #inside = np.einsum('ij,j->i', NN_coeffs['w_array_0'], scaled_labels) + NN_coeffs['b_array_0']
    #outside = np.einsum('ij,j->i', NN_coeffs['w_array_1'], sigmoid(inside)) + NN_coeffs['b_array_1']
    #spec = np.einsum('ij,j->i', NN_coeffs['w_array_2'], sigmoid(outside)) + NN_coeffs['b_array_2']

    tmp = np.dot(NN_coeffs['w_array_0'],scaled_labels)+NN_coeffs['b_array_0']
    nlayers=len(NN_coeffs['num_neurons'])
    for i in range(nlayers) :
        spec = np.dot(sigmoid(tmp),NN_coeffs['w_array_{:d}'.format(i+1)].T)+NN_coeffs['b_array_{:d}'.format(i+1)]
        tmp = spec

    try : 
        spec=spec[NN_coeffs['gdmodel']]
        cont = norm.cont(spec,spec*0.+1.,poly=True,order=4,chips=True,apstar=False)
        spec /=cont
    except: pass

    return spec

def test(pmn, pstd, mn, std, weights, biases,n=100, t0=[3750.,4500.], g0=2., mh0=0.) :
    """ Plots cross-sections of model for fit pixels
    """
    fig,ax=plots.multi(2,6,figsize=(8,12))

    xt=['Teff','logg','[M/H]','[alpha/M]','[C/M]','[N/M]']
    for i,ipar in enumerate([0,1,2,3,4,5]) : 
      for ipix in range(len(weights)) :
       for it0 in range(2) :
        pars=np.tile([t0[it0], g0, mh0, 0.0, 0., 0., 2.],(n,1))
        if ipar == 0 : pars[:,ipar]=np.linspace(3000.,8000.,n)
        elif ipar == 1 : pars[:,ipar]=np.linspace(-0.5,5.5,n)
        elif ipar == 2 : pars[:,ipar]=np.linspace(-2.5,1.,n)
        elif ipar == 3 : pars[:,ipar]=np.linspace(-0.5,1.0,n)
        elif ipar == 4 : pars[:,ipar]=np.linspace(-1.,1.,n)
        elif ipar == 5 : pars[:,ipar]=np.linspace(-0.5,2.,n)
        m=[]
        for ip in range(pars.shape[0]) : m.append(model((pars[ip,:]-pmn)/pstd,mn[ipix],std[ipix],weights[ipix],biases[ipix]))
        plots.plotl(ax[i,it0],pars[:,ipar],m,xt=xt[i])
        #m=[]
        #for ip in range(pars.shape[0]) : m.append(nets[ipix].predict((pars[ip,:].reshape(1,7)-pmn)/pstd)[0,0]*std[ipix]+mn[ipix])
        #plots.plotl(ax[i,it0],pars[:,ipar],m)
        if i == 0 : ax[i,it0].set_title('{:8.0f}{:7.2f}{:7.2f}'.format(t0[it0],g0,mh0))
    fig.tight_layout()

def lnprior(pars) :
    return 0.

def lnprob(pars,s,serr) :
    model=spectrum(s,*pars)
    return -0.5*np.sum((s-model)**2/serr**2) + lnprior(pars)
    
def solve_mcmc(spec, nburn=50, nsteps=500, nwalkers=100, eps=0.01) :
    s=spec[0]
    serr=spec[1]
    init=spec[2]   
    star=spec[3]
    ndim = len(init)
    pix = np.arange(0,len(s),1)
    gd = np.where(np.isfinite(s))[0]
    pos = [init + eps*np.random.randn(ndim)*init for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(s[gd], serr[gd]))
    print(init)
    print('running mcmc...')
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    fig =corner.corner(samples,show_titles=True,quantiles=[0.05,0.95])
    fig.savefig('mcmc/'+star+'.png')


def solve(spec) :
    """ Solve for parameters for a single input spectrum
    """
    s=spec[0]
    serr=spec[1]
    init=spec[2]
    bounds=spec[3]
    order=spec[4]
    pix = np.arange(0,len(s),1)
    gd = np.where(np.isfinite(s))[0]
    try:
        # do a least squares pass, which doesn't accomodate passing specerr for continuum
        try : fpars,fcov = curve_fit(spectrum,pix[gd],s[gd],sigma=serr[gd],p0=init,bounds=bounds)
        except : 
            print('curve_fit failed...')
            fpars = init
        newbounds=[]
        for i in range(len(bounds[0])) : newbounds.append((bounds[0][i],bounds[1][i]))
        try: res = minimize(func,fpars,args=(s[gd],serr[gd],order),bounds=newbounds)
        except: print('minimize failed')
    except ValueError:
        print("Error - value error")
        print(init)
        fpars=init*0.
    except RuntimeError:
        print("Error - curve_fit failed")
        fpars=init*0.

    #return fpars
    try : return res
    except: return 0


def normalize(pars) :
    """ bundled normalize for multi-threading
    """
    spec=pars[0]
    specerr=pars[1]
    pixels=pars[2]

    cont = norm.cont(spec,specerr,poly=False,chips=False,apstar=False,medfilt=400)
    nspec = spec/cont
    nspecerr = specerr/cont
    bd=np.where(np.isinf(nspec) | np.isnan(nspec) )[0]
    nspec[bd]=0.
    nspecerr[bd]=1.e10
    bd=np.where(np.isinf(nspecerr) | np.isnan(nspecerr) )[0]
    nspec[bd]=0.
    nspecerr[bd]=1.e10
    if pixels is not None : 
        nspec = nspec[pixels[0]:pixels[1]]
        nspecerr = nspecerr[pixels[0]:pixels[1]]
    return nspec,nspecerr

def fitmastar(model='test',field='mastar-goodspec-v2_7_1-trunk',star=None,nfit=0,order=0,threads=8,
              write=True,telescope='apo25m',pixels=None,hmask=False,mcmc=False) :
    """ Fit observed spectra in an input field, given a model
    """

    # get model and list of stars
    mod = get_model(model)
    nlab=len(mod['label_names'])
    bounds_lo=mod['x_min']
    bounds_hi=mod['x_max']

    # get stars
    stars=fits.open(field+'.fits')[1].data
    if nfit > 0 : stars = stars[0:nfit]
    if star is not None: 
        j=np.where(stars['MANGAID'] == star)[0]
        stars=stars[j]
    stars=Table(stars)
    stars['EBV'] = -1.

    # load up normalized spectra and uncertainties 
    norms=[]
    for i,star in enumerate(stars) :
        norms.append((star['FLUX'],np.sqrt(1./star['IVAR']),pixels))
            

    if threads==0 :
        output=[]
        for i in range(len(norms)) :
            out=normalize(norms[i])
            output.append(out)
    else :
        print('starting pool: ', len(norms))
        pool = mp.Pool(threads)
        output = pool.map_async(normalize, norms).get()
        pool.close()
        pool.join()

    # set initial guesses
    init=np.zeros([len(stars),nlab])
    bounds_lo=np.zeros([len(stars),nlab])
    bounds_hi=np.zeros([len(stars),nlab])
    j_teff=np.where(np.core.defchararray.strip(mod['label_names']) == 'TEFF')[0]
    init[:,j_teff] = 4500.
    j_logg=np.where(np.core.defchararray.strip(mod['label_names']) == 'LOGG')[0]
    init[:,j_logg] = 2.0
    j_rot=np.where(np.core.defchararray.strip(mod['label_names']) == 'LOG(VSINI)')[0]
    init[:,j_rot] = 1.01
    j_mh=np.where(np.core.defchararray.strip(mod['label_names']) == '[M/H]')[0]

    extcorr=fits.open('trunk/goodstars-v2_7_1-gaia-extcorr.fits')[1].data

    # rough color-temp interpolator from isochrone points
    color=[-0.457,-0.153,0.328,1.247,2.172,3.215]
    logte=[4.4822,4.1053,3.8512,3.678,3.5557,3.5246]
    f=interp1d(color,logte,kind='linear')

    specs=[]
    pix = np.arange(0,8575,1)
    allinit=[]
    for i,star in enumerate(stars) :
        j=np.where(extcorr['MANGAID'] == star['MANGAID'])[0]
        bprpc=extcorr['BPRPC'][j]
        star['EBV'] = extcorr['EBV'][j]
        if abs(bprpc) < 5 :
            bounds_lo[i,:] = mod['x_min']
            bounds_hi[i,:] = mod['x_max']
            teff_est= 10.**f(np.max([np.min([bprpc,color[-1]]),color[0]]))
            init[i,j_teff] = teff_est
            if teff_est > 5000. : init[i,j_rot] = 2.3
            if teff_est > 15000. : bounds_lo[i,j_mh] = -1
            print(i,star['MANGAID'],bprpc,init[i,:], len(stars))
        if hmask :
            bd= np.where((star['WAVE']>6563-100)&(star['WAVE']<6563+100) |
                         (star['WAVE']>4861-100)&(star['WAVE']<4861+100) |
                         (star['WAVE']>4341-100)&(star['WAVE']<4341+100) )[0]
            output[i][1][bd] = 1.e-5
        specs.append((output[i][0], output[i][1], init[i,:], (bounds_lo[i,:],bounds_hi[i,:]), order))

    # do the fits in parallel
    if threads==0 :
        output=[]
        for i in range(len(specs)) :
            out=solve(specs[i])
            print(i,stars[i])
            print(out.x)
            if out.x[0]>7000: pdb.set_trace()
            output.append(out)
    else :
        j=np.where(np.core.defchararray.strip(mod['label_names']) == 'LOGG')[0]
        for i,spec in enumerate(specs): 
            specs[i][2][j] = 1.
            print(specs[i][2])
        print('starting pool: ', len(specs))
        pool = mp.Pool(threads)
        output1 = pool.map_async(solve, specs).get()
        pool.close()
        pool.join()
        print('done pool 1')
        for i,spec in enumerate(specs): 
            specs[i][2][j] = 5.
            print(specs[i][2])
        print('starting pool 2: ', len(specs))
        pool = mp.Pool(threads)
        output2 = pool.map_async(solve, specs).get()
        pool.close()
        pool.join()
        print('done pool 2')
        output=[]
        for o1,o2 in zip(output1,output2) :
            print(o1.fun,o2.fun,o1.x,o2.x)
            if o1.fun < o2.fun : output.append(o1)
            else : output.append(o2)

    if mcmc :
        newspecs=[]
        for i,star in enumerate(stars) :
            newspecs.append((specs[i][0],specs[i][1],output[i].x,
                           '{:s}-{:d}-{:s}-{:d}'.format(star['MANGAID'],star['PLATE'],star['IFUDESIGN'],star['MJD'])))

        outmcmc=[]
        if threads== 0 :
            for i,star in enumerate(stars) :
                out=solve_mcmc(newspecs[i])
                outmcmc.append(out)
        else :
            pool = mp.Pool(threads)
            outmcmc = pool.map_async(solve_mcmc, newspecs).get()
            pool.close()
            pool.join()

    # output FITS table
    out=Table()
    out['MANGAID']=stars['MANGAID']
    out['EBV']=stars['EBV']
    try:
        out['OBJRA']=stars['OBJRA']
        out['OBJDEC']=stars['OBJDEC']
        out['PLATE']=stars['PLATE']
        out['IFUDESIGN']=stars['IFUDESIGN']
        out['MJD']=stars['MJD']
        out['MJDQUAL']=stars['MJDQUAL']
    except : pass
    length=len(out)
    params=np.array([o.x for o in output])
    out.add_column(Column(name='FPARAM',data=params))
    bd=np.any( (params>=bounds_hi-0.01*(bounds_hi-bounds_lo)) |
               (params<=bounds_lo+0.01*(bounds_hi-bounds_lo)), axis=1 )
    out.add_column(Column(name='VALID',data=(np.logical_not(bd).astype(int))))
    if pixels == None : out['WAVE']=stars['WAVE']
    else :out['WAVE']=stars['WAVE'][:,pixels[0]:pixels[1]]
    spec=[]
    err=[]
    bestfit=[]
    chi2=[]
    for i,star in enumerate(stars) :
        spec.append(specs[i][0])
        err.append(specs[i][1])
        sfit=spectrum(pix, *params[i])
        bestfit.append(sfit)
        chi2.append(np.nansum((specs[i][0]-sfit)**2/specs[i][1]**2))
    out.add_column(Column(name='SPEC',data=np.array(spec)))
    out.add_column(Column(name='ERR',data=np.array(err)))
    out.add_column(Column(name='SPEC_BESTFIT',data=np.array(bestfit)))
    out.add_column(Column(name='CHI2',data=np.array(chi2)))
    if write : out.write('nn-'+field+'-'+telescope+'.fits',format='fits',overwrite=True)
    return out

def train(file='all_noelem',name='test',plot=False,suffix='',fitfrac=0.5, steps=1e5, weight_decay = 0., num_neurons = [300,300], 
          lr=0.001, ind_label=np.arange(9),pixels=None,
          teff=[0,10000],logg=[-1,6],mh=[-3,1],am=[-1,1],cm=[-2,2],nm=[-2,2],
          raw=True,rot=False,elem=False,normalize=False,elems=None,label_names=None,trim=True,seed=777) :
    """ Train a neural net model on an input training set
    """

    spectra, labels = read(file,raw=raw, label_names=label_names,trim=trim)
 
    if normalize :
        print('normalizing...')
        gdspec=[]
        for i in range(spectra.shape[0]) :
            cont = norm.cont(spectra[i,:],spectra[i,:],poly=False,chips=False,medfilt=400)
            spectra[i,:] /= cont
            if pixels is None : 
                gd = np.where(np.isfinite(spectra[i,:]))[0]
                ntot=len(spectra[i,:])
            else : 
                gd = np.where(np.isfinite(spectra[i,pixels[0]:pixels[1]]))[0]
                ntot=len(spectra[i,pixels[0]:pixels[1]])
            if len(gd) == ntot : gdspec.append(i)
        if pixels is None : spectra=spectra[gdspec,:]
        else : spectra=spectra[gdspec,pixels[0]:pixels[1]]
        labels=labels[gdspec]

    # shuffle them and get fit and validation set
    print('shuffling...')
    shape=labels.shape
    np.random.seed(seed)
    ind_shuffle=np.random.permutation(shape[0])

    #----------------------------------------------------------------------------------------
    # choose only a certain labels

    try :
        gd=np.where((labels[ind_shuffle,0]>=teff[0]) & (labels[ind_shuffle,0]<=teff[1]) &
                    (labels[ind_shuffle,1]>=logg[0]) & (labels[ind_shuffle,1]<=logg[1]) &
                    (labels[ind_shuffle,2]>=mh[0]) & (labels[ind_shuffle,2]<=mh[1]) &
                    (labels[ind_shuffle,3]>=am[0]) & (labels[ind_shuffle,3]<=am[1]) &
                    (labels[ind_shuffle,4]>=cm[0]) & (labels[ind_shuffle,4]<=cm[1])  &
                    (labels[ind_shuffle,5]>=nm[0]) & (labels[ind_shuffle,5]<=nm[1])
                   )[0]
    except :
        gd=np.where((labels[ind_shuffle,0]>=teff[0]) & (labels[ind_shuffle,0]<=teff[1]) &
                    (labels[ind_shuffle,1]>=logg[0]) & (labels[ind_shuffle,1]<=logg[1]) &
                    (labels[ind_shuffle,2]>=mh[0]) & (labels[ind_shuffle,2]<=mh[1]) &
                    (labels[ind_shuffle,3]>=am[0]) & (labels[ind_shuffle,3]<=am[1]) 
                   )[0]
 
    nfit = int(fitfrac*len(gd))
    # separate into training and validation set
    training_spectra = spectra[ind_shuffle[gd],:][:nfit,:]
    training_labels = labels[ind_shuffle[gd],:][:nfit,:][:,ind_label]
    validation_spectra = spectra[ind_shuffle[gd],:][nfit:,:]
    validation_labels = labels[ind_shuffle[gd],:][nfit:,:][:,ind_label]
    model = training.neural_net(training_labels, training_spectra,\
                                validation_labels, validation_spectra,\
                                num_neurons = num_neurons, num_steps=steps, learning_rate=lr, weight_decay=weight_decay)
    model['label_names' ] = label_names
    model['data_file' ] = file
    model['nfit' ] = nfit
    model['ind_shuffle' ] = ind_shuffle[gd]
    model['teff_lim' ] = teff
    model['logg_lim' ] = logg
    model['mh_lim' ] = mh
    model['am_lim' ] = am
    model['cm_lim' ] = cm
    model['nm_lim' ] = nm
    model['learning_rate' ] = lr
    model['weight_decay' ] = weight_decay
    model['num_neurons' ] = num_neurons
    model['steps' ] = steps

    with open('{:s}.pkl'.format(name), 'wb') as f:  
        pickle.dump(model, f, protocol=2)

def read(file,raw=True,label_names=None,trim=True,ids=False) :
    """ Read input spectra and parameters
    """
    tab = Table.read(file+'.fits')
    spectra = tab['SPEC'].data.astype(float)
    if trim :
        gdpix=np.where(np.isfinite(spectra[0,:]))[0]
        spectra=spectra[:,gdpix]
    lab=[]
    if label_names is not None :
        for label in label_names : lab.append(tab[label])
    else :
        for label in tab.meta['LABELS'] : lab.append(tab[label].data)
    labels = np.array(lab).T
    if ids : return spectra, labels, tab['MANGAID'].data
    else : return spectra, labels

    '''
    hdulist = fits.open(file+'.fits')
    if raw : spectra = hdulist[1].data.astype("float")
    else : spectra = hdulist[2].data.astype("float")
    print(spectra.shape)
    if trim :
        gdpix=np.where(np.isfinite(spectra[0,:]))[0]
        spectra=spectra[:,gdpix]
        print(spectra.shape)

    # read labels
    labels = hdulist[0].data
    labels = np.array([labels[i] for i in range(len(labels))])

    try :
        all_label_names=[]
        for i in range(hdulist[0].header['NPAR']) :
            all_label_names.append(hdulist[0].header['PAR{:d}'.format(i)])
        all_label_names=np.array(all_label_names)
    except :
        all_label_names=ascii.read(file).colnames

    if label_names is not None :
        ind_label = []
        for label in label_names :
            j = np.where(np.core.defchararray.strip(all_label_names) == label)[0]
            ind_label.extend(j)
        ind_label = np.array(ind_label)
    else :
        ind_label = np.arange(len(all_label_names))

    if ids :
        return spectra, labels[:,ind_label], hdulist[3].data
    else :
        return spectra, labels[:,ind_label]
    '''

def plot(file='all_noelem',model='GKh_300_0',raw=True,plotspec=False,validation=True,normalize=False,
         pixels=None,teff=[0,10000],logg=[-1,6],mh=[-3,1],am=[-1,1],cm=[-2,2],nm=[-2,2],trim=True,ids=False) :
    ''' plots to assess quality of a model
    '''
    # load model and set up for use
    NN_coeffs = get_model(model)

    # read spectra and labels, and get indices for training and validation set
    if ids :true,labels,iden = read(file,raw=raw,label_names=NN_coeffs['label_names'],trim=trim,ids=ids)
    else : true,labels = read(file,raw=raw,label_names=NN_coeffs['label_names'],trim=trim)
    if normalize :
        print('normalizing...')
        gdspec=[]
        n=0
        for i in range(true.shape[0]) :
            print(i,labels[i])
            cont = norm.cont(true[i,:],true[i,:],poly=False,chips=False,medfilt=400)
            true[i,:] /= cont
            if pixels is None : 
                gd = np.where(np.isfinite(true[i,:]))[0]
                ntot=len(true[i,:])
            else : 
                gd = np.where(np.isfinite(true[i,pixels[0]:pixels[1]]))[0]
                ntot=len(true[i,pixels[0]:pixels[1]])
            if len(gd) == ntot :
                gdspec.append(i)
                n+=1
        print(n,true.shape)
        if pixels is None : true=true[gdspec,:]
        else : true=true[gdspec,pixels[0]:pixels[1]]
        labels=labels[gdspec]
        if ids : iden=iden[gdspec]

    #gd=np.where((labels[:,0]>=teff[0]) & (labels[:,0]<=teff[1]) &
    #            (labels[:,1]>=logg[0]) & (labels[:,1]<=logg[1]) &
    #            (labels[:,2]>=mh[0]) & (labels[:,2]<=mh[1]) &
    #            (labels[:,3]>=am[0]) & (labels[:,3]<=am[1]) &
    #            (labels[:,4]>=cm[0]) & (labels[:,4]<=cm[1])  &
    #            (labels[:,5]>=nm[0]) & (labels[:,5]<=nm[1]) 
    #           )[0]
    #pdb.set_trace()
    #true = true[gd]
    #labels = labels[gd]

    nfit = NN_coeffs['nfit']
    ind_shuffle = NN_coeffs['ind_shuffle']
    true = true[ind_shuffle]
    labels = labels[ind_shuffle]
    if ids : iden=iden[ind_shuffle] 
    if validation:
        true=true[nfit:]
        labels=labels[nfit:]
        if ids: iden=iden[nfit:]
    else :
        true=true[:nfit]
        labels=labels[:nfit]
        if ids: iden=iden[:nfit]

    # loop over the spectra
    if plotspec: plt.figure()
    nn=[]
    diff2=[]
    for i,lab in enumerate(labels) :
        # calculate model spectrum and accumulate model array
        pix = np.arange(8575)
        spec = spectrum(pix, *lab)
        nn.append(spec)
        tmp=np.sum((spec-true[i,:])**2)
        print(i,tmp,lab)
        diff2.append(tmp)
        if plotspec and tmp>100 :
            plt.clf()
            plt.plot(true[i,:],color='g')
            plt.plot(spec,color='b')
            plt.plot(spec-true[i,:],color='r')
            plt.show()
            pdb.set_trace()
        #n=len(np.where(np.abs(apstar[j]-true[i,j]) > 0.05)[0])
    nn=np.array(nn)
    diff2=np.array(diff2)
    #fig,ax=plots.multi(2,2,hspace=0.001,wspace=0.001,sharex=True,sharey=True)
    #plots.plotc(ax[0,0],labels[:,0],labels[:,1],labels[:,2],xr=[8000,3000],yr=[6,-1],zr=[-2.5,0.5])
    #plots.plotc(ax[1,0],labels[:,0],labels[:,1],labels[:,3],xr=[8000,3000],yr=[6,-1],zr=[-0.25,0.5])
    #plots.plotc(ax[1,1],labels[:,0],labels[:,1],diff2,xr=[8000,3000],yr=[6,-1],zr=[0,10])
    #ax[1,1].text(0.,0.9,'diff**2',transform=ax[1,1].transAxes)
    fig,ax=plots.multi(1,1,hspace=0.001,wspace=0.001,sharex=True,sharey=True)
    plots.plotc(ax,labels[:,0],labels[:,1],diff2,xr=[8000,3000],yr=[6,-1],zr=[0,10])
    if ids: 
        data=Table()
        data.add_column(Column(name='ID',data=iden))
        data.add_column(Column(name='TEFF',data=labels[:,0]))
        data.add_column(Column(name='LOGG',data=labels[:,1]))
        data.add_column(Column(name='MH',data=labels[:,2]))
        data.add_column(Column(name='AM',data=labels[:,3]))
        plots._data = data
        plots._id_cols = ['ID','TEFF','LOGG','MH','AM']
    plots.event(fig)
    plt.draw()
    key=' '
    sfig,sax=plots.multi(1,2,hspace=0.001,sharex=True)
    pdb.set_trace()
    print('entering event loop....')
    while key != 'e' and key != 'E' :
        x,y,key,index=plots.mark(fig)
        sax[0].cla()
        sax[0].plot(true[index,:],color='g')
        sax[0].plot(nn[index,:],color='b')
        sax[1].cla()
        sax[1].plot(nn[index,:]/true[index,:],color='g')
        plt.figure(sfig.number)
        plt.draw()

    fig.savefig(file+'_'+model+'.png')

    # histogram of ratio of nn to true
    print("making nn/raw comparison histogram ...")
    # pixels across sample
    fig,ax=plots.multi(2,2,figsize=(12,8))
    # percentiles across wavelength
    fig2,ax2=plots.multi(1,3,hspace=0.001)
    # in parameter space
    fig3,ax3=plots.multi(2,3,hspace=0.001,wspace=0.001)
    for f in [fig,fig2,fig3] :
        if validation : f.suptitle('validation set')
        else : f.suptitle('training set')

    # consider full sample and several bins in Teff and [M/H]
    tbins=[[3000,8000],[3000,4000],[4000,5000],[5000,6000],[3000,4000],[4000,5000],[5000,6000]]
    mhbins=[[-2.5,1.0],[-0.5,1.0],[-0.5,1.0],[-0.5,1.0],[-2.5,-0.5],[-2.5,-0.5],[-2.5,-0.5]]
    names=['all','3000<Te<4000, M/H>-0.5','4000<Te<5000, M/H>-0.5','5000<Te<6000, M/H>-0.5',
                 '3000<Te<4000, M/H<-0.5','4000<Te<5000, M/H<-0.5','5000<Te<6000, M/H<-0.5']
    colors=['k','r','g','b','c','m','y']
    lws=[3,1,1,1,1,1,1]

    for tbin,mhbin,name,color,lw in zip(tbins,mhbins,names,colors,lws) :
        gd = np.where( (labels[:,0] >= tbin[0]) & (labels[:,0] <= tbin[1]) &
                       (labels[:,2] >= mhbin[0]) & (labels[:,2] <= mhbin[1])) [0]
        print(tbin,len(gd))
        if len(gd) > 0 :
            t1=nn[gd,:]
            t2=true[gd,:]

            # differential fractional error of all pixels
            err=(t1-t2)/t2
            hist,bins=np.histogram(err.flatten(),bins=np.linspace(-0.2,0.2,4001))
            plots.plotl(ax[0,0],np.linspace(-0.200+0.005,0.2,4000),hist/hist.sum(),semilogy=True,xt='(nn-true)/true',
                        label=name,xr=[-0.1,0.25],color=color,linewidth=lw)
            ax[0,0].legend(fontsize='x-small')

            # cumulative fractional error of all pixels
            err=np.abs(err)
            hist,bins=np.histogram(err.flatten(),bins=np.logspace(-7,3,501))
            plots.plotl(ax[0,1],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),xt='nn/true',
                        label=name,color=color,linewidth=lw)
            ax[0,1].set_ylabel('Cumulative fraction, all pixels')

            # get percentiles across models at each wavelength
            p=[50,95,99]
            perc=np.percentile(err,p,axis=0)
            npix=perc.shape[1]
            for i in range(3) : 
                plots.plotl(ax2[i],np.arange(npix),perc[i,:],color=color,linewidth=lw,xt='Pixel number')
                ax2[i].text(0.05,0.9,'error at {:d} percentile'.format(p[i]),transform=ax2[i].transAxes)

            # cumulative of 50 and 95 percentile across models
            hist,bins=np.histogram(perc[0,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,0],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,ls=':',linewidth=lw)
            hist,bins=np.histogram(perc[1,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,0],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,linewidth=lw,ls='--')
            hist,bins=np.histogram(perc[1,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,0],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,linewidth=lw)
            ax[1,0].set_ylabel('Cumulative, fraction of pixels')

            # cumulative of 50 and 95 percentile across wavelengths
            p=[50,95,99,100]
            perc=np.percentile(err,p,axis=1)
            hist,bins=np.histogram(perc[0,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,1],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,ls=':',linewidth=lw)
            hist,bins=np.histogram(perc[1,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,1],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,linewidth=lw,ls='--')
            hist,bins=np.histogram(perc[1,:],bins=np.logspace(-7,3,501))
            plots.plotl(ax[1,1],np.logspace(-7,3,500),np.cumsum(hist)/np.float(hist.sum()),color=color,linewidth=lw)
            ax[1,1].set_ylabel('Cumulative, fraction of models')

            for ix,iy in zip([1,0,1],[0,1,1]) :
                ax[iy,ix].set_xlim(0.,0.01)
                ax[iy,ix].set_ylim(0.,1.0)
                ax[iy,ix].set_xlabel('|(nn-true)/true|')
                ax[iy,ix].set_xscale('log')
                ax[iy,ix].set_xlim(1.e-4,0.01)

            # Kiel diagram plots color-coded
            if lw == 3 :
                # color-code by value of 50, 95, and 99 percentile of wavelengths for each model
                p=[50,95,99]
                perc_mod=np.percentile(err,p,axis=1)
                dx=np.random.uniform(size=len(gd))*50-25
                dy=np.random.uniform(size=len(gd))*0.2-0.1
                for i in range(3) :
                    plots.plotc(ax3[i,0],labels[gd,0]+dx,labels[gd,1]+dy,perc_mod[i,:],
                                xr=[8000,3000],yr=[6,-1],zr=[0,0.1],xt='Teff',yt='log g')
                    ax3[i,0].text(0.1,0.9,'error at {:d} percentile'.format(p[i]),transform=ax3[i,0].transAxes)
                # color-code by fraction of pixels worse than 0.01
                for i,thresh in enumerate([0.01,0.05,0.1]):
                    mask=copy.copy(err)
                    mask[mask<=thresh] = 0
                    mask[mask>thresh] = 1
                    bdfrac=mask.sum(axis=1)/mask.shape[1]
                    axim=plots.plotc(ax3[i,1],labels[gd,0]+dx,labels[gd,1]+dy,bdfrac,
                                xr=[8000,3000],yr=[6,-1],zr=[0,0.1],xt='Teff')
                    ax3[i,1].text(0.1,0.9,'Fraction of pixels> {:4.2f}'.format(thresh),transform=ax3[i,1].transAxes)
                cax = plt.axes([0.05, 0.03, 0.9, 0.02])
                fig3.colorbar(axim,cax=cax,orientation='horizontal')

    fig.tight_layout()
    plt.draw()
    fig.savefig(file+'_'+model+'_1.png')
    fig2.savefig(file+'_'+model+'_2.png')
    fig3.savefig(file+'_'+model+'_3.png')
    pdb.set_trace()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    return nn, true, labels

if __name__ == '__main__' :

    #train( name='alllo', raw=False)
    #train( name='allhi', raw=True)
    train( teff=[3000,4000], mh=[-0.5,0.75] , name='Mhlo', raw=False)
    train( teff=[3000,4000], mh=[-0.5,0.75] , name='Mhhi', raw=True)
    #train( teff=[3500,6000], mh=[-0.5,0.75] , name='GKhlo', raw=False)
    #train( teff=[3500,6000], mh=[-0.5,0.75] , name='GKhhi', raw=True)
    #train( teff=[5500,8000], mh=[-0.5,0.75] , name='Fhlo', raw=False)
    #train( teff=[5500,8000], mh=[-0.5,0.75] , name='Fhhi', raw=True)
