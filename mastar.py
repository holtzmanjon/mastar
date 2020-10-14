# take MaStar-APOGEE x-matched file and stack spectra
#  of individual objects, removing those with any bad pixels
#  in the stack (after trimming ends)

# write into 3-extension fits files as input for NN train

import pdb
import nn
from astropy.io import fits
from astropy.table import Table, Column, vstack
#from tools import match
#from tools import plots
#from tools import html
import numpy as np
import scipy.signal
from scipy.ndimage import median_filter
import scipy.linalg
import os
from esutil import htm
import matplotlib.pyplot as plt
#from apogee.utils import apselect
#from apogee.utils import spectra
#from apogee.aspcap import ferre
#from apogee.aspcap import norm
import matplotlib.patches as patches
import multiprocessing as mp


from spec import rotate

def clean(mastar,plot=True,nomask=False) :
    """ return indices of "good" stars, removing
        those with MJDQUAL&bdmask with nomask=False
        and auto-detected red-upturn stars
    """
    # remove masked bad
    if nomask :
        gd=[]
    else :
        bdmask=2**1+2**4+2**5+2**6+2**7+2**8+2**9
        gd=np.where(mastar['MJDQUAL']&bdmask == 0)[0].tolist()

    # remove red upturn
    fit=np.where((mastar['WAVE'][0,:] > 8500) & (mastar['WAVE'][0,:] < 9000) )[0]
    test=np.where((mastar['WAVE'][0,:] > 9500) )[0]
    if plot : plt.clf()
    bd=[]
    for i in range(0,len(mastar),10) :
        par=np.polyfit(mastar['WAVE'][i,fit],np.log(mastar['FLUX'][i,fit]),1)
        fitflux = par[0]*mastar['WAVE'][i,test]+par[1]
        print(i,mastar['FLUX'][i,test].sum(), 1.10*np.exp(fitflux).sum(),len(fitflux),len(test))
        if mastar['FLUX'][i,test].sum() > 1.07*np.exp(fitflux).sum() :
            if plot :
                plt.clf()
                plt.plot(mastar['WAVE'][i,test],np.exp(fitflux))
                plt.plot(mastar['WAVE'][i,:],mastar['FLUX'][i,:])
                plt.draw()
                bd.append(i)
                pdb.set_trace()
        else : gd.append(i)
    print('bad: ', bd)
    return list(set(gd))

def stack(infile='mastar-goodspec-v2_7_1-trunk',apogee='r13/l33/allStar-r13-l33-58932',out='mastar_apogee_stack',cm=False) :
    """ Create list of MaStar objects with matching APOGEE spectra
        and stack multiple observations into a single
        Add in model spectra for warmer stars
    """
    # read MaStar file and filter
    mastar=fits.open(infile+'.fits')[1].data
    #gd = clean(mastar,plot=False)
    #mastar=mastar[gd]

    # read APOGEE file and filter
    apogee=fits.open(os.environ['APOGEE_ASPCAP']+'/'+apogee+'.fits')[1].data
    gd=apselect.select(apogee,badval=['STAR_BAD'])
    apogee=apogee[gd]

    # get matches within 3 arcsec
    h=htm.HTM()
    maxrad=3./3600.
    m1,m2,rad=h.match(apogee['RA'],apogee['DEC'],mastar['OBJRA'],mastar['OBJDEC'],maxrad,maxmatch=10)

    # other stars to remove, e.g., based on bad modeling in initial model
    bdstars=[]
    """
    bdstars=['3-15080992','3-105812093','7-12951805',
             '3-142131061','60-36093244935984954','3-126079790',
             '60-36091793237038712','3-126306258','3-24068365',
             '3-23593772','60-36093244935984954',
             '60-36091793237038712',
             '3-140608451', '7-14409160', '3-114437013', '7-27039482',
             '3-15080992', '3-148139562', '3-143029659', '3-103574358',
             '3-139462126', '3-136745748', '3-142131061',
             '3-142325470', '3-137769951', '3-141476933', '3-23120750',
             '60-3615860471750191104', '7-17756858', '3-49507807',
             '3-36341647', '3-22166243', '7-9789478', '3-124777940',
             '60-676895989836309632', '3-141873923', '3-140806405',
             '3-21696715', '7-7407046', '3-126309023', '3-125275478'],
    """

    allbd=[]
    ngd=0
    nlo=0
    lo=[]
    spec=[]
    param=[]
    fparam=[]
    name=[]
    stars=set(mastar['MANGAID'])

    nrot=5
    vsinis = 1.+np.arange(nrot)*np.log10(2.5)
    deltav=(np.log(mastar['WAVE'][0,1])-np.log(mastar['WAVE'][0,0]))*3.e5
    for star in stars :
        if star in bdstars : continue
        j = np.where(mastar['MANGAID'] == star)[0]
        print(star,len(j))
        # create weighted average spectrum
        num = np.sum(mastar['FLUX'][j,:]*mastar['IVAR'][j,:],axis=0)
        den = np.sum(mastar['IVAR'][j,:],axis=0)
        avg = num/den
        err = 1/den

        sn=avg/err
        # where S/N<2, set flux to 0
        faint = np.where(scipy.signal.medfilt(sn,21) < 2.)[0]
        avg[faint] = 0.
        if (sn[200:-200].min() < 10) :
            #print(star,apogee['PARAM'][j[0],:])
            lo.append(j[0])
            nlo+=1
   
        # only take spectra with no bad pixels (neglecting first and last 10)
        bd=np.where(np.isinf(err[10:-10]))[0]
        if len(bd) < 1 :
            allbd.extend(bd.tolist())
            spec.append(avg)
            par=apogee['FPARAM'][j[0],:]
            if cm : fparam.append([par[0],par[1],par[3],par[6],par[7],par[4]])
            else : fparam.append([par[0],par[1],par[3],par[6],par[7]])
            par=apogee['PARAM'][j[0],:]
            elem=apogee['X_M'][j[0],:]
            if cm : param.append([par[0],par[1],par[3],elem[5],par[7],par[4]])
            else : param.append([par[0],par[1],par[3],elem[5],par[7]])
            name.append(star)
            ngd+=1
            # if a slow rotator, broaden to accommodate different vsini
            if 10.**par[7] < 10. :
                for irot,vsini in enumerate(vsinis[1:]) :
                    print(vsini)
                    rot=rotate(deltav,10.**vsini)
                    conv=np.convolve(avg,rot,mode='same')
                    spec.append(conv)
                    if cm : fparam.append([par[0],par[1],par[3],par[6],vsini,par[4]])
                    else : fparam.append([par[0],par[1],par[3],par[6],vsini])
                    if cm : param.append([par[0],par[1],par[3],elem[5],vsini,par[4]])
                    else : param.append([par[0],par[1],par[3],elem[5],vsini])
                    name.append('{:s}_{:.1f}'.format(star,vsini))
  
    param=np.array(param)
    fparam=np.array(fparam)

    print('Number of good stars: ', ngd)
    print('Number of bad spectra: ',len(allbd),len(set(allbd)))

    # plots
    fig,ax=plots.multi(2,2)
    plots.plotc(ax[0,0],param[:,0],param[:,1],param[:,2],xr=[8000,3000],yr=[6,-1],zr=[-2,0.5],xt='Teff',yt='logg',zt='[M/H]',colorbar=True)
    ax[0,0].text(0.1,0.9,'[M/H]',transform=ax[0,0].transAxes)
    plots.plotc(ax[1,0],param[:,0],param[:,1],param[:,3],xr=[8000,3000],yr=[6,-1],zr=[-0.5,0.5],xt='Teff',yt='logg',zt='[alpha/M]',colorbar=True)
    ax[1,0].text(0.1,0.9,'[alpha/M]',transform=ax[1,0].transAxes)
    plots.plotc(ax[0,1],param[:,2],param[:,3],param[:,0],zr=[3500,6000],xr=[-2.5,1],yr=[-0.5,0.5],zt='Teff',xt='[M/H]',yt='[alpha/M]',colorbar=True)
    ax[1,0].text(0.1,0.9,'[alpha/M]',transform=ax[1,0].transAxes)
    if cm: 
        plots.plotc(ax[1,1],param[:,0],param[:,1],param[:,5],xr=[8000,3000],yr=[6,-1],zr=[-0.5,0.5],xt='Teff',yt='logg',zt='[C/M]',colorbar=True)
        ax[1,1].text(0.1,0.9,'[C/M]',transform=ax[1,1].transAxes)
    #plots.plotc(ax[0,1],fparam[:,0],fparam[:,1],fparam[:,2],xr=[8000,3000],yr=[6,-1],zr=[-2,0.5],xt='uncal Teff',yt='uncal logg',zt='uncal [M/H]')
    #plots.plotc(ax[1,1],fparam[:,0],fparam[:,1],fparam[:,3],xr=[8000,3000],yr=[6,-1],zr=[-0.5,0.5],xt='uncal Teff',yt='uncal logg',zt='uncal [alpha/H]')
    fig.tight_layout()
    fig.savefig(out+'.png')

    #output FITS file
    tab=Table()
    tab.add_column(Column(name='TEFF',data=param[:,0]))
    tab.add_column(Column(name='LOGG',data=param[:,1]))
    tab.add_column(Column(name='[M/H]',data=param[:,2]))
    tab.add_column(Column(name='[alpha/M]',data=param[:,3]))
    tab.add_column(Column(name='LOG(VSINI)',data=param[:,4]))
    tab.add_column(Column(name='MANGAID',data=np.array(name)))
    if cm: tab.add_column(Column(name='[C/M]',data=param[:,5]))
    tab.add_column(Column(name='SPEC',data=np.array(spec)))
    tab.meta['LABELS'] = ['TEFF','LOGG','[M/H]','[alpha/M]','LOG(VSINI)']
    tab.write(out+'.fits',overwrite=True)

    # add in model spectra
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    f=[]
    v=[]
    for file in ['f_nsc2','f_nsc3','f_nsc4','f_nsc5'] :
        synth = fits.open(file+'_rot.fits')[0]
        for irot,vsini in enumerate(spectra.fits2vector(synth.header,5)) :
            for imh,mh in enumerate(spectra.fits2vector(synth.header,4)) :
                if mh < -1 : continue
                for iteff,teff in enumerate(spectra.fits2vector(synth.header,3)) :
                    if teff < 6000 : continue
                    for ilogg,logg in enumerate(spectra.fits2vector(synth.header,2)) :
                        if mh>=-2.5 :
                            a.append(teff)
                            b.append(logg)
                            c.append(mh)
                            d.append(0.)
                            v.append(vsini)
                            e.append(file+'_{:.0f}_{:.1f}_{:.1f}'.format(teff,logg,mh))
                            f.append(synth.data[irot,imh,iteff,ilogg,:]/np.median(synth.data[irot,imh,iteff,ilogg,:]))
   
    synthtab=Table()
    synthtab.add_column(Column(name='TEFF',data=np.array(a)))
    synthtab.add_column(Column(name='LOGG',data=np.array(b)))
    synthtab.add_column(Column(name='[M/H]',data=np.array(c)))
    synthtab.add_column(Column(name='[alpha/M]',data=np.array(d)))
    synthtab.add_column(Column(name='LOG(VSINI)',data=np.array(v)))
    synthtab.add_column(Column(name='MANGAID',data=np.array(e)))
    if cm: synthtab.add_column(Column(name='[C/M]',data=np.array(d)))
    synthtab.add_column(Column(name='SPEC',data=np.array(f)))

    vstack([tab,synthtab]).write(out+'_synth.fits',overwrite=True)
    #p=fits.PrimaryHDU(np.array(param))
    #p.header['NPAR'] = 4
    #p.header['PAR0'] = 'Teff'
    #p.header['PAR1'] = 'logg'
    #p.header['PAR2'] = '[M/H]'
    #p.header['PAR3'] = '[alpha/M]'
    #if cm : p.header['PAR4'] = '[C/M]'
    #hdulist.append(p)
    #hdulist.append(fits.ImageHDU(np.array(spec)))
    #hdulist.append(fits.ImageHDU(np.array(spec)))
    #col=fits.Column(name='MANGAID',array=np.array(name),format='A20')
    #hdulist.append(fits.TableHDU.from_columns([col]))
    #hdulist.writeto(out+'.fits',overwrite=True)


def plot(file='nn-mastar-goodspec-v2_7_1-trunk-new_stack600',goodspec='mastar-goodspec-v2_7_1-trunk') :
    """ Plot results
    """

    pdb.set_trace()
    good=fits.open(goodspec+'.fits')[1].data
    out=fits.open(file+'.fits')[1].data
    bdmask=2**1+2**4+2**5+2**6+2**7+2**8+2**9
    gd=np.where((out['MJDQUAL']&bdmask == 0) & (out['VALID'] ==1))[0]
    bd=np.where(out['MJDQUAL']&bdmask != 0)[0]
    out=out[gd]
    good=good[gd]

    # get extinction/distance corrected GAIA mags
    gaia=fits.open('trunk/goodstars-v2_7_1-gaia-extcorr.fits')[1].data

    # match output and GAIA
    m1,m2=match.match(out['MANGAID'],gaia['MANGAID'])
    unq,udx,idx=np.unique(out['MANGAID'],return_inverse=True,return_index=True)

    # get GAIA mags for each spectrum
    mg=gaia[m2[idx]]['M_G']
    br=gaia[m2[idx]]['BPRPC']
    gd=np.where(br>-99)[0]
    print('len(gd): ', len(m1),len(gd))

    # GAIA plot
    fig,ax=plots.multi(1,1)
    plots.plotc(ax,br[gd],mg[gd],out['FPARAM'][m1[idx[gd]],2],xr=[0,3],yr=[12,-5],zr=[-2.5,0.5],
                size=1,colorbar=True,xt='BP-RP',yt='M_G',zt='[M/H]')

    # plots
    fig,ax=plots.multi(2,2)
    plots.plotc(ax[0,0],out['FPARAM'][m1[idx[gd]],0],out['FPARAM'][m1[idx[gd]],1],br[gd],xr=[8000,3000],yr=[6,-1],zr=[0,3],
                xt='Teff',yt='logg',zt='BP-RP',size=1,colorbar=True)
    plots.plotc(ax[0,1],out['FPARAM'][gd,0],out['FPARAM'][gd,1],out['FPARAM'][gd,2],xr=[8000,3000],yr=[6,-1],zr=[-2,0.5],
                xt='Teff',yt='logg',zt='[M/H]',size=1,colorbar=True)
    plots.plotc(ax[1,0],out['FPARAM'][gd,2],out['FPARAM'][gd,3],out['FPARAM'][gd,0],zr=[3000,7000],yr=[-0.5,1],xr=[-2,0.5],
                xt='[M/H]',yt='[alpha/M]',zt='Teff',size=1,colorbar=True)
    plots.plotc(ax[1,1],out['FPARAM'][gd,0],out['FPARAM'][gd,1],out['CHI2'][gd]/3763.,xr=[8000,3000],yr=[6,-1],zr=[0,10],
                size=1,colorbar=True,xt='Teff',yt='logg',zt='CHI2')
    plots._data = out
    plots._id_cols=['MANGAID']
    plots.event(fig)
    fig.tight_layout()
    fig_spec,ax_spec=plots.multi(1,3,hspace=0.001)
    key=' '
    while key != 'e' :
        x,y,key,index = plots.mark(fig,index=True)
        index=gd[index]
        print(x,y,key,index,out['FPARAM'][index,:])
        for i in range(3) : ax_spec[i].cla()
        plots.plotl(ax_spec[0],out['WAVE'][index],out['SPEC'][index],color='g')
        ax_spec[0].text(0.05,0.9,'{:7.1f}{:6.1f}{:6.1f}{:6.1f}'.format(*out['FPARAM'][index]),transform=ax_spec[0].transAxes)
        plots.plotl(ax_spec[0],out['WAVE'][index],out['SPEC_BESTFIT'][index],color='b')
        plots.plotl(ax_spec[1],out['WAVE'][index],out['SPEC'][index]/out['SPEC_BESTFIT'][index],yr=[0.5,1.5])
        plots.plotl(ax_spec[2],out['WAVE'][index],good['FLUX'][index,600:-200])
        plt.figure(fig_spec.number)
        plt.draw()

    plt.show()


init = False

def comp_setup() :
    """ Setup for spectral comparison
    """
    global wave, hdr, lib, mod, init

    if not init :
        hdr=ferre.rdlibhead('capcrop.hdr')
        wave=hdr['WAVE'][0]+np.arange(hdr['NPIX'])*hdr['WAVE'][1]

        lib=np.fromfile('capcrop.unf',dtype=np.float32)
        lib=np.reshape(lib,(15,9,5,11,11,4563))

        mod=nn.get_model('stack600')

def comp(teff=5000,logg=5,mh=0,am=0,vmicro=np.log10(1.),clear=True,hard=None) :
    """ Compare NN model spectra with library spectra
    """
    comp_setup()

    lab=[teff,logg,mh,am]
    pix = np.arange(4563)
    spec = nn.spectrum(pix, *lab)
    inds=np.round(([mh,am,vmicro,teff,logg]-hdr['LLIMITS'])/hdr['STEPS']).astype(int)
    synth = lib[inds[0],inds[1],inds[2],inds[3],inds[4],600:-200]
    synth = synth / norm.cont(synth,synth,medfilt=400,order=0,poly=False)

    fig,ax=plots.multi(1,2,hspace=0.001)
    plots.plotl(ax[0],10.**wave[600:-200],spec)
    plots.plotl(ax[0],10.**wave[600:-200],synth,yr=[0,2])
    plots.plotl(ax[1],10.**wave[600:-200],spec/synth,yr=[0.8,1.2])
    if hard is None :
        plt.draw()
    else :
        fig.savefig(hard+'.png')
        plt.close()

def comp_table() :

    grid=[]
    yt=[]
    for mh in np.arange(-2.5,1.0,1.0) :
        for logg in np.arange(0,5.5,1.) :
            row=[]
            xt=[]
            for teff in np.arange(3500,6500,500) :
                xt.append('Teff: {:d}'.format(teff))
                name='comp{:d}_{:.1f}_{:.1f}'.format(teff,logg,mh)
                comp(teff=teff,logg=logg,mh=mh,hard=name)
                row.append(name+'.png')
            grid.append(row)
            yt.append('logg: {:.1f}  [M/H]: {:.1f}'.format(logg,mh))

    html.htmltab(grid,file='comp.html',xtitle=xt,ytitle=yt)
                

def clusters():

    a=fits.open('trunk/mastarall-v2_7_1.fits')
    tab=Table(a['GOODVISITS'].data)
    tab['B']=tab['PSFMAG'][:,1]
    tab['R']=tab['PSFMAG'][:,3]
    apselect.clusters(tab,ratag='OBJRA',dectag='OBJDEC',
                      rvtag='HELIOV',idtag='MANGAID',btag='B',rtag='R')


def inspect(out,chi2=False,ebv=False,zr=None,model='new_apogee_stack_synthallrot600') :

    # Read median parameter catalog
    a=Table.read(out)

    # set up the plots with Kiel diagram
    fig_cmd,ax_cmd=plots.multi(1,1,hspace=0.001,wspace=0.002,sharex=True,sharey=True)
    if chi2 :
        z = a['CHI2']
        if zr is None : zr= [0,10000]
        zt = 'CHI2'
    elif ebv :
        z = a['EBV']
        if zr is None : zr= [0,1]
        zt = 'EBV'
    else :
        z = a['FPARAM'][:,2]
        if zr is None : zr= [-2.5,0.5]
        zt='[M/H]'
    plots.plotc(ax_cmd,a['FPARAM'][:,0],a['FPARAM'][:,1],z,xr=[31000,3000],yr=[6,-1],zr=zr,xt='Teff',yt='log g',colorbar=True,zt=zt)
    ax_cmd.set_xscale('log')
    plots._data=a
    plots._id_cols=['MANGAID','PLATE','IFUDESIGN','MJD','FPARAM']

    # load model (for mcmc) 
    if model is not None : mod = nn.get_model(model)

    # set up plot for spectra and best fits
    sfig,sax=plots.multi(1,2,hspace=0.001,wspace=0.002,sharex=True,sharey=True,figsize=(12,2))

    # start event handler and handle events
    plt.show()
    plots.event(fig_cmd)
    key=' '
    point = None
    while (key != 'e' and key !='E') :
        x,y,key,index = plots.mark(fig_cmd)
        if key == 'e' or key == 'E' : break
        print(x,y,key)
        if point is not None : 
            # remove previous marked point
            for p in point_cmd : p.remove()
            for t in text_cmd : t.remove()
            for p in point : p.remove()
            for t in text : t.remove()
            for t in tit : t.set_visible(False)
            for t in tit_cmd :t.set_visible(False)
        rad=10
        color='k'
        point=[]
        point_cmd=[]
        text=[]
        text_cmd=[]
        tit=[]
        tit_cmd=[]
        # given marked point, annotate the plots with info for this object
        tit_cmd.append(fig_cmd.suptitle('PLATE: {:d}  IFUDESIGN: {:s}  MJD: {:d}  INDEX: {:d}'.format(
                   a['PLATE'][index],a['IFUDESIGN'][index],a['MJD'][index],index)))
        plt.draw()

        # given marked point, draw a big point on all plots to locate it
        marker='s'
        point_cmd.append(ax_cmd.scatter([a['FPARAM'][index,0]],[a['FPARAM'][index,1]],s=40,marker=marker,
                         c=[z[index]],cmap='rainbow',vmin=zr[0],vmax=zr[1],edgecolors='k'))
        text_cmd.append(ax_cmd.text(0.95,0.9,'{:8.1f}{:6.1f}{:6.1f}{:8.0f}'.format(a['FPARAM'][index,0],a['FPARAM'][index,1],a['FPARAM'][index,2],a['CHI2'][index]),
                        transform=ax_cmd.transAxes,ha='right'))
        plt.figure(fig_cmd.number)
        plt.draw()

        # plot spectra, if you just replace previous data, then interactive limits are preserved!
        j=index
        try :
            l1.set_ydata(a['SPEC'][j,:])
            l2.set_ydata(a['SPEC_BESTFIT'][j,:])
            l2e.set_ydata(a['ERR'][j,:])
            l3.set_ydata(a['SPEC'][j,:]/a['SPEC_BESTFIT'][j,:])
        except :
            l1,=plots.plotl(sax[0],a['WAVE'][j,:],a['SPEC'][j,:],xr=[3600,10300],yr=[0,2],color='g')
            l2,=plots.plotl(sax[0],a['WAVE'][j,:],a['SPEC_BESTFIT'][j,:],xr=[3600,10300],yr=[0,2],color='b')
            l2e,=plots.plotl(sax[0],a['WAVE'][j,:],a['ERR'][j,:],xr=[3600,10300],yr=[0,2],color='b')
            l3,=plots.plotl(sax[1],a['WAVE'][j,:],a['SPEC'][j,:]/a['SPEC_BESTFIT'][j,:],
                        xr=[3600,10300],yr=[0.5,1.5],color='b')
             
        plt.figure(sfig.number)
        plt.draw()

        if key == 'm' :
            pdb.set_trace() 
            nn.solve_mcmc((a['SPEC'][j,:],a['ERR'][j,:],a['FPARAM'][j,:],a['MANGAID'][j]),eps=0.05)

def gauss(x,m,s) :
    return 1./s/np.sqrt(2.*np.pi) * np.exp(-0.5*(x-m)**2/s**2)

def homogenize_stack(file='mastar-goodspec-v3_0_1-v1_5_0',outfile=None,plot=False,nobj=0,threads=8) :
    """ LSF-homogenize a set of spectra
    """

    mastar=fits.open(file+'.fits')[1].data
    pixdisp = mastar['WAVE'][0,:]-np.roll(mastar['WAVE'][0,:],1)

    # filter for LSF
    pdb.set_trace()

    try:
        disp = fits.open('mastar_median_disp.fits')[0].data
    except: 
        disp=[] 
        for  i in range(len(mastar)) :
            print(i)
            disp.append(median_filter(mastar['DISP'][i,:],size=101))
        disp=np.array(disp)

    gdpix = np.where(( (mastar['WAVE'][0,:] > 3900) & (mastar['WAVE'][0,:] < 5500) ) |
                  ( (mastar['WAVE'][0,:] > 6500) & (mastar['WAVE'][0,:] < 7800) ) |
                  ( (mastar['WAVE'][0,:] > 8200) & (mastar['WAVE'][0,:] < 9800) ) ) [0]

    perc=[]
    gd=[]
    fig,ax=plots.multi(1,3,hspace=0.001)
    igd=-2
    for p in [25,50,75,90,95,97,98,99] :
    #for p in [98] :
        pp=np.percentile(disp,p,axis=0)
        out=np.all(disp[:,gdpix]<pp[gdpix],axis=1)
        gd.append(np.where(out)[0])
        ngd=len(np.where(out)[0])
        plots.plotl(ax[0],mastar['WAVE'][0,:],pp,yt='disp',label='{:d}% : {:d}/{:d}'.format(p,ngd,len(mastar)))
        plots.plotl(ax[1],mastar['WAVE'][0,:],mastar['WAVE'][0,:]/(pp*2.354),yt='R')
        plots.plotl(ax[2],mastar['WAVE'][0,:],pp/pixdisp,yt='pixels')
        print(p,ngd)
        perc.append(pp)
    ax[0].legend(fontsize='xx-small')
    fig.savefig('mastar_median_lsf.png')
    pdb.set_trace()

    # target LSF
    outsig=median_filter(perc[igd]/pixdisp,size=31)
    plots.plotl(ax[2],mastar['WAVE'][0,:],outsig,linewidth=5,color='k')
    mastar=mastar[gd[0]]

    # get unique stars, and create output
    stars = set(mastar['MANGAID'])
    out_dtype = np.dtype([('MANGAID','S64'),('OBJRA',float),('OBJRA',float),('FLUX',(float,4563)),('IVAR',(float,4563)),
                         ('WAVE',(float,4563))])
    out = np.zeros(len(stars),dtype=out_dtype)

    pars=[]
    for i,star in enumerate(stars) :
        if nobj > 0 and i>nobj : break
        j = np.where(mastar['MANGAID'] == star)[0]
        for jj in j :
            sig = median_filter(mastar['DISP'][jj]/pixdisp,size=31)
            pars.append((mastar['WAVE'][jj],mastar['FLUX'][jj],sig,outsig))

   # now do the homogenization, in parallel if requested
    if threads == 0 :
        output=[]
        for par in pars :
            print(par)
            output.append(do_homogenize(par))
    else :
        print('running pool')
        pool = mp.Pool(threads)
        output = pool.map_async(do_homogenize, pars).get()
        pool.close()
        pool.join()
        print('done pool')

    ii=0 
    for i,star in enumerate(stars) :
        if nobj > 0 and i>nobj : break
        j = np.where(mastar['MANGAID'] == star)[0]
        if plot :fig,ax=plots.multi(1,3,hspace=0.001,sharex=True)

        for jj in j :
            if plot :
                plots.plotl(ax[0],mastar['WAVE'][jj],norm(mastar['FLUX'][jj]),yr=[0.5,1.5])
                sig = median_filter(mastar['DISP'][jj]/pixdisp,size=31)
                plots.plotl(ax[2],mastar['WAVE'][jj],mastar['DISP'][jj]/pixdisp)
                plots.plotl(ax[2],mastar['WAVE'][jj],sig,color='k')
                plots.plotl(ax[2],mastar['WAVE'][jj],sig/sig*1.2,color='r')
            mastar['FLUX'][jj] = output[ii]
            if plot : plots.plotl(ax[1],mastar['WAVE'][jj],norm(mastar['FLUX'][jj]))
            ii += 1

        num = np.sum(mastar['FLUX'][j,:]*mastar['IVAR'][j,:],axis=0)
        den = np.sum(mastar['IVAR'][j,:],axis=0)
        avg = num/den
        err = 1/den
        try :
            out['MANGAID'][i] = star
            out['OBJRA'][i] = mastar['OBJRA'][j[0]]
            out['OBJDEC'][i] = mastar['OBJDEC'][j[0]]
            out['FLUX'][i] = avg
            out['IVAR'][i] = den**2
            out['WAVE'][i] = mastar['WAVE'][j[0]]
            if plot :
                plots.plotl(ax[1],out['WAVE'][i],norm(out['FLUX'][i]),color='k')
                pdb.set_trace()
                plt.close()
        except: pdb.set_trace()

    out=Table(out)
    out.write(outfile,overwrite=True)
    return out

def norm(spec) :
    return spec / median_filter(spec,size=201)

def do_homogenize(pars,plot=False) :
    """ homogenize a single spectrum
    """
    wave=pars[0]
    spec=pars[1]
    disp=pars[2]
    outsig=pars[3]
    nlsf = 9
    ab = np.zeros([2*nlsf+1,len(spec)])
    for x in np.arange(len(spec)) :
        sig=np.min([np.max([disp[x],0.1]),10.])
        y=gauss(np.arange(-nlsf,nlsf+1),0,sig)
        ab[:,x] = y

    med=scipy.signal.medfilt(spec,201)
    if plot :
        fig,ax=plots.multi(1,2,hspace=0.001,sharex=True)
        plots.plotl(ax[0],wave,spec/med,yr=[-0.5,3])
        plots.plotl(ax[2],wave,disp)
    try:
        out=scipy.linalg.solve_banded((nlsf,nlsf), ab, spec)
        #kernel=gauss(np.arange(-nlsf,nlsf+1),0,outsig)
        #new=np.convolve(out,kernel,mode='same')
        new=np.full_like(out,0.)
        for i in range(nlsf,len(new)-nlsf) :
            kernel=gauss(np.arange(-nlsf,nlsf+1),0,outsig[i])
            new[i] = np.sum(out[i-nlsf:i+nlsf+1]*kernel[::-1])
        
        if plot: 
            plots.plotl(ax[1],wave,new/med)
            pdb.set_trace()
            plt.close()
    except: 
        print('failed')
        new = None
 
    return new

