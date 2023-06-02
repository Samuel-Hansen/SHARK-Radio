
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Size plots"""

import functools
import numpy as np
from pylab import scatter
import pylab
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import common
import utilities_statistics as us
import pandas as pd
import matplotlib as mpl
from collections import OrderedDict
from matplotlib.pyplot import cm
import time
import cmocean.cm as cmo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patheffects as pe
import statistics as stat
import cmocean
from scipy.optimize import curve_fit

from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM

mpc_to_cm = 3.086e24

cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121, Tcmb0=2.725)

d10pc = 3.086e+19 #10 parsecs in cm
dfac10pc = 4 * np.pi * d10pc**2


cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121, Tcmb0=2.725)
#zlist = [0]
zlist = [0,0.194738848008908]
#zlist = [0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501]
#0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501,5.02220991014863]
#zlist = [0,0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501,5.02220991014863] #5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086, 8.94312532315157, 9.95650268434316]
##################################
#Constants
RExp     = 1.67
MpcToKpc = 1e3
G        = 4.299e-9 #Gravity constant in units of (km/s)^2 * Mpc/Msun
c_speed  = 299792458.0 #m/s
PI       = 3.141592654

mlow = 6.5
mupp = 12.5
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0
Lsun = 3.828 * 10**26 #W
msun = -26.74

## AGN Constants

alpha_adaf = 0.1
delta = 0.0005
beta = 1 - alpha_adaf / 0.55
eta_edd = 4
mcrit_nu = 0.001 * (delta / 0.0005) * (1 - beta) / beta * alpha_adaf**2
spin = 0.1

A_adaf = 5e-4
A_td = A_adaf/100

def prepare_data(hdf5_data, seds, seds_bands, seds_bc, index, LFs_dust, obsdir):
    
    (h0, volh, mdisk, mbulge, sfr_disk, sfr_burst, typ,mgas_metals_bulge, mgas_metals_disk,mgas_bulge,mgas_disk,rgas_disk,rstar_disk,matom_disk,mmol_disk,galaxy_id, mbh,mbh_acc_hh, mbh_acc_sb) = hdf5_data
    bin_it = functools.partial(us.wmedians, xbins=xmf)

    lir_disk = seds[0]
    lir_bulge = seds[1]
    lir_total = seds[2] #total IR luminosity in units of Lsun

    lir_cont_bc = seds_bc[1]

    lir_total = np.array(lir_total, dtype=np.float64)
    Tbc = 50.0
    Tdiff = 25.0

    
    Teff = Tbc * lir_cont_bc[0] + Tdiff * (1 - lir_cont_bc[0]) #check if fraction
    
    #luminosity with dust temperature

    lir_total_W = lir_total[0] * Lsun  #units of W
    seds_disk = seds_bands[0]
    seds_bulge = seds_bands[1]
    seds_total = seds_bands[2] 
     
    band_14 = seds_total[12,:] #the absolute magnitude of the 1.4 GHz band
    band_30 = seds_total[13,:] #the absolute magnitude of the 3.0 GHz band


    L1p4radio = seds_total[12,:]
    #compute luminosity function

    ind = np.where((lir_total > 0) & (lir_total < 1e20))
    H, bins_edges = np.histogram(np.log10(lir_total[ind]),bins=np.append(mbins,mupp))
    LFs_dust[index,:] = LFs_dust[index,:] + H

    #median Lradio vs LIR
    lir_total = lir_total[0,:]
    ind = np.where((lir_total > 1e6) & (lir_total < 1e20))
    lir_selected = lir_total[ind]
    lradio_selected = L1p4radio[ind] 
   # print(lir_selected.shape, lradio_selected.shape)
    meds_radio = bin_it(x=lir_selected, y=lradio_selected)
  
    return(volh, h0, band_14, band_30, lir_total_W,Teff)
    
def dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst,fields):
    h0 = float(h0)

    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m

    parm_lst = fields['galaxies']
    
    df = pd.DataFrame(columns = parm_lst)

    
    print('Creating Data Frame')
    for z in range(len(zlist)):
        df_temp = pd.DataFrame(columns = parm_lst)
        mstars_tot_array = hdf5_lir_lst[z][2] + hdf5_lir_lst[z][3]
        ind0 = np.where(mstars_tot_array > 0)
        print("Creating data for z = ",str(round(z,2)))
        for i in range(len(parm_lst)):
            idx = i + 2
            parm = parm_lst[i]

            parm_ary = np.array(hdf5_lir_lst[z][idx],dtype=np.float64)
            parm_ary = parm_ary[ind0]
            df_temp[parm] = list(parm_ary)
        z_array=np.empty(len(parm_ary))
        z_array.fill(zlist[z])
        df_temp['z'] = z_array

        df_temp['teff'] = Teff_lst[z]
        df_temp['lir_w'] = lir_total_W_lst[z]

        ir_mags = seds_bands_lst[z][2][9:16] #infrared magnitudes
        
        ir_lum = 10**((ir_mags+48.6)/(-2.5)) * (4*np.pi*(d * 100)**2) / 1e7 #IR flux in W/Hz
  
        df_temp['fir_lum'] = lir_total_W_lst[z]
        qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst, q_ionis = bressan_model(seds_nodust_lst,seds_bands_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)
        df_temp['qir_bress'] = qir_lst_bress[z]
        df_temp['rad_lum'] = bress_rad_lum_lst[z]
        df_temp['freefree'] = freefree_lst[z]
        df_temp['sync'] = sync_lst[z]
        df_temp['q_ionis'] = q_ionis[z]
        
        df_temp['ir_mag_14'] = seds_bands_lst[z][2][12] #infrared magnitudes at 14GHz


        df = pd.concat([df,df_temp])

    df['mstars_tot'] = (df['mstars_disk']+ df['mstars_bulge'])/h0

    df['mgas'] = (df['mgas_disk']+ df['mgas_bulge']) /h0
    df['sfr'] = (df['sfr_disk'] + df['sfr_burst'])/ 1e9 / h0  
    df['gas_metal'] = ((df['mgas_metals_bulge']+df['mgas_metals_disk'])/(df['mgas'])*h0)/0.018
    
    print(df['rgas_disk'])
    print(min(df['rgas_disk']))
    
    df['gas_surf_dens'] = (df['mgas_disk']/h0)/(2*(df['rgas_disk']/h0)**2*np.pi + 10**(-30))
    df['mgas'] = (df['mgas_disk']+ df['mgas_bulge']) /h0
    


    sf_lst = []
    
    df = df.astype(np.float64)
    
    
    df['sfg/q'] = 'q'
    df['sf_test'] = 1
    
    print("Finding SFG")
    for z in zlist:
        print("Finding SFG for z = ",str(round(z,2)))
        
        sfg_df = df[['sfr','mstars_tot']] [((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)) ]
        
        sfr = sfg_df['sfr']
        mst = sfg_df['mstars_tot']
        sfr = np.log10(sfr)
        mst = np.log10(mst)
        try:
            a,b = np.polyfit(mst,sfr,1)
        except:
            print("using a and b from last round")
        df['sf_test'] = np.log10(df['sfr']) - (a * np.log10(df['mstars_tot']) + b)
        df.loc[((df['z'] == z)&(df['sf_test'] > -0.3)), 'sfg/q'] = 'sf'

    df['macc'] = (df['bh_accretion_rate_hh'] + df['bh_accretion_rate_sb'])/h0/1e9
    
    print("THis is df mbh")
    print(df['m_bh'])
    
    df['m_bh'] = df['m_bh'] / h0
    obs_selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15) #GHz
    
    macc = df['macc']
    mbh = df['m_bh']

    Ljet_ADAF, Ljet_td, mdot_norm = radio_luminosity_agn(mbh, macc)
    
    Ljet_ADAF = np.array(Ljet_ADAF)
    
    Ljet_td = np.array(Ljet_td)
    
    mdot_norm = np.array(mdot_norm)
    
    

    
    
   # lum_radio_agn = np.zeros(shape = (len(obs_selection_freq), len(macc)))
   # lum_radio_agn_total = np.zeros(shape = (len(macc)))
    lum_radio_agn = radio_luminosity_per_freq(Ljet_ADAF, Ljet_td, mdot_norm, mbh, 1.4)

    print("THis is lum radio agn")
    print(lum_radio_agn)
    
    
    
    df['agn_rad'] = lum_radio_agn*1e-7
    
    df['agn_rad'] = df['agn_rad'].fillna(0)
    
    df['rad_tot'] = df['agn_rad'] + df['rad_lum']

    print(df['rad_lum'])
    print(df['agn_rad'])
    print(df['rad_tot'])
    
    df['sfr_ir'] = df['lir_w']/(5.8*1e9*Lsun)
    
    df['novak_r'] = np.log10(df['rad_tot']/df['sfr_ir'])
        
    print("THis is novak_r")
    print(df['novak_r'])
    df['qir_agn'] = np.log10(df['lir_w']/3.75e12) - np.log10(df['agn_rad'])    
    df['qir_tot'] = np.log10(df['lir_w']/3.75e12) - np.log10(df['rad_tot'])
    
    zobs = df['z']
    zobs = zobs.replace(0, 0.0001)
    print("This is zobs")
    print(zobs)
    
    dL = np.array(cosmo.comoving_distance(zobs) * (1.0 + zobs)) #luminosity distance in Mpc
    
    
    
    print("This is dL")
    print(dL)
    
    dfacgals = 4 * np.pi * (dL * mpc_to_cm)**2.0 #in cm^2
    
    print("this is dfacgals")
    print(dfacgals)
    dfacgals = 4 * PI * (dL * mpc_to_cm)**2.0 #in cm^2

    dL_dale = dL * 3.086e22

    df['Lrad_tot_flux'] = df['rad_tot']*1e7 / dfacgals* 1e26 #in mJy
    
    
    ir_mag_14 = df['ir_mag_14'] #infrared magnitudes at 14GHz
    d = 3.08567758128*10**17 #distance in m
    ir_lum = 10**((ir_mag_14+48.6)/(-2.5)) * (4*np.pi*(d * 100)**2) #IR flux in W/Hz
   # print(df['Lir_tot_flux'])    
    df['Lir_tot_flux'] = ir_lum/dfacgals*1e26
    
    print("This is flux")
    print(df['Lrad_tot_flux'])
    print(df['Lir_tot_flux'])    
    
   # print(df)
    
    return df


    

    
def GAMA_plots(df):
    
    
    df_line = pd.DataFrame()
    
    
    
    gd = pd.read_csv('GAMA_data.csv') #GAMA Data

    gd['qir'] = (np.log10(gd['DustLum_50']*Lsun/3.75*10**12) - np.log10(gd['radioLum2']))*10**(-1)
    

    gd['qir_err'] = (1/np.log(10)) * ((gd['DustLum_84'] - gd['DustLum_16'])/gd['DustLum_50'] + gd['radioLum2_err']/gd['radioLum2']) * (gd['radioLum2']/((gd['DustLum_50']*Lsun)/3.75*10**12))
    qir_gama = gd['qir']
    qir_err = gd['qir_err']
    m = gd['StellarMass_50']
    sfr_gama = gd['SFR_50']
    radlum = gd['radioLum2']
    firlum = gd['DustLum_50']*Lsun
    radlum_err = gd['radioLum2_err']
    firlum_err = ((gd['DustLum_16']*Lsun),(gd['DustLum_84']*Lsun))
    plt.clf()

    m_err = (gd['StellarMass_50']-gd['StellarMass_16'],gd['StellarMass_84']-gd['StellarMass_50'])
    sfr_err = (gd['SFR_50']-gd['SFR_16'],gd['SFR_84']-gd['SFR_50'])
    
    GAMA_df = df[['qir_bress','rad_lum','fir_lum','mstars_tot','sfr']] [((df['z'] == 0)&(df['fir_lum']>1e35)&(df['mstars_tot']>10**(8.8))&(df['mstars_tot']<1e12)&(df['sfg/q'] =='sf')) ]   #Selects all galaxies that correspond to these parameters which match GAMA. 
    
    qir = GAMA_df['qir_bress']
    rad = GAMA_df['rad_lum']
    fir = GAMA_df['fir_lum']
    mst = GAMA_df['mstars_tot']   ##Total stellar mass
    sfr = GAMA_df['sfr']

    fig, ax = plt.subplots(1,2)
    vmin = 4
    vmax = 30
    mid_rad,med_fir,low_fir,high_fir = median_line(rad,fir,True,True)
    
    df_line['mid_rad'] = mid_rad
    df_line['med_fir'] = med_fir
    df_line['low_fir'] = low_fir
    df_line['upp_fir'] = upp_fir
            
    
    
    #ax[0,0].hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    ax[0].plot(mid_rad,med_fir,'red')
    ax[0].fill_between(mid_rad,low_fir,high_fir,color = 'red',alpha = 0.5)
    ax[0].errorbar(radlum,firlum, xerr = radlum_err,yerr = firlum_err,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'GAMA Data',ecolor='black',elinewidth = 0.5)
    ax[0].set_xlabel('L$_{rad/1.4GHz}$/ W/Hz',fontsize = 20)
    ax[0].set_ylabel('(L$_{IR}$) W',fontsize = 20)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    mid_m_lst,med_sfr_lst,low_sfr_lst,high_sfr_lst = any_sigma_lines(mst,sfr,True,True,100,0)
    
    df_line['mid_mst'] = mid_m_lst
    df_line['med_sfr'] = med_sfr_lst
    df_line['low_sfr'] = low_sfr_lst
    df_line['upp_sfr'] = high_sfr_lst
    
    
    
    ax[1].plot(mid_m_lst,med_sfr_lst,'red')
    ax[1].fill_between(mid_m_lst,low_sfr_lst,high_sfr_lst,color = 'red',alpha = 0.5)
    ax[1].errorbar(m,sfr_gama, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='None', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5,markersize = 10)
    ax[1].set_xlabel("Stellar Mass/$M_\odot$",fontsize = 20)
    ax[1].set_ylabel("SFR/$M_\odot$/yr",fontsize = 20)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    fig.set_size_inches(12, 12)
    plt.tight_layout()
    plt.savefig('plots/GAMA_plts_model.pdf')
    plt.show()
    
    plt.clf()
    
    
    
   # cb = plt.colorbar(c)
   # cb.set_label('Number count')
   # plt.show()
   # plt.savefig('plots/GAMA_rad_lir.pdf')
   # plt.show()
    #plt.clf()
    
    
    
    mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,True,False)
    
    df_line['med_qir'] = med_qir
    df_line['low_qir'] = low_qir
    df_line['upp_qir'] = high_qir
    
    
    
    delv_qir = 2.586 + (-0.124)*(np.log10(mst) - 10) #Eqn from Delvecchio et al. 2021 from fig. 14
    
    
    
    
    
    mid_m,med_delv,low_delv,high_delv = median_line(mst,delv_qir,True,False)

    
    plt.plot(mid_m,med_qir,'red')
    plt.fill_between(mid_m,low_qir,high_qir,color = 'red',alpha = 0.5, label = 'SHARK')
    plt.plot(mid_m,med_delv,color = 'red')
    plt.fill_between(mid_m,low_delv,high_delv,color = 'red',alpha = 0.5, label = 'Delvecchio et al. 2021')
    plt.errorbar(m,qir_gama, xerr = m_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5)
    plt.xscale('log')
    plt.xlabel("Stellar Mass/$M_\odot$",fontsize = 12)
    plt.ylabel("qir",fontsize = 12)
    plt.legend()
    fig.set_size_inches(12, 12)
    plt.savefig('plots/GAMA_mstar_qir.pdf')
    plt.show()
    plt.clf()
    
    df_line.to_csv('GAMA_data_SHARK.csv')

def delvecchio_agn_removal(lst,m1,m2):
    plt.clf()
    ###provide a list of combined AGNs and SFGs qir
    label = str(m1) + "$\leq log_{10}(M_{*}M_{\odot}) \leq$" + str(m2)
    mlow = -3
    mupp = 5
    dm = 0.1
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    lst = np.array(lst)
    qhist, _ = np.histogram(lst, bins=np.append(mbins,mupp))
    
    ###Find qir_peak###
    ind = np.where(qhist == max(qhist))
    ind = ind[0]
    qir_peak = qhist[ind]
    ind = ind[0]
    
    ###Classify everything beyond this peak as a SFG
    
    qir_beyond_peak = qhist[ind:]
    xmf_beyond_peak = xmf[ind:]
                         
    ###Flip this histogram about this peak to find all SFGs
    
    
    qir_sym = list(np.flip(qir_beyond_peak)) + list(qir_beyond_peak)
    xmf_sym = list(min(xmf_beyond_peak) - np.flip(xmf_beyond_peak - min(xmf_beyond_peak))) +list(xmf_beyond_peak)
    qhist4 = np.histogram(qir_sym, bins=xmf_sym)

    ###Fit a Gaussian to this population
    def func(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    parameters, covariance = curve_fit(func, xmf_sym,qir_sym,maxfev=10**(8))
    mean = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    xmf_sym = np.array(xmf_sym)

    fit_y = func(xmf_sym, mean, fit_B,fit_C)
    
    ###Find the mean and sigma of this gaussian
    ind = np.where(qir_sym == max(qir_sym))
    qir_avg = xmf_sym[ind][0]

    
    hm = 0.5*qir_avg

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    low_hm = find_nearest(xmf_sym,hm)
    upp_hm = qir_avg + abs(qir_avg - low_hm)
    fwhm = upp_hm - low_hm
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    agn_cut = qir_avg - 2*sigma

    mean_ind = np.where(fit_y == max(fit_y))
    mean = qir_avg
    qir_sig = abs(fit_C)
    
    
    ###Find the AGN Cut and define those below this as AGNs
    agn_cut = mean - 2*qir_sig
    
    if len(qhist) > len(qir_sym):
    
        zero = np.zeros(len(qhist)-len(qir_sym))
        qir_sym2 = list(zero) + list(qir_sym)
        agn_his = qhist - qir_sym2
    elif len(qhist) < len(qir_sym):
        
        print("The histogram is not behaving as expected!")
        
        n = len(qir_sym)-len(qhist)
        qir_sym2 = qir_sym[n:]
        agn_his = qhist - qir_sym2
    else:
        print("There are no AGNs in this plot")
        agn_his = qhist - qir_sym
        
        



    parameters, covariance = curve_fit(func, xmf,agn_his,maxfev=10**(8))

    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    xmf_sym = np.array(xmf_sym)

    fit_y = func(xmf, fit_A, fit_B,fit_C)
    #plt.plot(xmf_sym,qir_sym, 'o', label='data')
    plt.plot(xmf, fit_y, '-', label='AGN fit')
    plt.legend()


    plt.plot(xmf,agn_his,label = 'AGN')
    plt.plot(xmf_sym,qir_sym,label = 'SFG')
    plt.plot(xmf,qhist,label = 'Total population')
    
    sig_lab = "$\sigma = $" + str(round(sigma,2))
    qir_lab = "$q_{IR,peak} = $" + str(round(mean,3))
    
    plt.annotate(qir_lab, xy=(0.7, 0.8), xycoords='axes fraction')
    plt.annotate(sig_lab, xy=(0.7, 0.95), xycoords='axes fraction')
    plt.annotate(label, xy=(0.7, 0.7), xycoords='axes fraction')
        
    #plt.plot(xmf,qhist1,label = '')
    #plt.plot(xmf,qhist3)
    plt.legend()
    
    save_name = 'Plots/delv_agn/agn_hist_' + str(m1) + '.pdf'
    plt.savefig(save_name)
    
    return agn_cut

def qir_z_removal(zlst,qir):
    
    qir = np.asarray(qir).ravel()
    zlst = np.asarray(zlst).ravel()

    ###Finds the qir_z fit
    def qir_z_curve(z,a,b):
        return a*(1+z)**b

    params, _ = curve_fit(qir_z_curve, np.array(zlst), np.array(qir),p0=[2.7,-0.055],maxfev=5000)
    a1, b1= params[0], params[1]

    y = qir_z_curve(zlst,a1,b1)
    
    ###Corrects for any z influence on qir
    
    qir_cor = qir/(1+zlst)**(b1)

    params, _ = curve_fit(qir_z_curve, np.array(zlst), np.array(qir_cor))
    a2, b2= params[0], params[1]

    if b2 > 0.5:
        for i in range(100):
            print('Something has gone badly wrong!')
            
    
    return a1, b1,a2,b2, qir_cor ###a1 and b1 are the parameters from the first fit. They can tell you about the z evolution of the set. a2 and b2 are parameters of the second fit. They are a test to see if it's been corrected correctly. qir_cor is the qir population corrected for z.

def delvecchio_agn_selection(df):
    
    
    mlst = [12,11,10.5]
    
    ###Fit the total qir to redshift including AGN emission###
    ###High masses###
    flux_limit = 1e-3
    df['delv_agn'] = 'AGN'
    
    for q in range(len(mlst)-1):
        m2 = mlst[q]
        m1 = mlst[q+1]
        print("Removing Galaxies with ", m1, "< M* < ",m2)

 
        df2 = df[['qir_tot','z']] [((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit))]
        qir_tot = df2['qir_tot']
        z = df2['z']

        a1, b1,a2,b2, qir_cor = qir_z_removal(z,qir_tot)
        
        s1 = str(m1)
        s2 = str(m2)
        
        if m1 == 10.5:
            s1 = '10d5'
        elif m2 == 10.5:
            s2 = '10d5'
        

        agn_cut = delvecchio_agn_removal(qir_cor,s1,s2)
    
        ###Remove AGNs from population and go again
        
        df2 = df[['qir_tot','z']] [((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['qir_tot']>agn_cut)&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit))]
        qir_tot = df2['qir_tot']
        z = df2['z']
        
        a1, b1,a2,b2, qir_cor = qir_z_removal(z,qir_tot)

        
        s1 = str(m1) + "2"
        s2 = str(m2) + "2"
        
        if m1 == 10.5:
            s1 = '10d52'
        elif m2 == 10.5:
            s2 = '10d52'
        
        agn_cut = delvecchio_agn_removal(qir_cor,s1,s2)
        df.loc[((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['qir_tot']>agn_cut)&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit)), 'delv_agn'] = 'sfg'
        
    ###Fit the total qir to redshift including AGN emission for M* > 10.5###
    df2 = df[['qir_tot','z']] [((df['mstars_tot']>10**(10.5)) & (df['mstars_tot']<10**(12)))]
    qir_tot = df2['qir_tot']
    z = df2['z']

    def qir_z_curve(z,a,b):
        return a*(1+z)**b

    params, _ = curve_fit(qir_z_curve, np.array(z), np.array(qir_tot))
    a1, b1= params[0], params[1]    ###These are the parameters of the curves of 10.5 < M* < 12. We assume that lower masses will have the same slope
    
    ###Find the AGNs for 8 - 10.5
    
    mlst = [10.5,10,9.5,9,8]
    print("This is b1")
    print(b1)
    def qir_z_curve_lowmass(z,a):
        return a*(1+z)**b1
    
    for q in range(len(mlst)-1):
        m2 = mlst[q]
        m1 = mlst[q+1]
        print("Removing Galaxies with ", m1, "< M* < ",m2)

        
        df2 = df[['qir_tot','z']] [((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit))]
        qir_tot = df2['qir_tot']
        z = df2['z']

        params, _ = curve_fit(qir_z_curve, np.array(z), np.array(qir_tot))
        print("This is b1")
        print(b1)
        a1 = params[0]
        
        
        qir_cor = qir_tot/(1+z)**b1
        
        s1 = str(m1)
        s2 = str(m2)
        
        if m1 == 10.5:
            s1 = '10d5'
        elif m2 == 10.5:
            s2 = '10d5'
        elif m1 == 9.5:
            s1 = '9d5'
        elif m2 == 9.5:
            s2 = '9d5'
        

        agn_cut = delvecchio_agn_removal(qir_cor,s1,s2)
    
        ###Remove AGNs from population and go again
        
        df2 = df[['qir_tot','z']] [((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['qir_tot']>agn_cut)&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit))]
        qir_tot = df2['qir_tot']
        z = df2['z']
        
        params, _ = curve_fit(qir_z_curve, np.array(z), np.array(qir_tot))
        print("This is b1")
        print(b1)
        a1 = params[0]
        
        
        qir_cor = qir_tot/(1+z)**b1

        
        s1 = str(m1) + "2"
        s2 = str(m2) + "2"
        
        if m1 == 10.5:
            s1 = '10d52'
        elif m2 == 10.5:
            s2 = '10d52'
        elif m1 == 9.5:
            s1 = '9d52'
        elif m2 == 9.5:
            s2 = '9d52'
        
        
        agn_cut = delvecchio_agn_removal(qir_cor,s1,s2)
        df.loc[((df['mstars_tot']>10**(m1)) & (df['mstars_tot']<10**(m2))&(df['qir_tot']>agn_cut)&(df['Lrad_tot_flux'] > flux_limit)&(df['Lir_tot_flux'] > flux_limit)), 'delv_agn'] = 'sfg'
        
    

    print("It worked!!!")

    
    print(df['delv_agn'])

    return df
    
def delv_qir_plts(df):
    df = delvecchio_agn_selection(df)
    plt.clf()
    
#     flux_limit = 0
    
#     df2 = df[['qir_tot','z','mstars_tot','qir_bress']] [((df['mstars_tot']>1e8) & (df['mstars_tot']<1e12)&(df['delv_agn'] == 'sfg'))]
#     df3 = df[['qir_tot','z','mstars_tot','qir_bress']] [((df['mstars_tot']>1e8) & (df['mstars_tot']<1e12))]
    
#     print("this is df2")
#     print(df2)
    
#     print("This is df3")
#     print(df3)
    
#     qir_tot_del = df2['qir_tot'] ###qir including AGN emission with delvecchio cut###
#     z_del = np.array(df2['z']) ###z with delvecchio cut###
#     mst_del = np.log10(df2['mstars_tot']) ###mstars with delvecchio cut###
#     qir_sfg_del = df2['qir_bress'] ###qir without AGN emission with delvecchio cut###

#     z_del = np.append(z_del,[-1,1])
    
#     qir_tot_sf = df3['qir_tot'] ###qir including AGN emission###
#     z_sf = np.array(df3['z']) ###z###
#     mst_sf = np.log10(df3['mstars_tot']) ###mstars###
#     qir_sfg_sf = df3['qir_bress'] ###qir without AGN emission###

#     z_sf = np.append(z_sf,[-1,1])    
#     fig, ax = plt.subplots(1,2)
#     ###a = agn, s = sf###
#     mid_zda,med_qitda,low_qitda,high_qitda = median_line(z_del,qir_tot_del,False,False) ###calculate qir_tot median lines and mid z points for Delvecchio cut ###
#     mid_zds,med_qitds,low_qitds,high_qitds = median_line(z_del,qir_sfg_del,False,False) ###calculate qir median lines and mid z points for Delvecchio cut ###    
#     mid_zsa,med_qisa,low_qisa,high_qisa = median_line(z_sf,qir_tot_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###
#     mid_zss,med_qiss,low_qiss,high_qiss = median_line(z_sf,qir_sfg_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###        
    
#     ###Get rid of nans###
    
#     mid_zda = np.array(mid_zda)[~np.isnan(mid_zda)]
#     med_qitda = np.array(med_qitda)[~np.isnan(med_qitda)]
#     low_qitda = np.array(low_qitda)[~np.isnan(low_qitda)]
#     high_qitda = np.array(high_qitda)[~np.isnan(high_qitda)]

#     mid_zds = np.array(mid_zds)[~np.isnan(mid_zds)]
#     med_qitds = np.array(med_qitds)[~np.isnan(med_qitds)]
#     low_qitds = np.array(low_qitds)[~np.isnan(low_qitds)]
#     high_qitds = np.array(high_qitds)[~np.isnan(high_qitds)]

# #     mid_zsa = np.array(mid_zsa)[~np.isnan(mid_zsa)]
# #     med_qisa = np.array(med_qisa)[~np.isnan(med_qisa)]
# #     low_qisa = np.array(low_qisa)[~np.isnan(low_qisa)]
# #     high_qisa = np.array(high_qisa)[~np.isnan(high_qisa)]

# #     mid_zss = np.array(mid_zss)[~np.isnan(mid_zss)]
# #     med_qiss = np.array(med_qiss)[~np.isnan(med_qiss)]
# #     low_qiss = np.array(low_qiss)[~np.isnan(low_qiss)]
# #     high_qiss = np.array(high_qiss)[~np.isnan(high_qiss)]
#     print(mid_zda)
#     print(low_qitda)
    
    
    
#     ax[0].plot(mid_zda,med_qitda,'blue',linewidth = 3,label = 'Total qir, Delv. Cut')
#     ax[0].fill_between(mid_zda,low_qitda,high_qitda,color = 'blue',alpha = 0.5)    

#     ax[0].plot(mid_zds,med_qitds,'red',linewidth = 3,label = 'SF qir, Delv. Cut')
#     ax[0].fill_between(mid_zda,low_qitda,high_qitda,color = 'red',alpha = 0.5)  
    
#     ax[0].plot(mid_zsa,med_qisa,'orange',linewidth = 3,label = 'Total qir, no cut')
#     ax[0].fill_between(mid_zsa,low_qisa,high_qisa,color = 'orange',alpha = 0.5)   
    
#     ax[0].plot(mid_zss,med_qiss,'purple',linewidth = 3,label = 'SF qir, no cut')
#     ax[0].fill_between(mid_zss,low_qiss,high_qiss,color = 'purple',alpha = 0.5)     
    
#     ###Find the median lines for qir vs. mstar###
    

    
#     mid_mda,med_qitda,low_qitda,high_qitda = median_line(mst_del,qir_tot_del,False,False) ###calculate qir_tot median lines and mid z points for Delvecchio cut ###
#     mid_mds,med_qitds,low_qitds,high_qitds = median_line(mst_del,qir_sfg_del,False,False) ###calculate qir median lines and mid z points for Delvecchio cut ###    
#     mid_msa,med_qisa,low_qisa,high_qisa = median_line(mst_del,qir_tot_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###
#     mid_mss,med_qiss,low_qiss,high_qiss = median_line(mst_del,qir_sfg_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###        
    
    
#     print(mid_msa,med_qisa)
#     print(mid_mss,med_qiss)
#     ax[1].plot(mid_mda,med_qitda,'blue',linewidth = 3,label = 'Total qir, Delv. Cut')
#     ax[1].fill_between(mid_mda,low_qitda,high_qitda,color = 'blue',alpha = 0.5)    

#     ax[1].plot(mid_mds,med_qitds,'red',linewidth = 3,label = 'SF qir, Delv. Cut')
#     ax[1].fill_between(mid_mda,low_qitda,high_qitda,color = 'red',alpha = 0.5)  
    
#     ax[1].plot(mid_msa,med_qisa,'orange',linewidth = 3,label = 'Total qir, no cut')
#     ax[1].fill_between(mid_msa,low_qisa,high_qisa,color = 'orange',alpha = 0.5)   
    
#     ax[1].plot(mid_mss,med_qiss,'purple',linewidth = 3,label = 'SF qir, no cut')
#     ax[1].fill_between(mid_mss,low_qiss,high_qiss,color = 'purple',alpha = 0.5)     
#     plt.legend()
#     plt.savefig('Plots/delv_agn/delv_qir_m_z.pdf')
    
#     plt.clf()

    
    df2 = df[['qir_tot','z','mstars_tot','qir_bress']] [((df['mstars_tot']>1e8) & (df['mstars_tot']<1e12)&(df['delv_agn'] == 'sfg'))]
    df3 = df[['qir_tot','z','mstars_tot','qir_bress']] [((df['mstars_tot']>1e8) & (df['mstars_tot']<1e12))]
    
    df_dd = pd.read_csv('Delvecchio_data.csv')
    print(df2)
    i = 0
    z = 0
    q = 0
    h = 0
    g = 0
    mlst = [8.0,9.0,9.5,10.0,10.5,11.0,12.0]
    fig, ax = plt.subplots(2, 3)
    
    delv_nrm = [2.83,2.78,2.75,2.65,2.58,2.56]  ###normalisations from Delvecchio
    delv_err = [0.1,0.03,0.02,0.03,0.01,0.02] ###errors from Delvecchio

    x = np.linspace(0,6)


    
    nrm = delv_nrm[i]
    err = delv_err[i]
    
    y_med = (nrm)*(1+x)**(-0.055)
    y_min = (nrm-err)*(1+x)**(-0.055-0.018)
    y_max = (nrm+err)*(1+x)**(-0.055+0.018)


    
    for i in range(6):
        
        g = i//3
        h = i%3
        

        
        m = mlst[i]
        n = mlst[i+1]
        
        label = str(m) + "$\\rm \leq log_{10}(M/M_{\odot}) \leq$" + str(n)
        df2 = df[['qir_tot','z','qir_bress']] [((df['mstars_tot']>10**(m)) & (df['mstars_tot']<10**(n))&(df['delv_agn'] == 'sfg'))]
        df3 = df[['qir_tot','z','qir_bress']] [((df['mstars_tot']>10**(m)) & (df['mstars_tot']<10**(n)))]
        
        qir_tot_del = df2['qir_tot'] ###qir including AGN emission with delvecchio cut###
        z_del = np.array(df2['z']) ###z with delvecchio cut###
       # mst_del = np.log10(df2['mstars_tot']) ###mstars with delvecchio cut###
        qir_sfg_del = df2['qir_bress'] ###qir without AGN emission with delvecchio cut###
        
        z_del = np.append(z_del,[-0.0001,5.2])
        qir_tot_del = np.append(qir_tot_del,[np.nan,np.nan])
        qir_sfg_del = np.append(qir_sfg_del,[np.nan,np.nan])
        
        qir_tot_sf = df3['qir_tot'] ###qir including AGN emission###
        z_sf = np.array(df3['z']) ###z###
       # mst_sf = np.log10(df3['mstars_tot']) ###mstars###
        qir_sfg_sf = df3['qir_bress'] ###qir without AGN emission###

        z_sf = np.append(z_sf,[-0.0001,5.2])
        qir_tot_sf = np.append(qir_tot_sf,[np.nan,np.nan])
        qir_sfg_sf = np.append(qir_sfg_sf,[np.nan,np.nan])
                
        mid_zda,med_qitda,low_qitda,high_qitda = median_line(z_del,qir_tot_del,False,False) ###calculate qir_tot median lines and mid z points for Delvecchio cut ###
        mid_zds,med_qitds,low_qitds,high_qitds = median_line(z_del,qir_sfg_del,False,False) ###calculate qir median lines and mid z points for Delvecchio cut ###    
        mid_zsa,med_qisa,low_qisa,high_qisa = median_line(z_sf,qir_tot_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###
        mid_zss,med_qiss,low_qiss,high_qiss = median_line(z_sf,qir_sfg_sf,False,False) ###calculate qit_tot median lines and mid z points for all Shark###      
        
        mid_zsa = np.array(mid_zsa)[~np.isnan(mid_zsa)]
        med_qisa = np.array(med_qisa)[~np.isnan(med_qisa)]
        low_qisa = np.array(low_qisa)[~np.isnan(low_qisa)]
        high_qisa = np.array(high_qisa)[~np.isnan(high_qisa)]

        mid_zss = np.array(mid_zss)[~np.isnan(mid_zss)]
        med_qiss = np.array(med_qiss)[~np.isnan(med_qiss)]
        low_qiss = np.array(low_qiss)[~np.isnan(low_qiss)]
        high_qiss = np.array(high_qiss)[~np.isnan(high_qiss)]
        

        ###Get rid of nans###

        mid_zda = np.array(mid_zda)[~np.isnan(mid_zda)]
        med_qitda = np.array(med_qitda)[~np.isnan(med_qitda)]
        low_qitda = np.array(low_qitda)[~np.isnan(low_qitda)]
        high_qitda = np.array(high_qitda)[~np.isnan(high_qitda)]

        mid_zds = np.array(mid_zds)[~np.isnan(mid_zds)]
        med_qitds = np.array(med_qitds)[~np.isnan(med_qitds)]
        low_qitds = np.array(low_qitds)[~np.isnan(low_qitds)]
        high_qitds = np.array(high_qitds)[~np.isnan(high_qitds)]
        
        delv_df = df_dd[['qir_nondet','dqir_nondet','qir_all','dqir_all','zmean']] [((df_dd['mass_min'] == m) & (df_dd['mass_max'] == n)) ]          
        qir_nondet = delv_df['qir_nondet']
        qir_nondet_err = delv_df['dqir_nondet']
        qir_all = delv_df['qir_all']
        qir_all_err = delv_df['dqir_all']
        z_delv = delv_df['zmean']
        
        ax[g,h].plot(mid_zda,med_qitda,'blue',linewidth = 3,label = 'Total qir, Delv. Cut')
        ax[g,h].fill_between(mid_zda,low_qitda,high_qitda,color = 'blue',alpha = 0.5)    

       # ax[g,h].plot(mid_zds,med_qitds,'red',linewidth = 3,label = 'SF qir, Delv. Cut')
       # ax[g,h].fill_between(mid_zda,low_qitda,high_qitda,color = 'red',alpha = 0.5)  

       # ax[g,h].plot(mid_zsa,med_qisa,'orange',linewidth = 3,label = 'Total qir, no cut')
       # ax[g,h].fill_between(mid_zsa,low_qisa,high_qisa,color = 'orange',alpha = 0.5)   

      #  ax[g,h].plot(mid_zss,med_qiss,'purple',linewidth = 3,label = 'SF qir, no cut')
      #  ax[g,h].fill_between(mid_zss,low_qiss,high_qiss,color = 'purple',alpha = 0.5)     
        
      #  ax[g,h].errorbar(z_delv, qir_all, yerr=qir_all_err, fmt="o",label = 'Average individual detetections and stacked non-detections')
      #  ax[g,h].errorbar(z_delv, qir_nondet, yerr=qir_nondet_err, fmt="o",label = 'Stacks undetections')
    
        ax[g,h].set_xlabel('z',size = 15)
        ax[g,0].set_ylabel('$\\rm q_{IR}$',size = 15)
        ax[g,h].text(1,1,label)
        ax[g,h].tick_params(bottom = True, top = True,direction = 'inout',labelsize=15)
        ax[g,h].tick_params(which = 'minor',bottom = True, top = True,direction = 'inout')
        ax[0,h].tick_params(labelbottom=False, top = True,labelsize=15) 
        ax[0,h].tick_params(top = False)
        ax[g,1].tick_params(labelleft=False)
        ax[g,2].tick_params(labelleft=False)
        ax[g,h].set_xlim(0,5)
        ax[g,h].set_ylim(0,4)
        nrm = delv_nrm[i]
        err = delv_err[i]
    
        y_med = (nrm)*(1+x)**(-0.055)
        y_min = (nrm-err)*(1+x)**(-0.055-0.018)
        y_max = (nrm+err)*(1+x)**(-0.055+0.018)


        ax[g,h].plot(x,y_med,'red')
        ax[g,h].fill_between(x,y_min,y_max,alpha = 0.5,color = 'red')

        
        
    plt.subplots_adjust(wspace=0, hspace=0)
    leg = ax[1,2].legend(loc='lower right', bbox_to_anchor=(0, 0))
    leg.get_frame().set_linewidth(0.0)
    fig.set_size_inches(12, 8)
    plt.savefig("plots/delv_plt_agn.pdf")
    plt.show()
    
def lo_faro_plots(df):
    
    df_line = pd.DataFrame()
    Lsun = 3.828 * 10**26 #W
    
    ###Reading in and setting up data from Lo Faro et al. ###
    
    df_lf = pd.read_csv('french_paper_2_pg16.csv')
    
    df_lf['LIR/W'] = df_lf['LIR'] * Lsun

    df_lf['qir'] = np.log10((df_lf['LIR/W']/(3.75*10**12))/df_lf['L1.4'])
    
    df_lf['L1.4_err'] = 0.13 + df_lf['qir_err']#approximate error of L_1.4
    
    ### Reading in LIRG data from Lo Faro et al. ###
    
    LIRG_lf = df_lf[['SFR10','M_star','LIR/W','L1.4','qir','L1.4_err','qir_err']] [(df_lf['z'] < 1.5) ]
    
    sfr_L = np.log10(LIRG_lf['SFR10'])
    mst_L = np.log10(LIRG_lf['M_star'])
    lir_L = np.log10(LIRG_lf['LIR/W'])
    rad_L = np.log10(LIRG_lf['L1.4'])
    qir_L = LIRG_lf['qir']
    Lrad_err_L = LIRG_lf['L1.4_err']
    qir_err_L = LIRG_lf['qir_err']
    
    ###Reading in ULIRG from Lo Faro et al.###
    
    ULIRG_lf = df_lf[['SFR10','M_star','LIR/W','L1.4','qir','L1.4_err','qir_err']] [(df_lf['z'] > 1.5) ]
    
    sfr_U = np.log10(ULIRG_lf['SFR10'])
    mst_U = np.log10(ULIRG_lf['M_star'])
    lir_U = np.log10(ULIRG_lf['LIR/W'])
    rad_U = np.log10(ULIRG_lf['L1.4'])
    qir_U = ULIRG_lf['qir']
    Lrad_err_U = ULIRG_lf['L1.4_err']
    qir_err_U = ULIRG_lf['qir_err']
    

    
    ###Reading in data from SHARK dataframe for LIRGs###
    
    
    LIRG = df[['qir_bress','rad_lum','fir_lum','mstars_tot','sfr']] [(df['fir_lum']>1e11*Lsun)&(df['fir_lum']<1e12*Lsun) ]

    sfr_SL = np.log10(LIRG['sfr']) #SFR for SHARK LIRGs
    mst_SL = np.log10(LIRG['mstars_tot'])
    lir_SL = np.log10(LIRG['fir_lum'])
    rad_SL = np.log10(LIRG['rad_lum'])
    qir_SL = LIRG['qir_bress']
    
    ###Reading in data from SHARK dataframe for ULIRGs

    ULIRG = df[['qir_bress','rad_lum','fir_lum','mstars_tot','sfr']] [(df['fir_lum']>1e12*Lsun)&(df['fir_lum']<1e13*Lsun) ]
    
    sfr_SU = np.log10(ULIRG['sfr']) #SFR for SHARK ULIRGs
    mst_SU = np.log10(ULIRG['mstars_tot'])
    lir_SU = np.log10(ULIRG['fir_lum'])
    rad_SU = np.log10(ULIRG['rad_lum'])
    qir_SU = ULIRG['qir_bress']

    ###Reading in data from SHARK dataframe for both LIRGs and ULIRGs
    
    all_LIRG = df[['rad_lum','fir_lum']] [(df['fir_lum']>1e11*Lsun) ]
    
    rad_all = np.log10(all_LIRG['rad_lum'])
    fir_all = np.log10(all_LIRG['fir_lum'])
    
    ###Creating test plots###
    vmin = 1
    vmax = 100
    fig, ax = plt.subplots(1,2)
    
    ###Creating median lines for a single line in Lir vs. Lrad
    
    mid_fir,med_rad,low_rad,high_rad = median_line(fir_all,rad_all,False,False)
    where_LIRG = np.where(mid_fir < np.log10(1e12*Lsun)) #finds the LIRGs
    where_ULIRG = np.where(mid_fir > np.log10(1e12*Lsun)) #finds the ULIRGs
    
    med_rad_L = med_rad[where_LIRG] #finds the radio of LIRGs
    mid_fir_L = mid_fir[where_LIRG] #finds the fir of LIRGs
    high_rad_L = high_rad[where_LIRG]
    low_rad_L = low_rad[where_LIRG]
    
    
    med_rad_U = med_rad[where_ULIRG] #finds the radio of ULIRGs
    mid_fir_U = mid_fir[where_ULIRG] #finds the fir of ULIRGs
    high_rad_U = high_rad[where_ULIRG]
    low_rad_U = low_rad[where_ULIRG]

    
    med_rad_U = np.append(med_rad_L[-1],med_rad_U)
    mid_fir_U = np.append(mid_fir_L[-1],mid_fir_U)
    high_rad_U = np.append(high_rad_L[-1],high_rad_U)
    low_rad_U = np.append(low_rad_L[-1],low_rad_U)

    lo_faro_lirg_rad_med = np.log10(10**(mid_fir_L)/(3.75*1e12)) - 2.83
    lo_faro_lirg_rad_low = np.log10(10**(mid_fir_L)/(3.75*1e12)) - 2.93
    lo_faro_lirg_rad_upp = np.log10(10**(mid_fir_L)/(3.75*1e12)) - 2.73

    lo_faro_ulirg_rad_med = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.76
    lo_faro_ulirg_rad_upp = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.86
    lo_faro_ulirg_rad_low = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.66

    sargent_ulirg_rad_med = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.672
    sargent_ulirg_rad_upp = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.793
    sargent_ulirg_rad_low = np.log10(10**(mid_fir_U)/(3.75*1e12)) - 2.551

   # ax[0].hexbin(rad_SL,lir_SL,mincnt = 1,cmap='Blues', gridsize = 50,alpha = 0.75,label = 'SHARK - LIRGs',extent = (22,24.5, 37.85, 39.50),vmin = vmin,vmax = vmax)
   # ax[0].hexbin(rad_SU,lir_SU,mincnt = 1,cmap='Reds', gridsize = 50,alpha = 0.75,label = 'SHARK - ULIRGs',extent = (22,24.5, 37.85, 39.50),vmin = vmin,vmax = vmax)
    temp_rad_L = np.zeros(30)
    temp_fir_L = np.zeros(30)
    temp_low_L = np.zeros(30)
    temp_upp_L = np.zeros(30)
    
    temp_rad_U = np.zeros(30)
    temp_fir_U = np.zeros(30)
    temp_low_U = np.zeros(30)
    temp_upp_U = np.zeros(30)    
    
    for i in range(len(med_rad_L)):
        temp_rad_L[i] = med_rad_L[i]
        temp_fir_L[i] = med_fir_L[i]
        temp_low_L[i] = low_rad_L[i]
        temp_upp_L[i] = high_rad_L[i]
    
    for i in range(len(data_2)):
         
        temp_rad_U[-i-1] = med_rad_U
        temp_fir_U[-i-1] = med_rad_U
        temp_low_U[-i-1] = med_fir_U 
        temp_upp_U[-i-1] = high_rad_U 
    
    df_line['med_rad_L'] = temp_rad_L
    df_line['mid_fir_L'] = temp_fir_L    
    df_line['low_rad_L'] = temp_low_L
    df_line['upp_rad_L'] = temp_upp_L
    
    df_line['med_rad_U'] = temp_rad_U
    df_line['mid_fir_U'] = temp_fir_U    
    df_line['low_rad_U'] = temp_low_U
    df_line['upp_rad_U'] = temp_upp_U
    
    ax[0].plot(med_rad_L,mid_fir_L,'blue',linewidth = 3,label = 'SHARK - LIRGs')
    ax[0].fill_betweenx(mid_fir_L,low_rad_L,high_rad_L,color = 'blue',alpha = 0.5)
    ax[0].plot(med_rad_U,mid_fir_U,'red',linewidth = 3,label = 'SHARK - ULIRGs')
    ax[0].fill_betweenx(mid_fir_U,low_rad_U,high_rad_U,color = 'red',alpha = 0.5)
    ax[0].plot(lo_faro_lirg_rad_med,mid_fir_L,'black',linewidth = 3,linestyle = 'solid',label = 'Lo Faro et al. (2015)')
    ax[0].plot(lo_faro_ulirg_rad_med,mid_fir_U,'black',linewidth = 3,linestyle = 'solid')
    ax[0].plot(sargent_ulirg_rad_med,mid_fir_U,'black',linewidth = 3,linestyle = 'dashed',label = 'Sargent et al. (2010)')    
    ax[0].errorbar(rad_L,lir_L, xerr = Lrad_err_L,yerr = 0.13,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - LIRGs',ecolor='black', markersize='10')
    ax[0].errorbar(rad_U,lir_U, xerr = Lrad_err_U,yerr = 0.13,fmt="*",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - ULIRGs',ecolor='black', markersize='10')
    ax[0].set_xlim(22,24.5)
    ax[0].set_ylim(37.85,39.50)    
    ax[0].set_xlabel('$Log_{10}$(L$_{rad/1.4GHz}$/ W/Hz)',fontsize = 20)
    ax[0].set_ylabel('$Log_{10}$((L$_{IR}$)/W)',fontsize = 20)
                                         
    ax[0].legend()
    mid_m_L,med_sfr_L,low_sfr_L,high_sfr_L = median_line(mst_SL,sfr_SL,False,False)
    mid_m_U,med_sfr_U,low_sfr_U,high_sfr_U = median_line(mst_SU,sfr_SU,False,False)
   # c1 = ax[1].hexbin(mst_SL,sfr_SL,mincnt = 1,cmap='Blues', gridsize = 50,alpha = 0.75,extent = (10,12, 0, 3),vmin = vmin,vmax = vmax)
   # c2 = ax[1].hexbin(mst_SU,sfr_SU,mincnt = 1,cmap='Reds', gridsize = 50,alpha = 0.75,extent = (10,12, 0, 3),vmin = vmin,vmax = vmax)
    
    df_line['med_mst_L'] = pd.Series(mid_m_L)
    df_line['med_sfr_L'] = pd.Series(med_sfr_L)    
    df_line['low_sfr_L'] = pd.Series(low_sfr_L)
    df_line['upp_sfr_L'] = pd.Series(high_sfr_L)
    
    df_line['med_mst_U'] = pd.Series(mid_m_U)
    df_line['med_sfr_U'] = pd.Series(med_sfr_U)    
    df_line['low_sfr_U'] = pd.Series(low_sfr_U)
    df_line['upp_sfr_U'] = pd.Series(high_sfr_U)
    
    
    
    ax[1].plot(mid_m_L,med_sfr_L,'blue',label = 'SHARK - LIRGs',linewidth = 3)
    ax[1].fill_between(mid_m_L,low_sfr_L,high_sfr_L,color = 'blue',alpha = 0.5)
    ax[1].plot(mid_m_U,med_sfr_U,'red',label = 'SHARK - ULIRGs',linewidth = 3)
    ax[1].fill_between(mid_m_U,low_sfr_U,high_sfr_U,color = 'red',alpha = 0.5)
    ax[1].errorbar(mst_L,sfr_L, xerr = 0.2,yerr = 0.2,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - LIRGs',ecolor='black', markersize='10')
    ax[1].errorbar(mst_U,sfr_U, xerr = 0.2,yerr = 0.2,fmt="*",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - ULIRGs',ecolor='black', markersize='10')
    ax[1].set_xlabel("$Log_{10}$(Stellar Mass/$M_\odot$)",fontsize = 20)
    ax[1].set_ylabel("$Log_{10}$(SFR/$M_\odot$/yr)",fontsize = 20)
    ax[1].set_xlim(9,12)
    ax[1].set_ylim(0,3)
    
   # cb1 = plt.colorbar(c1,pad=0,fraction=0.27)
   # cb2 = plt.colorbar(c2,pad=0,fraction=0.1)
    #cb2.set_ticks([])
   # cb1.set_label('Number Count')
    
    ax[0].legend()
    fig.set_size_inches(15,15)
    plt.savefig('plots/Lo_Faro_plts.pdf')
    
    plt.show()    
    plt.clf()
    fig, ax = plt.subplots(1,1)
    ###Creating qir mstar plots###
    lin_mst = np.linspace(9,12,2) 
    lirg_lf_mid = np.full(len(lin_mst),2.83)
    lirg_lf_low = np.full(len(lin_mst),2.73)
    lirg_lf_upp = np.full(len(lin_mst),2.93)

    ulirg_lf_mid = np.full(len(lin_mst),2.76)
    ulirg_lf_low = np.full(len(lin_mst),2.66)
    ulirg_lf_upp = np.full(len(lin_mst),2.86)

    ulirg_sar_mid = np.full(len(lin_mst),2.672)
    ulirg_sar_low = np.full(len(lin_mst),2.793)
    ulirg_sar_upp = np.full(len(lin_mst),2.551)
    
    
    mid_m_U,med_qir_U,low_qir_U,high_qir_U = median_line(mst_SU,qir_SU,False,False)
    mid_m_L,med_qir_L,low_qir_L,high_qir_L = median_line(mst_SL,qir_SL,False,False)
    

    df_line['med_qir_L'] = pd.Series(med_qir_L)    
    df_line['low_qir_L'] = pd.Series(low_qir_L)
    df_line['upp_qir_L'] = pd.Series(high_qir_L)
    
    df_line['med_qir_U'] = pd.Series(med_qir_U)    
    df_line['low_qir_U'] = pd.Series(low_qir_U)
    df_line['upp_qir_U'] = pd.Series(high_qir_U)
    
    

  #  c1 = ax.hexbin(mst_SL,qir_SL,mincnt = 1,cmap='Blues', gridsize = 30,alpha = 0.75,extent = (10,12, 2, 3.2),vmin = vmin,vmax = vmax)
  #  c2 = ax.hexbin(mst_SU,qir_SU,mincnt = 1,cmap='Reds', gridsize = 30,alpha = 0.75,extent = (10,12, 2, 3.2),vmin = vmin,vmax = vmax
    ax.plot(mid_m_L,med_qir_L,'blue',label = 'SHARK - LIRGs',linewidth = 3) 
    ax.fill_between(mid_m_L,low_qir_L,high_qir_L,color = 'blue',alpha = 0.5)
    ax.plot(mid_m_U,med_qir_U,'red',label = 'SHARK - ULIRGs',linewidth = 3)
    ax.fill_between(mid_m_U,low_qir_U,high_qir_U,color = 'red',alpha = 0.5)
    ax.plot(lin_mst,lirg_lf_mid,label = 'Lo Faro et al. (2015) - LIRGs',color = 'blue',linestyle = 'dashed',linewidth = 3)
    ax.plot(lin_mst,ulirg_lf_mid,label = 'Lo Faro et al. (2015) - ULIRGs',color = 'red',linestyle = 'dashed',linewidth = 3)
    ax.plot(lin_mst,ulirg_sar_mid,label = 'Sargent et al. (2010) - ULIRGs',color = 'black',linestyle = 'dashed',linewidth = 3)
    ax.errorbar(mst_L,qir_L, xerr = 0.2,yerr = qir_err_L ,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - LIRGs',ecolor='black', markersize='10')
    ax.errorbar(mst_U,qir_U, xerr = 0.2,yerr = qir_err_U ,fmt="*",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - ULIRGs',ecolor='black', markersize='10') 
    ax.set_xlabel("$Log_{10}$(Stellar Mass/$M_\odot$)",fontsize = 20)
    ax.set_ylabel("qir",fontsize = 20)
    ax.set_xlim(9,12)
    ax.set_ylim(2.2,3.2)
  #  cb1 = plt.colorbar(c1,pad=0,fraction=0.335)
  #  cb2 = plt.colorbar(c2,pad=0,fraction=0.075)
  #  cb2.set_ticks([])
  #  cb1.set_label('Number Count')
    ax.legend()
    fig.set_size_inches(10, 10)
    plt.savefig('plots/Lo_Faro_mstar_qir.pdf')
    plt.show()
    plt.clf()
    ###Creating the qir vs. sfr plot###
    fig, ax = plt.subplots(1,1)
    
    lin_sfr = np.linspace(0,3,2)

    lirg_lf_mid = np.full(len(lin_sfr),2.83)

    ulirg_lf_mid = np.full(len(lin_sfr),2.76)


    ulirg_sar_mid = np.full(len(lin_sfr),2.672)
    
    
    mid_s_U,med_qir_U,low_qir_U,high_qir_U = median_line(sfr_SU,qir_SU,False,False)
    mid_s_L,med_qir_L,low_qir_L,high_qir_L = median_line(sfr_SL,qir_SL,False,False)
    
    df_line['mid_sfr_L'] = pd.Series(mid_s_L)
    df_line['med_qir_L_sfr'] = pd.Series(med_qir_L)    
    df_line['low_qir_L_sfr'] = pd.Series(low_qir_L)
    df_line['upp_qir_L_sfr'] = pd.Series(high_qir_L)
    
    df_line['mid_sfr_U'] = pd.Series(mid_s_U)
    df_line['med_qir_U_sfr'] = pd.Series(med_qir_U)    
    df_line['low_qir_U_sfr'] = pd.Series(low_qir_U)
    df_line['upp_qir_U_sfr'] = pd.Series(high_qir_U)   

    
    
    
    
    ax.plot(mid_s_L,med_qir_L,'blue',label = 'SHARK - LIRGs',linewidth = 3) 
    ax.fill_between(mid_s_L,low_qir_L,high_qir_L,color = 'blue',alpha = 0.5)
    ax.plot(mid_s_U,med_qir_U,'red',label = 'SHARK - LIRGs',linewidth = 3)
    ax.fill_between(mid_s_U,low_qir_U,high_qir_U,color = 'red',alpha = 0.5)
    
    ax.plot(lin_sfr,lirg_lf_mid,label = 'Lo Faro et al. (2015) - LIRGs',color = 'blue',linestyle = 'dashed',linewidth = 3)
    ax.plot(lin_sfr,ulirg_lf_mid,label = 'Lo Faro et al. (2015) - ULIRGs',color = 'red',linestyle = 'dashed',linewidth = 3)
    ax.plot(lin_sfr,ulirg_sar_mid,label = 'Sargent et al. (2010) - ULIRGs',color = 'black',linestyle = 'dashed',linewidth = 3)
    
    ax.errorbar(sfr_L,qir_L, xerr = 0.2,yerr = qir_err_L ,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - LIRGs',ecolor='black', markersize='10')
    ax.errorbar(sfr_U,qir_U, xerr = 0.2,yerr = qir_err_U ,fmt="*",markerfacecolor='white', markeredgecolor='black',label = 'Lo Faro et al. (2015) - ULIRGs',ecolor='black', markersize='10') 
    ax.set_xlabel("$Log_{10}$(SFR/$M_\odot$/yr)",fontsize = 20)
    ax.set_ylabel("qir",fontsize = 20)
    ax.set_xlim(0,3)
    ax.set_ylim(2.2,3.2)
  #  cb1 = plt.colorbar(c1,pad=0,fraction=0.335)
  #  cb2 = plt.colorbar(c2,pad=0,fraction=0.075)
  #  cb2.set_ticks([])
  #  cb1.set_label('Number Count')
    ax.legend()
    fig.set_size_inches(10, 10)
    plt.savefig('plots/Lo_Faro_sfr_qir.pdf')
    plt.show()

    df_line.to_csv('lo_faro_plots_SHARK_lines_new.csv')
    
    ###Creating plots for LIRGs
    
    mid_rad,med_fir,low_fir,high_fir = median_line(rad_SL,lir_SL,False,False)
    #ax[0,0].hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    ax[0].plot(mid_rad,med_fir,'blue')
    ax[0].fill_between(mid_rad,low_fir,high_fir,color = 'blue',alpha = 0.5)
    ax[0].errorbar(rad_L,lir_L, xerr = 0.13,yerr = 0.13,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lor Faro et al.',ecolor='black',elinewidth = 0.5)
    ax[0].set_xlabel('L$_{rad/1.4GHz}$/ W/Hz',fontsize = 12)
    ax[0].set_ylabel('(L$_{IR}$) W',fontsize = 12)

    
    mid_m,med_sfr,low_sfr,high_sfr = median_line(mst_SL,sfr_SL,False,False)
    ax[1].plot(mid_m,med_sfr,'blue')
    ax[1].fill_between(mid_m,low_sfr,high_sfr,color = 'blue',alpha = 0.5)
    ax[1].errorbar(mst_L,sfr_L, xerr =0.2,yerr = 0.2,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5)
    ax[1].set_xlabel("Stellar Mass/$M_\odot$",fontsize = 12)
    ax[1].set_ylabel("SFR/$M_\odot$/yr",fontsize = 12)
    
    fig.set_size_inches(11, 5)
    plt.tight_layout()
    plt.savefig('plots/Lo_Faro_LIRG.pdf')
    plt.show()
    
    plt.clf()
    
    
    mid_m,med_qir,low_qir,high_qir = median_line(mst_SL,qir_SL,False,False)
    plt.plot(mid_m,med_qir,'blue')
    plt.fill_between(mid_m,low_qir,high_qir,color = 'blue',alpha = 0.5, label = 'SHARK')
    plt.errorbar(mst_L,qir_L, xerr = 0.2,yerr = 0.26,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5)
    plt.xlabel("Stellar Mass/$M_\odot$",fontsize = 12)
    plt.ylabel("qir",fontsize = 12)
    fig.set_size_inches(10, 10)
    plt.savefig('plots/Lo_Faro_LIRG_mstar_qir.pdf')
    plt.show()
    plt.clf()
    
  
    ###Creating plots for ULIRGs

    fig, ax = plt.subplots(1,2)
    
    mid_rad,med_fir,low_fir,high_fir = median_line(rad_SU,lir_SU,False,False)
    #ax[0,0].hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    ax[0].plot(mid_rad,med_fir,'blue')
    ax[0].fill_between(mid_rad,low_fir,high_fir,color = 'blue',alpha = 0.5)
    ax[0].errorbar(rad_U,lir_U, xerr = 0.13,yerr = 0.13,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'Lor Faro et al.',ecolor='black',elinewidth = 0.5)
    ax[0].set_xlabel('L$_{rad/1.4GHz}$/ W/Hz',fontsize = 12)
    ax[0].set_ylabel('(L$_{IR}$) W',fontsize = 12)
    
    mid_m,med_sfr,low_sfr,high_sfr = median_line(mst_SU,sfr_SU,False,False)
    ax[1].plot(mid_m,med_sfr,'blue')
    ax[1].fill_between(mid_m,low_sfr,high_sfr,color = 'blue',alpha = 0.5)
    ax[1].errorbar(mst_U,sfr_U, xerr =0.2,yerr = 0.2,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5)
    ax[1].set_xlabel("Stellar Mass/$M_\odot$",fontsize = 12)
    ax[1].set_ylabel("SFR/$M_\odot$/yr",fontsize = 12)

    
    fig.set_size_inches(11, 5)
    plt.tight_layout()
    plt.savefig('plots/Lo_Faro_ULIRG.pdf')
    plt.show()
    
    plt.clf()
    
    
    mid_m,med_qir,low_qir,high_qir = median_line(mst_SU,qir_SU,False,False)
    plt.plot(mid_m,med_qir,'blue')
    plt.fill_between(mid_m,low_qir,high_qir,color = 'blue',alpha = 0.5, label = 'SHARK')
    plt.errorbar(mst_U,qir_U, xerr = 0.2,yerr = 0.26,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.5)
    plt.xlabel("Stellar Mass/$M_\odot$",fontsize = 12)
    plt.ylabel("qir",fontsize = 12)
    fig.set_size_inches(10, 10)
    plt.savefig('plots/Lo_Faro_ULIRG_mstar_qir.pdf')
    plt.show()
    plt.clf()

def qir_plots(df):
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    
    c = plt.hexbin(mstars,qir,mincnt = 100, cmap='rainbow', gridsize = 30)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel('qir')
    plt.title('qir vs. Stellar Mass')
    cb = plt.colorbar(c)
    cb.set_label('Number Count')
    
    plt.show()
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'Teff']
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    
    c = plt.hexbin(mstars,qir,C=teff,mincnt = 100, cmap='rainbow', gridsize = 30)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel('qir')
    plt.title('qir vs. Stellar Mass')
    cb = plt.colorbar(c)
    cb.set_label('Number Count')
    
    plt.show()
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    fir = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'fir_flux'])
    rad = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'bress_rad_lum'])
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    fig, ax = plt.subplots(1,2)
    
    c = ax[0].hexbin(mstars,qir,C=fir,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(fir),vmax = max(fir))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[0]])
    cb1.set_label('Log10(FIR/W)')
    
    c = ax[1].hexbin(mstars,qir,C=rad,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(rad),vmax = max(rad))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[1]])
    cb1.set_label('Log10(Lrad/W/Hz)')
    
    plt.show()
    
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    fir = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'fir_flux'])
    rad = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'bress_rad_lum'])
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    fig, ax = plt.subplots(1,2)
    
    fir = fir - mstars
    rad = rad - mstars
    
    
    c = ax[0].hexbin(mstars,qir,C=fir,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(fir),vmax = max(fir))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[0]])
    cb1.set_label('Log10(FIR/W) - Log10(Stellar Mass/$M_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C=rad,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(rad),vmax = max(rad))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[1]])
    cb1.set_label('Log10(Lrad/W/Hz) -  Log10(Stellar Mass/$M_\odot$)')
    
    plt.show()
    
    
    fig, ax = plt.subplots(1,2)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'Teff']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(metallicity,qir,False,False)
    ax[0].hexbin(metallicity,qir,C=teff,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Gas Metallicity)")
    ax[0].set_ylabel('qir')
    ax[0].set_title('Centrals')
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'Teff']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(metallicity,qir,False,False)
    
    c=ax[1].hexbin(metallicity,qir,C=teff,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Gas Metallicity)")
    ax[1].set_ylabel('qir')
    ax[1].set_title('Sattellites')
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Effective Dust Temperature (K)')
    
    plt.show()

    
def qir_mstar(df):
    mlst = [1e8,1e9,1e10,1e11,1e12]
    df_line = pd.DataFrame()
    fig, ax = plt.subplots(1,2)
    
    colours = ['blue','red','orange','magenta','teal','cyan','olive','yellow','grey','white']
    
    for m in range(len(mlst)-1):
        
        m1 = mlst[m]
        m2 = mlst[m+1]
        
        qir_all_med = []
        qir_all_upp = []
        qir_all_low = []
        
        qir_sfg_med = []
        qir_sfg_upp = []
        qir_sfg_low = []
        ###getting the data at different redshifts###
        for z in zlist:
            
            qir_all = df.loc[((df['mstars_tot']>m1)&(df['mstars_tot']<m2)&(df['z']==z)),'qir_bress'] #all galaxies
            try:
                qir_all_med.append(np.median(qir_all))
                qir_all_upp.append(np.percentile(qir_all,84))
                qir_all_low.append(np.percentile(qir_all,16))
            except:
                qir_all_med.append(np.nan)
                qir_all_upp.append(np.nan)
                qir_all_low.append(np.nan)                
            
            
            qir_sfg = df.loc[((df['mstars_tot']>m1)&(df['mstars_tot']<m2)&(df['z']==z)&(df['sfg/q'] =='sf')),'qir_bress'] #all SFG
            try:
                qir_sfg_med.append(np.median(qir_sfg))
                qir_sfg_upp.append(np.percentile(qir_sfg,84))
                qir_sfg_low.append(np.percentile(qir_sfg,16))            
            except:
                qir_sfg_med.append(np.nan)
                qir_sfg_upp.append(np.nan)
                qir_sfg_low.append(np.nan)   
                
        qir_all_med_lab = 'qir_all_med_m1_' + str(m1)
        qir_all_low_lab = 'qir_all_med_m1_' + str(m1)
        qir_all_upp_lab = 'qir_all_med_m1_' + str(m1) 

        qir_sfg_med_lab = 'qir_sfg_med_z_' + str(m1) 
        qir_sfg_low_lab = 'qir_sfg_med_z_' + str(m1) 
        qir_sfg_upp_lab = 'qir_sfg_med_z_' + str(m1)

        df_line[qir_all_med_lab] = qir_all_med
        df_line[qir_all_low_lab] = qir_all_low
        df_line[qir_all_upp_lab] = qir_all_upp       

        df_line[qir_sfg_med_lab] = qir_sfg_med
        df_line[qir_sfg_low_lab] = qir_sfg_low
        df_line[qir_sfg_upp_lab] = qir_sfg_upp      
        
        
        ###plotting the different mass bins###
        label = str(np.log10(m1)) + " $\leq$ $Log_{10}$(M/$M_\odot$) $\leq$ " + str(np.log10(m2))
        ax[0].plot(zlist,qir_all_med,color = black,linewidth = 5)
        ax[0].plot(zlist,qir_all_med,color = colours[m],linewidth = 3,label = label)
        
        ax[0].fill_between(zlist,qir_all_upp,qir_all_low,color = colours[m],alpha = 0.5)
        ax[1].plot(zlist,qir_sfg_med,color = black,linewidth = 5,label = label)        
        ax[1].plot(zlist,qir_sfg_med,color = colours[m],linewidth = 3,label = label)
        ax[1].fill_between(zlist,qir_sfg_upp,qir_sfg_low,color = colours[m],alpha = 0.5)
    
    ax[0].text(1,3.25,'All Galaxies')
    ax[1].text(1,3.25,'SFGs')    
    
    ax[0].set_xlabel("z",fontsize = 20)
    ax[1].set_xlabel("z",fontsize = 20)
    ax[0].set_ylabel("qir",fontsize = 20)
    
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,3.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,3.5)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15) 
    ax[1].tick_params(labelleft=False)
    leg = ax[1].legend(loc='lower left',frameon = False)
    leg.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(15, 10)
    plt.savefig('plots/qir_z_mstar.pdf')
    plt.show()
        
    df_line.to_csv('qir_z_mstar_lines.csv')

def qir_mstar_ulirgs(df):
    mlst = [1e8,1e9,1e10,1e11,1e12]
    df_line = pd.DataFrame()
    fig, ax = plt.subplots(1,2)
    
    colours = ['blue','red','orange','magenta','teal','cyan','olive','yellow','grey','white']
    
    for m in range(len(mlst)-1):
        
        m1 = mlst[m]
        m2 = mlst[m+1]
        
        qir_all_med = []
        qir_all_upp = []
        qir_all_low = []
        
        qir_sfg_med = []
        qir_sfg_upp = []
        qir_sfg_low = []
        ###getting the data at different redshifts###
        for z in zlist:
            
            qir_all = df.loc[((df['mstars_tot']>m1)&(df['mstars_tot']<m2)&(df['z']==z)&(df['fir_lum']>1e11*Lsun)&(df['fir_lum']<1e12*Lsun),'qir_bress')] #lirgs
            try:
                qir_all_med.append(np.median(qir_all))
                qir_all_upp.append(np.percentile(qir_all,84))
                qir_all_low.append(np.percentile(qir_all,16))
            except:
                qir_all_med.append(np.nan)
                qir_all_upp.append(np.nan)
                qir_all_low.append(np.nan)                
            
            
            qir_sfg = df.loc[((df['mstars_tot']>m1)&(df['mstars_tot']<m2)&(df['z']==z)&(df['fir_lum']>1e12*Lsun)&(df['fir_lum']<1e13*Lsun)),'qir_bress'] #ulirgs
            try:
                qir_sfg_med.append(np.median(qir_sfg))
                qir_sfg_upp.append(np.percentile(qir_sfg,84))
                qir_sfg_low.append(np.percentile(qir_sfg,16))            
            except:
                qir_sfg_med.append(np.nan)
                qir_sfg_upp.append(np.nan)
                qir_sfg_low.append(np.nan)   
                
        qir_all_med_lab = 'qir_lirg_med_m1_' + str(m1)
        qir_all_low_lab = 'qir_lirg_low_m1_' + str(m1)
        qir_all_upp_lab = 'qir_lirg_upp_m1_' + str(m1) 

        qir_sfg_med_lab = 'qir_ulirg_med_z_' + str(m1) 
        qir_sfg_low_lab = 'qir_ulirg_low_z_' + str(m1) 
        qir_sfg_upp_lab = 'qir_ulirg_upp_z_' + str(m1)

        df_line[qir_all_med_lab] = qir_all_med
        df_line[qir_all_low_lab] = qir_all_low
        df_line[qir_all_upp_lab] = qir_all_upp       

        df_line[qir_sfg_med_lab] = qir_sfg_med
        df_line[qir_sfg_low_lab] = qir_sfg_low
        df_line[qir_sfg_upp_lab] = qir_sfg_upp      
        
        
        ###plotting the different mass bins###
        label = str(np.log10(m1)) + " $\leq$ $Log_{10}$(M/$M_\odot$) $\leq$ " + str(np.log10(m2))
        ax[0].plot(zlist,qir_all_med,color = black,linewidth = 5)
        ax[0].plot(zlist,qir_all_med,color = colours[m],linewidth = 3,label = label)
        
        ax[0].fill_between(zlist,qir_all_upp,qir_all_low,color = colours[m],alpha = 0.5)
        ax[1].plot(zlist,qir_sfg_med,color = black,linewidth = 5,label = label)        
        ax[1].plot(zlist,qir_sfg_med,color = colours[m],linewidth = 3,label = label)
        ax[1].fill_between(zlist,qir_sfg_upp,qir_sfg_low,color = colours[m],alpha = 0.5)
    
    ax[0].text(1,3.25,'All Galaxies')
    ax[1].text(1,3.25,'SFGs')    
    
    ax[0].set_xlabel("z",fontsize = 20)
    ax[1].set_xlabel("z",fontsize = 20)
    ax[0].set_ylabel("qir",fontsize = 20)
    
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,3.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,3.5)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15) 
    ax[1].tick_params(labelleft=False)
    leg = ax[1].legend(loc='lower left',frameon = False)
    leg.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(15, 10)
    plt.savefig('plots/qir_z_mstar_ulirgs.pdf')
    plt.show()
        
    df_line.to_csv('qir_z_mstar_ulirgs.csv')

                              
def sfr_m_hex(df):
    
    z = 0
    
    mst = np.log10(df.loc[((df['z']==z)&(df['mstars_tot'] > 10**(8))&(df['mstars_tot'] < 10**(12)),'mstars_tot')])
    qir = df.loc[((df['z']==z)&(df['mstars_tot'] > 10**(8))&(df['mstars_tot'] < 10**(12)),'qir_bress')] 
    sfr = np.log10(df.loc[((df['z']==z)&(df['mstars_tot'] > 10**(8))&(df['mstars_tot'] < 10**(12)),'sfr')])                              
    mid_mst,med_sfr,low_sfr,upp_sfr = median_line(mst,sfr,False,False)
    
    for z in zlist:
        print("Finding SFG for z = ",str(round(z,2)))
        
        sfg_df = df[['sfr','mstars_tot']] [((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)) ]
        
        sfr1 = sfg_df['sfr']
        mst1 = sfg_df['mstars_tot']
        sfr1 = np.log10(sfr1)
        mst1 = np.log10(mst1)

        a,b = np.polyfit(mst1,sfr1,1)
    
    
    fig, ax = plt.subplots(1,2)             
    
    mst_def = np.linspace(8,12)
    sfr_def = a*mst_def + b
    
    
    c = ax[0].hexbin(mst,sfr,cmap = 'rainbow', gridsize = 30,mincnt = 10)
    ax[0].plot(mid_mst,med_sfr,color = 'black',linewidth = 5)
    ax[0].plot(mid_mst,upp_sfr,color = 'black',linewidth = 5,linestyle = 'dashed')    
    ax[0].plot(mid_mst,low_sfr,color = 'black',linewidth = 5,linestyle = 'dashed')
    ax[0].plot(mst_def,sfr_def,color = 'red',linewidth = 3,path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    ax[0].set_ylabel('SFR',fontsize = 20)
    ax[0].set_ylabel('Stellar Mass',fontsize = 20)   
    
    cb = fig.colorbar(c,ax=ax[0])
    cb.set_label('Number Count',fontsize = 20)
    
    ax[0].set_xlim(8,12)
    
    
    c = ax[1].hexbin(mst,sfr,C=qir, gridsize = 30,reduce_C_function=np.median,mincnt = 10)
    ax[1].plot(mid_mst,med_sfr,color = 'black',linewidth = 5)
    ax[1].plot(mid_mst,upp_sfr,color = 'black',linewidth = 5,linestyle = 'dashed')    
    ax[1].plot(mid_mst,low_sfr,color = 'black',linewidth = 5,linestyle = 'dashed')   
    ax[1].plot(mst_def,sfr_def,color = 'red',linewidth = 3,path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    cb = fig.colorbar(c,ax=ax[1])
    cb.set_label('Number Count',fontsize = 20)
    ax[1].set_ylabel('SFR',fontsize = 20)
    ax[1].set_ylabel('Stellar Mass',fontsize = 20)       
    ax[1].set_xlim(8,12)
        
    #plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(12, 12)
    plt.tight_layout()   
    plt.savefig("plots/sfr_m_hex.pdf")
    plt.show()
                              
                              
def central_counter(lst):
    lst_len = len(lst)
    counter = 0
    for i in lst:
        if i == 0:
            counter += 1
    
    return counter/lst_len
    

def met_dist_cent(df):
    
    fig, ax = plt.subplots(1,3)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'Teff']
    typ = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'type']
    dist_ms = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sf_test']
    
    ssfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr']) - mstars
    
    
    n = 1
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0].hexbin(mstars,qir,C = metallicity,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[0]])
    cb1.set_label('$Log_{10}$(Gas Metallicity/$Z_{gas}/Z_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C = ssfr,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median,vmin = -12,vmax = -10)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[1]])
    cb1.set_label('SSFR')
    
    c = ax[2].hexbin(mstars,qir,C = typ,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=central_counter)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[2].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[2]])
    cb1.set_label('Central Fraction')
    fig.suptitle("qir vs. stellar mass for all galaxies")
    plt.show()
    
    fig, ax = plt.subplots(1,3)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'Teff']
    typ = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'type']
    dist_ms = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] =='sf')),'sf_test']
    
    
    
    
    n = 1
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0].hexbin(mstars,qir,C = metallicity,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[0]])
    cb1.set_label('$Log_{10}$(Gas Metallicity/$Z_{gas}/Z_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C = dist_ms,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[1]])
    cb1.set_label('Distance to Main Sequence/$log_{10}$ (M/yr)')
    
    c = ax[2].hexbin(mstars,qir,C = typ,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=central_counter)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[2].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[2]])
    cb1.set_label('Central Fraction')
    fig.suptitle("qir vs. stellar mass for star-forming galaxies")
    plt.show()

    
def ionising_phot_sfr(df):
    
    q_h = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'ionising_photons'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr'])
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(q_h,sfr,False,False)
    n = 1
    plt.hexbin(q_h,sfr,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Q(H))")
    plt.ylabel("Log10(SFR)")
    plt.title("SFR vs. Q(H)")
    plt.show()
    
    
def SFR_z(df):
    
    sfr_med = []
    sfr_upper = []
    sfr_lower = []
    zlst = []
    
    for z in zlist:
    
        sfr = np.log10(df.loc[(df['z'] == z),'sfr'])
        
        sfr_med.append(np.median(sfr))
        sfr_upper.append(np.percentile(sfr,84))
        sfr_lower.append(np.percentile(sfr,16))

    plt.plot(zlist,sfr_med,'black')
    plt.plot(zlist,sfr_upper,'black',linestyle='dashed')
    plt.plot(zlist,sfr_lower,'black',linestyle='dashed')
    plt.title("SFR vs. z")
    plt.show()
    
    qir = df.loc[((df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)),'qir']

    
def sfr_function(df,h0,volh):
    vol = volh/h0**3
    mlow = 8
    mupp = 12
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    df_line = pd.DataFrame()
    slow = -3
    supp = 3
    
    sf_df = pd.read_csv('SFR_function2.csv')
    sf_df['z'] = round(sf_df['Z'])
    sf_df['source_2'] = sf_df['source'] + ' z = ' + sf_df['Z'].astype(str)

    zlst = [0,1,2,3,5,6,7,8]
    fig, ax = plt.subplots(3,3)
    fmt_lst = ['o','^','s','P','p','X','D','>','<','v']
    col_lst = ['blue','red','orange','magenta','teal','cyan','olive','yellow','grey','white']
    source_lst = np.sort(np.unique(sf_df['source'])) #source list
    source_lst2 = np.sort(np.unique(sf_df['source_2'])) #source list
    q = 0
    
    
    for z in zlst:
        g = q//3
        h = q%3
        sbins = np.arange(slow, supp, dm)
        xmf = sbins + dm/2.0 #setting up the bins
        sfr = np.log10(df.loc[((round(df['z']) == z)),'sfr'])
        n = 100
        high_sfr,low_sfr,med_sfr = bootstrap(sfr,sbins,supp,n,False)
        
        med_sfr = np.log10(med_sfr/vol/dm)
        high_sfr = np.log10(high_sfr/vol/dm)
        low_sfr = np.log10(low_sfr/vol/dm)
        
        bin_lab = 'sbins_'+str(round(z))
        med_lab = 'med_sfr'+str(round(z))        
        low_lab = 'low_sfr'+str(round(z))
        upp_lab = 'upp_sfr'+str(round(z))
        
        df_line[bin_lab] = sbins
        df_line[med_lab] = med_sfr
        df_line[low_lab] = low_sfr
        df_line[upp_lab] = high_sfr
        
        ax[g,h].plot(xmf,med_sfr,color = 'blue',label = 'SHARK')
        ax[g,h].fill_between(xmf,low_sfr,high_sfr,color = 'blue',alpha = 0.5)


        sf_df2 = sf_df[['SFR','SFRF','SFR_err_upp','SFR_err_low','source_2','source']] [((sf_df['z'] == z))]
        k = 0
    
        sou_lst = np.sort(np.unique(sf_df2['source'])) #source list
        sou_lst2 = np.sort(np.unique(sf_df2['source_2'])) #source list
        for source, source2 in zip(sou_lst,sou_lst2):
            idx = np.where(source_lst == source)[0][0]

            sf_df3 = sf_df2[['SFR','SFRF','SFR_err_upp','SFR_err_low']] [((sf_df2['source_2'] == source2))]

            sfr = np.log10(sf_df3['SFR'])
            sfr_err = 0.3

            sfrf = np.log10(sf_df3['SFRF']*10**(-2))
            sfrf_err_upp = np.log10(sf_df3['SFRF']*10**(-2) + sf_df3['SFR_err_upp']*10**(-2)) - sfrf
            sfrf_err_low = sfrf - np.log10(sf_df3['SFRF']*10**(-2) - sf_df3['SFR_err_low']*10**(-2))

            sfrf_err = [sfrf_err_low,sfrf_err_upp]
            fmt = fmt_lst[idx]
            col = col_lst[idx]
            label = source2
            tit = 'z = ' + str(z)

            ax[g,h].errorbar(sfr,sfrf,xerr = sfr_err,yerr=sfrf_err, fmt=fmt,markerfacecolor=col, markeredgecolor='black',label = label,elinewidth=2,ecolor = 'black')
            ax[g,h].set_xlim(-4.3,2.8)
            ax[g,h].set_ylim(-6.3,0.3)
            ax[g,h].set_xlabel('$Log_{10}$(SFR)[$M_{\odot} yr^{-1}]$',size = 15)
            ax[g,0].set_ylabel('$\dfrac{dn}{dLog_{10}(\phi_{SFR})}[Mpc^{-3}]$',size = 15)
            ax[g,h].tick_params(bottom = True, top = True,direction = 'inout',labelsize=15)
            ax[g,h].tick_params(which = 'minor',bottom = True, top = True,direction = 'inout')
            ax[0,h].tick_params(labelbottom=False, top = True,labelsize=15) 
            ax[0,h].tick_params(top = False)
            ax[g,1].tick_params(labelleft=False)
            ax[g,2].tick_params(labelleft=False)
            ax[g,h].minorticks_on()
            ax[g,h].text(1,-1,tit)
            leg = ax[g,h].legend(frameon = False)
            leg.get_frame().set_linewidth(0.0)
            k+= 1
        q += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(20, 20)
    plt.savefig('plots/sfr_function.pdf')
    plt.show()
    
    df_line.to_csv('sfr_func_SHARK.csv')




    
def all_galaxies_gas_disks(df):
    
    n = 1
    
    fig, ax = plt.subplots(2,4)
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'qir_bress']
    mmol_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mmol_disk'])
    matom_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'matom_disk'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'Teff']
    r_gas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'r_gas_disk'])
    mgas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mgas_disk'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr'])
    mgas_metals_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mgas_metals_disk'])
    gas_surf_dens = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'gas_surf_dens'])
   
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0,0].hexbin(mstars,qir,C=mmol_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
   # ax[0,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,0]])
    cb1.set_label('Log10(Molecular Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,1].hexbin(mstars,qir,C=matom_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[0,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,1]])
    cb1.set_label('Log10(Atomic Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,2].hexbin(mstars,qir,C=mgas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,2]])
    cb1.set_label('Log10(Total Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,3].hexbin(mstars,qir,C=mgas_metals_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,3]])
    cb1.set_label('Log10(Total Metals in Disk Gas/$M_\odot$)')
    
    c = ax[1,0].hexbin(mstars,qir,C=teff,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,0]])
    cb1.set_label('Effective Dust Temperature/K')
    
    c = ax[1,1].hexbin(mstars,qir,C=r_gas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,1]])
    cb1.set_label('Log10(Half-mass radius of the gas disk/Mpc)')
    
    c = ax[1,2].hexbin(mstars,qir,C=sfr,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,2]])
    cb1.set_label("Log10(SFR/$M_\odot$/yr)")
    
    c = ax[1,3].hexbin(mstars,qir,C=gas_surf_dens,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[1,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,3]])
    cb1.set_label("Log10(Gas Surface Density/$M_\odot$/$Mpc^2$)")
    plt.suptitle("qir vs. total stellar mass - all galaxies")
    fig.set_size_inches(18.5, 10.5)
    fig.savefig("all_qir_m_disk_8_9.png")
    plt.show()
        

        
def sf_galaxies_gas_disks(df):
    
    n = 1
    
    fig, ax = plt.subplots(2,4)
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'qir_bress']
    mmol_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'mmol_disk'])
    matom_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'matom_disk'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'Teff']
    r_gas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'r_gas_disk'])
    mgas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'mgas_disk'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'sfr'])
    mgas_metals_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'mgas_metals_disk'])
    gas_surf_dens = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfg/q'] == 'sf')),'gas_surf_dens'])
   
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0,0].hexbin(mstars,qir,C=mmol_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
   # ax[0,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,0]])
    cb1.set_label('Log10(Molecular Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,1].hexbin(mstars,qir,C=matom_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[0,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,1]])
    cb1.set_label('Log10(Atomic Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,2].hexbin(mstars,qir,C=mgas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,2]])
    cb1.set_label('Log10(Total Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,3].hexbin(mstars,qir,C=mgas_metals_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,3]])
    cb1.set_label('Log10(Total Metals in Disk Gas/$M_\odot$)')
    
    c = ax[1,0].hexbin(mstars,qir,C=teff,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,0]])
    cb1.set_label('Effective Dust Temperature/K')
    
    c = ax[1,1].hexbin(mstars,qir,C=r_gas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,1]])
    cb1.set_label('Log10(Half-mass radius of the gas disk/Mpc)')
    
    c = ax[1,2].hexbin(mstars,qir,C=sfr,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,2]])
    cb1.set_label("Log10(SFR/$M_\odot$/yr)")
    
    c = ax[1,3].hexbin(mstars,qir,C=gas_surf_dens,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[1,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,3]])
    cb1.set_label("Log10(Gas Surface Density/$M_\odot$/$Mpc^2$)")
    plt.suptitle("qir vs. total stellar mass - all galaxies")
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("sf_qir_m_disk_8_9.png")
    plt.show()

def rad_lum_func_plt_z0(df,h0,volh):
    
    
    
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    df_line = pd.DataFrame()
    df_line['xmf'] = xmf
    
    df2 = pd.read_csv("bonata_data_z0.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
    butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
    butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
    butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    n = 100
    
    
    
    bress_rad_lum = df.loc[(df['z'] == 0),'rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    high_lst,low_lst,med_lst = bootstrap(np.array(blum),mbins,mupp,n,False)
    
    high_lst = high_lst/vol/dm
    low_lst = low_lst/vol/dm
    med_lst = med_lst/vol/dm

    df_line['upp_noerr'] = high_lst
    df_line['low_noerr'] = low_lst
    df_line['med_noerr'] = med_lst
    
    high_err,low_err,med_err = bootstrap(np.array(blum),mbins,mupp,n,True)
    
    high_err = high_err/vol/dm
    low_err = low_err/vol/dm
    med_err = med_err/vol/dm

    df_line['upp_err'] = high_err
    df_line['low_err'] = low_err
    df_line['med_err'] = med_err    
    
    
    
    plt.plot(xmf,np.log10(med_lst),'blue')
    plt.fill_between(xmf,np.log10(low_lst),np.log10(high_lst),color = 'blue',alpha = 0.5)
    plt.plot(xmf,np.log10(med_err),'orange')
    plt.fill_between(xmf,np.log10(low_err),np.log10(high_err),color = 'orange',alpha = 0.5)
    
    
    
    
   # plt.plot(xmf,np.log10(bhist),'black',linestyle = 'dashed')
    #plt.plot(xmf,np.log10(bhist2),'red')

    plt.errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    plt.errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    plt.errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    plt.errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    plt.xlabel("log$_{10}$(L$_{1.4GHz}$) (W/Hz)",size=13)
    plt.ylabel("log$_{10}$(Φ) (Mpc$^{-3}$ dex$^{-1}$)",size=13)
    plt.xlim(19,25)
    plt.ylim(-7,0)
    plt.legend()
    plt.savefig("plots/rad_lum_func_z0.pdf",format='pdf')
    plt.show()
    df_line.to_csv('1d4GHz_rad_lum_func_0_data.csv')
    
def bootstrap(ary,mbins,mupp,n,convolve):
    hist_array = np.zeros((n, len(mbins))) #creates an array that will be filled
    for i in range(n):
        if convolve == True:    
            random_ary = np.random.choice(ary, size=ary.shape, replace=True) + np.random.normal(loc=0.0, scale=0.3, size=len(ary)) #makes a random list from the provided list with replacement. Convolves it with a random gaussian
        else:
            random_ary = np.random.choice(ary, size=ary.shape, replace=True) #makes a random list from the provided list with replacement.
        hist_array[i] = np.histogram(random_ary,bins=np.append(mbins,mupp))[0] #creates a histogram                                                                                                                                                                                
    high = np.percentile(hist_array,84,axis = 0)
    low = np.percentile(hist_array,16,axis = 0)
    med = np.median(hist_array,axis = 0)
    return high,low,med

def rad_lum_func_agn(df,h0,volh):
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    df_line = pd.DataFrame()
    df_line['xmf'] = xmf
    #print("This is bress rad lum lst")
    #print(bress_rad_lum_lst)
    
    df2 = pd.read_csv("bonata_data_z0.csv")
    
#     bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
#     bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
#     bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
#     bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
#     butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
#     butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
#     butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
#     butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']

#     ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
#     ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
#     ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
#     ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    ###Redshift Plots###

    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
   
    fig, ax = plt.subplots(2,2)

 #   zlist = [0.909822023685613, 2.00391410007239, 3.0191633709527,0]

    zlist = [0,0.909822023685613, 2.00391410007239,3.0191633709527,3.95972701662501]
    q = 0
    n = 10
    for z in zlist:
        
        g = q//2
        h = q%2
        q += 1
        bon_data = 'bonata_data_z' + str(round(z)) + '.csv'
    
        df2 = pd.read_csv(bon_data)
#         try:
#             bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
#             bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
#             bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
#             bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
#         except:
#             bonatoy = np.NaN
#             bonatox = np.NaN
#             bonatoerr_up = np.NaN
#             bonatoerr_down = np.NaN
            
#         try:
#             butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
#             butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
#             butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
#             butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']
#         except:
#             butlery = np.NaN
#             butlerx = np.NaN
#             butlererr_up = np.NaN
#             butlererr_down = np.NaN
        
#         try:
#             ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
#             ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
#             ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
#             ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
#         except:
#             ocrany = np.NaN
#             ocranx = np.NaN
#             ocranerr_up = np.NaN
#             ocranerr_down = np.NaN
        
        try:
            novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
            novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
            novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
            novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
        except:
            novaky = np.NaN
            novakx = np.NaN
            novakerr_up = np.NaN
            novakerr_down = np.NaN

        n = 100
    
        bress_rad_lum = df.loc[(df['z'] == z),'rad_lum']
        agn_rad = df.loc[(df['z'] == z),'agn_rad']
        
        blum = np.log10(bress_rad_lum) #logging the dataset
    
        high_err,low_err,med_err = bootstrap(np.array(blum),mbins,mupp,n,True)
        high_lst,low_lst,med_lst = bootstrap(np.array(blum),mbins,mupp,n,False)   
        
        high_lst = high_lst/vol/dm
        low_lst = low_lst/vol/dm
        med_lst = med_lst/vol/dm
        
        upp_lab = 'upp_noerr_' + str(round(z))
        low_lab = 'low_noerr_' + str(round(z))
        med_lab = 'med_noerr_' + str(round(z))
        
        df_line[upp_lab] = high_lst
        df_line[low_lab] = low_lst
        df_line[med_lab] = med_lst

        high_err = high_err/vol/dm
        low_err = low_err/vol/dm
        med_err = med_err/vol/dm
        
        upp_lab = 'upp_err_' + str(round(z))
        low_lab = 'low_err_' + str(round(z))
        med_lab = 'med_err_' + str(round(z))
        
        df_line[upp_lab] = high_err
        df_line[low_lab ] = low_err
        df_line[med_lab] = med_err  
        

    

      #  ax[g,h].plot(xmf,np.log10(med_lst),'blue',label = 'SFGs')
      #  ax[g,h].fill_between(xmf,np.log10(low_lst),np.log10(high_lst),color = 'blue',alpha = 0.5)
      #  ax[g,h].plot(xmf,np.log10(med_err),'orange',label = 'SFGs + 0.3 dex')
      #  ax[g,h].fill_between(xmf,np.log10(low_err),np.log10(high_err),color = 'orange',alpha = 0.5)
        
        agn_rad = df.loc[(df['z'] == z),'agn_rad']
        
        alum = np.log10(agn_rad) #logging the dataset
    
        upp_agn,low_agn,med_agn = bootstrap(np.array(alum),mbins,mupp,n,False)
        
        upp_agn = upp_agn/vol/dm
        low_agn = low_agn/vol/dm
        med_agn = med_agn/vol/dm
        

        upp_lab = 'upp_agn_' + str(round(z))
        low_lab = 'low_agn_' + str(round(z))
        med_lab = 'med_agn_' + str(round(z))
        
        df_line[upp_lab] = upp_agn
        df_line[low_lab ] = low_agn
        df_line[med_lab] = med_agn 
        
        
     #   ax[g,h].plot(xmf,np.log10(med_agn),'red',label = 'AGNs')
     #   ax[g,h].fill_between(xmf,np.log10(low_agn),np.log10(upp_agn),color = 'red',alpha = 0.5)
        
        tot_rad = df.loc[(df['z'] == z),'rad_tot']        
        
        
        tlum = np.log10(tot_rad) #logging the dataset
    
        upp_tot,low_tot,med_tot = bootstrap(np.array(tlum),mbins,mupp,n,False)
        
        upp_tot = upp_tot/vol/dm
        low_tot = low_tot/vol/dm
        med_tot = med_tot/vol/dm
        
        
     #   ax[g,h].plot(xmf,np.log10(med_tot),'green',label = 'AGNs + SFGs')
      #  ax[g,h].fill_between(xmf,np.log10(low_tot),np.log10(upp_tot),color = 'green',alpha = 0.5)
        
        
        upp_lab = 'upp_tot_' + str(round(z))
        low_lab = 'low_tot_' + str(round(z))
        med_lab = 'med_tot_' + str(round(z))
        
        df_line[upp_lab] = upp_tot
        df_line[low_lab ] = low_tot
        df_line[med_lab] = med_tot  
        
        r = 22*(1+z)**(0.013)
        
        print("This is r")
        print(r)
        nov_rad = df.loc[((df['z'] == z)&(df['novak_r'] < r)),'rad_tot']
        
        

        nlum = np.log10(nov_rad) #logging the dataset
        
        print("This is nov_rad")
        print(nov_rad)
        
        
        
        upp_nov,low_nov,med_nov = bootstrap(np.array(nlum),mbins,mupp,n,False)
        
        upp_nov = upp_nov/vol/dm
        low_nov = low_nov/vol/dm
        med_nov = med_nov/vol/dm
        
        
      #  ax[g,h].plot(xmf,np.log10(med_nov),'purple',label = 'Novak et al. selection')
       # ax[g,h].fill_between(xmf,np.log10(low_nov),np.log10(upp_nov),color = 'purple',alpha = 0.5)
        
        upp_lab = 'upp_nov_' + str(round(z))
        low_lab = 'low_nov_' + str(round(z))
        med_lab = 'med_nov_' + str(round(z))
        
        df_line[upp_lab] = upp_nov
        df_line[low_lab ] = low_nov
        df_line[med_lab] = med_nov 
        

        # ax[g,h].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="8",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020',markersize='10')
        # ax[g,h].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019',markersize='10')
        # ax[g,h].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020',markersize='10')
     #   ax[g,h].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020',markersize='10')
        
        z_label = 'z = '+ str(round(z))
        
     #   ax[1,h].set_xlabel("log$_{10}$(L$_{1.4GHz}$) (W/Hz)",size=20)
     #   ax[g,0].set_ylabel("log$_{10}$(Φ) (Mpc$^{-3}$ dex$^{-1}$)",size=20)
     #   ax[1,h].tick_params(bottom = True, top = True)
     #   ax[0,h].tick_params(labelbottom=False) 
     #   ax[g,1].tick_params(labelleft=False) 
     #   ax[g,h].text(24,-1,z_label) 

     #   ax[g,h].set_xlim(18.7,25.3)
     #   ax[g,h].set_ylim(-7.3,0.3)
      #  leg = ax[1,0].legend(frameon=False)
      #  leg.get_frame().set_linewidth(0.0)
  #  plt.subplots_adjust(wspace=0, hspace=0)
  #  fig.set_size_inches(12,12)

#    plt.savefig("plots/1d4GHz_rad_lum_func_multi_z_agn.pdf")
#    plt.show()
    
    df_line.to_csv('1d4GHz_rad_lum_func_z_data_agn.csv')


def rad_lum_func_z(df,h0,volh):
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    df_line = pd.DataFrame()
    df_line['xmf'] = xmf
    #print("This is bress rad lum lst")
    #print(bress_rad_lum_lst)
    
    df2 = pd.read_csv("bonata_data_z0.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
    butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
    butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
    butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    ###Redshift Plots###

    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
   
    fig, ax = plt.subplots(2,2)

    zlist = [0.909822023685613, 2.00391410007239, 3.0191633709527,3.95972701662501]
    q = 0
    n = 10
    for z in zlist:
        
        g = q//2
        h = q%2
        q += 1
        bon_data = 'bonata_data_z' + str(round(z)) + '.csv'
    
        df2 = pd.read_csv(bon_data)
        try:
            bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
            bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
            bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
            bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
        except:
            bonatoy = np.NaN
            bonatox = np.NaN
            bonatoerr_up = np.NaN
            bonatoerr_down = np.NaN
            
        try:
            butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
            butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
            butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
            butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']
        except:
            butlery = np.NaN
            butlerx = np.NaN
            butlererr_up = np.NaN
            butlererr_down = np.NaN
        
        try:
            ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
            ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
            ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
            ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
        except:
            ocrany = np.NaN
            ocranx = np.NaN
            ocranerr_up = np.NaN
            ocranerr_down = np.NaN
        
        try:
            novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
            novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
            novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
            novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
        except:
            novaky = np.NaN
            novakx = np.NaN
            novakerr_up = np.NaN
            novakerr_down = np.NaN

        n = 100
    
        bress_rad_lum = df.loc[(df['z'] == z),'rad_lum']

        blum = np.log10(bress_rad_lum) #logging the dataset
    
        high_err,low_err,med_err = bootstrap(np.array(blum),mbins,mupp,n,True)
        high_lst,low_lst,med_lst = bootstrap(np.array(blum),mbins,mupp,n,False)   
        
        high_lst = high_lst/vol/dm
        low_lst = low_lst/vol/dm
        med_lst = med_lst/vol/dm
        
        upp_lab = 'upp_noerr_' + str(round(z))
        low_lab = 'low_noerr_' + str(round(z))
        med_lab = 'med_noerr_' + str(round(z))
        
        df_line[upp_lab] = high_lst
        df_line[low_lab] = low_lst
        df_line[med_lab] = med_lst

        high_err = high_err/vol/dm
        low_err = low_err/vol/dm
        med_err = med_err/vol/dm
        
        upp_lab = 'upp_err_' + str(round(z))
        low_lab = 'low_err_' + str(round(z))
        med_lab = 'med_err_' + str(round(z))
        
        df_line[upp_lab] = high_err
        df_line[low_lab ] = low_err
        df_line[med_lab] = med_err  
        

    

        ax[g,h].plot(xmf,np.log10(med_lst),'blue',label = 'SHARK')
        ax[g,h].fill_between(xmf,np.log10(low_lst),np.log10(high_lst),color = 'blue',alpha = 0.5)
        ax[g,h].plot(xmf,np.log10(med_err),'orange',label = 'SHARK + 0.3 dex')
        ax[g,h].fill_between(xmf,np.log10(low_err),np.log10(high_err),color = 'orange',alpha = 0.5)

        ax[g,h].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="8",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020',markersize='10')
        ax[g,h].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019',markersize='10')
        ax[g,h].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020',markersize='10')
        ax[g,h].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='None', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020',markersize='10')
        
        z_label = 'z = '+ str(round(z))
        
        ax[1,h].set_xlabel("log$_{10}$(L$_{1.4GHz}$) (W/Hz)",size=20)
        ax[g,0].set_ylabel("log$_{10}$(Φ) (Mpc$^{-3}$ dex$^{-1}$)",size=20)
        ax[1,h].tick_params(bottom = True, top = True)
        ax[0,h].tick_params(labelbottom=False) 
        ax[g,1].tick_params(labelleft=False) 
        ax[g,h].text(24,-1,z_label) 

        ax[g,h].set_xlim(18.7,25.3)
        ax[g,h].set_ylim(-7.3,0.3)
        leg = ax[1,0].legend(frameon=False)
        leg.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(12,12)

    plt.savefig("plots/1d4GHz_rad_lum_func_multi_z.pdf")
    plt.show()
    
    df_line.to_csv('1d4GHz_rad_lum_func_z_data.csv')
    
def delta_qir(df):
    
        
    #zlst = [0, 0.194738848008908]
    q = 0
    
    gal_id_0 = df.loc[((df['z'] == 0)&(df['sfg/q'] == 'sf')),'galaxy_id']

    gal_id_19 = df.loc[((df['z'] == 0.194738848008908)&(df['sfg/q'] == 'sf')),'galaxy_id']
    gi = df.loc[df['galaxy_id'].isin(gal_id_0)]
    gal_id_19 = gi.loc[((df['z'] == 0.194738848008908)&(df['sfg/q'] == 'sf')),'galaxy_id']
    gi = gi.loc[gi['galaxy_id'].isin(gal_id_19)]

    gi = gi.sort_values(by=['galaxy_id'], ascending=True)
    qir_0 = gi.loc[(gi['z'] == 0),'qir_bress']    
    qir_19 = gi.loc[(gi['z'] == 0.194738848008908),'qir_bress']
    gal_id = gi.loc[(gi['z'] == 0),'galaxy_id']    
    delta_qir = np.array(qir_0) - np.array(qir_19)
    print("This is length of delta qir")
    print(len(delta_qir))
    print("This is the length of galaxy_id")
    print(len(gal_id))
    plt.hexbin(gal_id,delta_qir,mincnt = 4, cmap='rainbow', gridsize = 30) 
    plt.xlabel('galaxy id')
    plt.ylabel('delta qir')
    plt.title('z = 0 - z = 0.19')
    plt.savefig('plots/delta_qir_0_19.pdf')
    plt.show()
    
    
    gal_id_0 = df.loc[((df['z'] == 0.909822023685613)&(df['sfg/q'] == 'sf')),'galaxy_id']

    gal_id_19 = df.loc[((df['z'] == 0.194738848008908)&(df['sfg/q'] == 'sf')),'galaxy_id']
    gi = df.loc[df['galaxy_id'].isin(gal_id_0)]
    gal_id_19 = gi.loc[((df['z'] == 0.194738848008908)&(df['sfg/q'] == 'sf')),'galaxy_id']
    gi = gi.loc[gi['galaxy_id'].isin(gal_id_19)]

    gi = gi.sort_values(by=['galaxy_id'], ascending=True)
    qir_0 = gi.loc[(gi['z'] == 0.909822023685613),'qir_bress']    
    qir_19 = gi.loc[(gi['z'] == 0.194738848008908),'qir_bress']
    gal_id = gi.loc[(gi['z'] == 0.909822023685613),'galaxy_id']    
    delta_qir = np.array(qir_0) - np.array(qir_19)
    print("This is length of delta qir")
    print(len(delta_qir))
    print("This is the length of galaxy_id")
    print(len(gal_id))
    plt.hexbin(gal_id,delta_qir,mincnt = 4, cmap='rainbow', gridsize = 30) 
    plt.xlabel('galaxy id')
    plt.ylabel('delta qir')
    plt.title('z = 1 - z = 0.19')
    plt.savefig('plots/delta_qir_1_19.pdf')
    plt.show()

    
def delv_plt(df):
    
    df2 = pd.read_csv('Delvecchio_data.csv')
    print(df2)
    i = 0
    z = 0
    q = 0
    h = 0
    g = 0
    mlst = [8,9,9.5,10,10.5,11,12]
    fig, axs = plt.subplots(2, 3)
    for i in range(6):
        m = mlst[i]
        n = mlst[i+1]
        qir_bress_median_lst = []
        qir_median_lst = []
        qir_bress_low_lst = []
        qir_bress_high_lst = []
        qir_low_lst = []
        qir_high_lst = []
        zlst = []
        for z in zlist:
            qir_bress_sf = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))& (df['sfg/q'] == 'sf')&(df['z'] == z)),'qir_bress']
            qir_bress = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))&(df['z'] == z)),'qir_bress']
            try:
                qir_bress_low_lst.append(np.percentile(qir_bress_sf,16))
                qir_bress_high_lst.append(np.percentile(qir_bress_sf,84))
                qir_bress_median_lst.append(np.median(qir_bress_sf))
                
                qir_low_lst.append(np.percentile(qir_bress,16))
                qir_high_lst.append(np.percentile(qir_bress,84))
                qir_median_lst.append(np.median(qir_bress))
            except:
                qir_bress_low_lst.append(np.nan)
                qir_bress_high_lst.append(np.nan)
                qir_bress_median_lst.append(np.nan)
            
                qir_low_lst.append(np.nan)
                qir_high_lst.append(np.nan)
                qir_median_lst.append(np.nan)
            
        delv_df = df2[['qir_nondet','dqir_nondet','qir_all','dqir_all','zmean']] [((df2['mass_min'] == m) & (df2['mass_max'] == n)) ]          
        qir_nondet = delv_df['qir_nondet']
        qir_nondet_err = delv_df['dqir_nondet']
        qir_all = delv_df['qir_all']
        qir_all_err = delv_df['dqir_all']
        z_delv = delv_df['zmean']
        titlst = ['$10^{8}$< M/$M_\odot$ < $10^{9}$','$10^{9}$< M/$M_\odot$ < $10^{9.5}$','$10^{9.5}$< M/$M_\odot$ < $10^{10}$','$10^{10}$< M/$M_\odot$ < $10^{10.5}$','$10^{10.5}$< M/$M_\odot$ < $10^{11}$','$10^{11}$< M/$M_\odot$ < $10^{12}$']

        
        
        
        g = q//3
        h = q%3
        q +=1
        axs[g, h].plot(zlist,qir_bress_median_lst,color = 'red')
        axs[g, h].fill_between(zlist,qir_bress_low_lst,qir_bress_high_lst,alpha = 0.2, color= 'red',label = 'SFG')
        axs[g, h].plot(zlist,qir_median_lst,color = 'blue')
        axs[g, h].fill_between(zlist,qir_low_lst,qir_high_lst,alpha = 0.2, color= 'blue',label= 'All Galaxies')
        
        axs[g, h].errorbar(z_delv, qir_all, yerr=qir_all_err, fmt="o",label = 'Average individual detetections and stacked non-detections')
        axs[g, h].errorbar(z_delv, qir_nondet, yerr=qir_nondet_err, fmt="o",label = 'Stacks undetections')
        axs[g, h].set_ylim([0,4])
        axs[g, h].set_xlim([0,5])
        axs[g, h].set_title(titlst[i],fontsize = 12.5)
        for ax in axs.flat:
            ax.set(xlabel='z', ylabel='qir')
            ax.label_outer()
    plt.subplots_adjust(wspace=0, hspace=0)
    leg = axs[1,2].legend(loc='lower right', bbox_to_anchor=(0, 0))
    leg.get_frame().set_linewidth(0.0)
    fig.set_size_inches(12, 8)
    plt.savefig("plots/delv_plt.pdf")
    plt.show()

def ir_rad_qir_m(df,h):
    import matplotlib.colors as colors    
    ###defining colourmaps###
    cmap1 = colors.LinearSegmentedColormap.from_list("", ["darkred","lightgrey","darkblue"])
    cmap2 = colors.LinearSegmentedColormap.from_list("", ["salmon","lightgrey","cornflowerblue"])
    cmap3 = colors.LinearSegmentedColormap.from_list("", ["darkorange","lightgrey","cornflowerblue"])   
    
    
    cmap4 = colors.LinearSegmentedColormap.from_list("", ["green","lightgrey","darkmagenta"])
    cmap5 = colors.LinearSegmentedColormap.from_list("", ["lawngreen","lightgrey","violet"])
    cmap6 = colors.LinearSegmentedColormap.from_list("", ["teal","lightgrey","crimson"])    
    
    fig, ax = plt.subplots(3, 2)
    
    q = 0
    ###Reads in the data from the dataframe
    df2 = df[['qir_bress','mstars_tot','fir_lum','rad_lum']] [((df['z'] == 0) & (df['mstars_tot'] > 1e8)& (df['mstars_tot'] < 1e12)& (df['qir_bress'] < 3.3)& (df['qir_bress'] > 0)) ]
    df_line = pd.DataFrame()
    qir = df2['qir_bress']
    mst = np.log10(df2['mstars_tot'])
    fir = np.log10(df2['fir_lum'])
    fim = fir - mst
    rad = np.log10(df2['rad_lum'])
    ram = rad - mst
    
    fmed = np.median(fim)
    rmed = np.median(ram)
    
    fif = fim - fmed
    rif = ram - rmed
    
    ###Creates the median lines
    mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,False,False)
    
    df_line['mid_mst'] = mid_m
    df_line['med_qir'] = med_qir
    df_line['low_qir'] = low_qir
    df_line['upp_qir'] = high_qir
    
    parm_lst = [fir,fim,fif,rad,ram,rif]
    clab_lst = ["log$_{10}$ Lir/[W]", "log$_{10} Lir/M_{*}/[W/M_{\odot}$]",'$\Delta L_{IR}$',"log$_{10} L_{rad/1.4GHz}/[W]$","log$_{10} L_{rad/1.4GHz}/M_{*}/[W/M_{\odot}]$",'$\Delta L_{rad/1.4GHz}$']
    vmin_lst = [1,30,-3.5,12,-5,-8]
    vmax_lst = [4,50,-2.5,15,1,3]

    df_hex = pd.DataFrame()
    q = 0
    vmin_lst = [1,30,-3.5,12,-5,-8]
    vmax_lst = [4,50,-2.5,15,1,3]
    
    light_lir_cmap = cmocean.tools.lighten(cmo.amp, 0.75)
    lighter_lir_cmap = cmocean.tools.lighten(cmo.amp, 0.5)    

    light_rad_cmap = cmocean.tools.lighten(cmo.haline, 0.75)
    lighter_rad_cmap = cmocean.tools.lighten(cmo.haline, 0.5)    
    
    cmap_lst = [cmocean.cm.amp,light_lir_cmap,lighter_lir_cmap,cmocean.cm.haline,light_rad_cmap,lighter_rad_cmap]

    df_hex = pd.DataFrame()

    for q in range(6):
        h = q//3
        g = q%3
        print("This is g,h")
        print(g,h)
        parm = parm_lst[q]
        clab = clab_lst[q]
        cmap = cmap_lst[q]
        vmin = vmin_lst[q]
        vmax = vmax_lst[q]
        c = ax[g,h].hexbin(mst,qir,C = parm,cmap = cmap,mincnt = 10, gridsize = 30)
        ax[g,h].plot(mid_m,med_qir,'black')
        ax[g,h].plot(mid_m,low_qir,'black',linestyle='dashed')   
        ax[g,h].plot(mid_m,high_qir,'black',linestyle='dashed')
        ax[1,h].set_xlabel("$Log_{10}$(M_{*}/$M_\odot$)",fontsize = 20)
        ax[g,0].set_ylabel('qir',fontsize = 20)
        cb = fig.colorbar(c,ax=ax[g,h])
        cb.set_label(clab,fontsize = 20)
        
        parm_lab = str(parm) + '_hex'
        
        offsets = c.get_offsets()
        arr = c.get_array()
        df_hex['mst_hex'] = pd.Series(offsets[:,0])
        df_hex['qir_hex'] = pd.Series(offsets[:,1])
        df_hex[parm_lab] = pd.Series(arr)
        
    #plt.subplots_adjust(wspace=0, hspace=0)
    print(df_line)
    fig.set_size_inches(12, 12)
    plt.tight_layout()   
    plt.savefig('plots/qir_m_ir_rad.pdf')
    plt.show()

    
    
def extinct_qir_m(df,h):
    import matplotlib.colors as colors    
    ###defining colourmaps###
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","lightgrey","blue"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow","lightgrey","magenta"])
    cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgreen","lightgrey","indigo"])
    cmap4 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange","lightgrey","darkblue"])
    cmap5 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["maroon","lightgrey","olive"])
    cmap6 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","lightgrey","cyan"])
    
    fig, ax = plt.subplots(1,1)
    df2 = df[['qir_bress','mstars_tot','teff','rstar_disk','gas_surf_dens','sfr','mgas']] [((df['z'] == 0) & (df['mstars_tot'] > 1e8)& (df['mstars_tot'] < 1e9)& (df['qir_bress'] < 3.3)& (df['qir_bress'] > 0)) ]   
    qir = df2['qir_bress']
    mst = np.log10(df2['mstars_tot'])
    c = ax.hexbin(mst,qir,mincnt = 10, gridsize = 30,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=1e0, vmax=1e4, base=10),cmap = cmocean.cm.dense)
    
    mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,False,False)
    
    
    ax.plot(mid_m,med_qir,'black')
    ax.plot(mid_m,low_qir,'black',linestyle='dashed')   
    ax.plot(mid_m,high_qir,'black',linestyle='dashed')
    ax.set_ylabel('qir',fontsize = 20)
    cb = fig.colorbar(c)
    cb.set_label('Number Count',fontsize = 20)    
    ax.set_xlabel("$M_{*}/[M_\odot]$",fontsize = 20)
    ax.set_ylabel("qir",fontsize = 20)
    leg = ax.legend(frameon=False,loc='lower right')
    leg.get_frame().set_linewidth(0.0)
    fig.set_size_inches(8, 8)
    plt.show()
    plt.savefig("plots/qir_numbercount.pdf")
    
    plt.clf()
    
    fig, ax = plt.subplots(3,2)
    
    q = 0
    ###Reads in the data from the dataframe
    df2 = df[['qir_bress','mstars_tot','teff','rstar_disk','gas_surf_dens','sfr','mgas']] [((df['z'] == 0) & (df['mstars_tot'] > 1e8)& (df['mstars_tot'] < 1e9)& (df['qir_bress'] < 3.3)& (df['qir_bress'] > 0)) ]
    df_line = pd.DataFrame()
    qir = df2['qir_bress']
    mst = np.log10(df2['mstars_tot'])
    tef = df2['teff']
    hmr = np.log10(df2['rstar_disk']/h)
    gas = np.log10(df2['mgas']) - mst
    gsd = np.log10(df2['gas_surf_dens'])
    sfr = np.log10(df2['sfr']) - mst
    sfd = np.log10((df2['sfr']/(2*np.pi*(df2['rstar_disk']/h)**2)))
    
    ###Creates the median lines
    mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,False,False)
    
    df_line['mid_mst'] = mid_m
    df_line['med_qir'] = med_qir
    df_line['low_qir'] = low_qir
    df_line['upp_qir'] = high_qir
    
    parm_lst = [tef,hmr,sfr,gas,gsd,sfd]
    clab_lst = ['Log$_{10}(T_{eff}/[K])$','Log$_{10}(r_{0.5mass}/[kpc])$','Log$_{10}$(sSFR/[$yr^{-1}$])','Log$_{10}(M_{gas}/M_{*})$','Log$_{10}(\Sigma_{gas}/[M_\odot/M^{2}$])','Log$_{10}(\Sigma_{SFR}/[M_{\odot}yr^{-1}kpc^{-2}]])$']
    vmin_lst = [1,30,-3.5,12,-5,-8]
    vmax_lst = [4,50,-2.5,15,1,3]
    cmap_lst = [cmocean.cm.thermal,cmocean.cm.speed,cmocean.cm.solar,cmocean.cm.deep,cmocean.cm.matter,cmocean.cm.algae]

    df_hex = pd.DataFrame()

    for q in range(6):
        h = q//3
        g = q%3
        print("This is g,h")
        print(g,h)
        parm = parm_lst[q]
        clab = clab_lst[q]
        cmap = cmap_lst[q]
        vmin = vmin_lst[q]
        vmax = vmax_lst[q]
        c = ax[g,h].hexbin(mst,qir,C = parm,cmap = cmap,mincnt = 10, gridsize = 30)
        ax[g,h].plot(mid_m,med_qir,'black')
        ax[g,h].plot(mid_m,low_qir,'black',linestyle='dashed')   
        ax[g,h].plot(mid_m,high_qir,'black',linestyle='dashed')
        ax[1,h].set_xlabel("$Log_{10}$(M_{*}/$M_\odot$)",fontsize = 20)
        ax[g,0].set_ylabel('qir',fontsize = 20)
        cb = fig.colorbar(c,ax=ax[g,h])
        cb.set_label(clab,fontsize = 20)
        
        parm_lab = str(parm) + '_hex'
        
        offsets = c.get_offsets()
        arr = c.get_array()
        df_hex['mst_hex'] = pd.Series(offsets[:,0])
        df_hex['qir_hex'] = pd.Series(offsets[:,1])
        df_hex[parm_lab] = pd.Series(arr)

        
    #plt.subplots_adjust(wspace=0, hspace=0)
    print(df_line)
    fig.set_size_inches(12, 12)
    plt.tight_layout()   
    plt.savefig('plots/extinct_qir_m.pdf')
    plt.show()
    df_line.to_csv('qir_mst_line.csv')        
    df_hex.to_csv('qir_mst_hex.csv')   
    
    
def extinct_qir_m_z(df,h):
    import matplotlib.colors as colors    
    ###defining colourmaps###
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","lightgrey","blue"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow","lightgrey","magenta"])
    cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgreen","lightgrey","indigo"])
    cmap4 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange","lightgrey","darkblue"])
    cmap5 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["maroon","lightgrey","olive"])
    cmap6 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","lightgrey","cyan"])
    
    fig, ax = plt.subplots(1,1)
    
    for z in zlist:
        df2 = df[['qir_bress','mstars_tot','teff','rstar_disk','gas_surf_dens','sfr','mgas']] [((df['z'] == z) & (df['mstars_tot'] > 1e8)& (df['mstars_tot'] < 1e12)& (df['qir_bress'] < 3.3)& (df['qir_bress'] > 0)) ]   
        qir = df2['qir_bress']
        mst = np.log10(df2['mstars_tot'])
        c = ax.hexbin(mst,qir,mincnt = 100, gridsize = 30,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=1e0, vmax=1e4, base=10),cmap = cmocean.cm.dense)

        mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,False,False)


        ax.plot(mid_m,med_qir,'black')
        ax.plot(mid_m,low_qir,'black',linestyle='dashed')   
        ax.plot(mid_m,high_qir,'black',linestyle='dashed')
        ax.set_ylabel('qir',fontsize = 20)
        cb = fig.colorbar(c)
        cb.set_label('Number Count',fontsize = 20)    
        ax.set_xlabel("$M_{*}/[M_\odot]$",fontsize = 20)
        ax.set_ylabel("qir",fontsize = 20)
        leg = ax.legend(frameon=False,loc='lower right')
        leg.get_frame().set_linewidth(0.0)
        fig.set_size_inches(8, 8)
        plt.show()
        save_name = "plots/qir_numbercount_z" + str(round(z)) + ".pdf"
        plt.savefig(save_name)

    plt.clf()
    
    for z in zlist:
        fig, ax = plt.subplots(3,2)

        q = 0


        ###Reads in the data from the dataframe
        df2 = df[['qir_bress','mstars_tot','teff','rstar_disk','gas_surf_dens','sfr','mgas']] [((df['z'] == 0) & (df['mstars_tot'] > 1e8)& (df['mstars_tot'] < 1e12)& (df['qir_bress'] < 3.3)& (df['qir_bress'] > 0)) ]
        df_line = pd.DataFrame()
        qir = df2['qir_bress']
        mst = np.log10(df2['mstars_tot'])
        tef = df2['teff']
        hmr = np.log10(df2['rstar_disk']/h)
        gas = np.log10(df2['mgas']) - mst
        gsd = np.log10(df2['gas_surf_dens'])
        sfr = np.log10(df2['sfr']) - mst
        sfd = np.log10((df2['sfr']/(2*np.pi*(df2['rstar_disk']/h)**2)))

        ###Creates the median lines
        mid_m,med_qir,low_qir,high_qir = median_line(mst,qir,False,False)

        df_line['mid_mst'] = mid_m
        df_line['med_qir'] = med_qir
        df_line['low_qir'] = low_qir
        df_line['upp_qir'] = high_qir

        parm_lst = [tef,hmr,sfr,gas,gsd,sfd]
        clab_lst = ['Log$_{10}(T_{eff}/[K])$','Log$_{10}(r_{0.5mass}/[kpc])$','Log$_{10}$(sSFR/[$yr^{-1}$])','Log$_{10}(M_{gas}/M_{*})$','Log$_{10}(\Sigma_{gas}/[M_\odot/M^{2}$])','Log$_{10}(\Sigma_{SFR}/[M_{\odot}yr^{-1}kpc^{-2}]])$']
        vmin_lst = [1,30,-3.5,12,-5,-8]
        vmax_lst = [4,50,-2.5,15,1,3]
        cmap_lst = [cmocean.cm.thermal,cmocean.cm.speed,cmocean.cm.solar,cmocean.cm.deep,cmocean.cm.matter,cmocean.cm.algae]

        df_hex = pd.DataFrame()

        for q in range(6):
            h = q//3
            g = q%3
            print("This is g,h")
            print(g,h)
            parm = parm_lst[q]
            clab = clab_lst[q]
            cmap = cmap_lst[q]
            vmin = vmin_lst[q]
            vmax = vmax_lst[q]
            c = ax[g,h].hexbin(mst,qir,C = parm,cmap = cmap,mincnt = 100, gridsize = 30)
            ax[g,h].plot(mid_m,med_qir,'black')
            ax[g,h].plot(mid_m,low_qir,'black',linestyle='dashed')   
            ax[g,h].plot(mid_m,high_qir,'black',linestyle='dashed')
            ax[1,h].set_xlabel("$Log_{10}$(M_{*}/$M_\odot$)",fontsize = 20)
            ax[g,0].set_ylabel('qir',fontsize = 20)
            cb = fig.colorbar(c,ax=ax[g,h])
            cb.set_label(clab,fontsize = 20)

            parm_lab = str(parm) + '_hex'

            offsets = c.get_offsets()
            arr = c.get_array()
            df_hex['mst_hex'] = pd.Series(offsets[:,0])
            df_hex['qir_hex'] = pd.Series(offsets[:,1])
            df_hex[parm_lab] = pd.Series(arr)


        #plt.subplots_adjust(wspace=0, hspace=0)
        print(df_line)
        fig.set_size_inches(12, 12)
        plt.tight_layout()   
        save_name = 'plots/extinct_qir_m_z_' + str(z) + ".pdf"
        plt.savefig(save_name)
        plt.show()
     

    
    
def qir_m_z(df):
    
    fig, ax = plt.subplots(2, 3)
    #zlst = [0, 0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501]
    zlst = [0, 0.194738848008908]
    q = 0
    for z in zlst:
        g = q//3
        h = q%3
        qir = df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['sfg/q'] == 'sf')&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3.2 )&(df['qir_bress'] > -2)),'qir_bress']
        mstar = np.log10(df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['sfg/q'] == 'sf')&(df['mstars_tot'] < 1e12)&(df['qir_bress'] <3.2 )&(df['qir_bress'] > -2)),'mstars_tot'])
        z_label = 'z = ' + str(round(z,2))
        n_gal = 'n = ' + str(len(qir))
        mid_mstar_lst,med_qir_lst,low_1_qir_lst,high_1_qir_lst,low_2_qir_lst,high_2_qir_lst,low_3_qir_lst,high_3_qir_lst = med_sig_lines(mstar,qir,False,False)
        c = ax[g,h].hexbin(mstar,qir,cmap = 'rainbow',mincnt = 10, gridsize = 30,vmin = 10, vmax = 1000)
        ax[g,h].plot(mid_mstar_lst,med_qir_lst,'black')
        ax[g,h].plot(mid_mstar_lst,low_1_qir_lst,'black',linestyle = 'dashed')
        ax[g,h].plot(mid_mstar_lst,high_1_qir_lst,'black',linestyle = 'dashed')
        ax[g,h].plot(mid_mstar_lst,low_2_qir_lst,'black',linestyle = 'dotted')
        ax[g,h].plot(mid_mstar_lst,high_2_qir_lst,'black',linestyle = 'dotted')
        ax[g,h].plot(mid_mstar_lst,low_3_qir_lst,'black',linestyle = 'dashdot')
        ax[g,h].plot(mid_mstar_lst,high_3_qir_lst,'black',linestyle = 'dashdot')

        ax[g, 0].set_ylabel("qir")
        ax[1, h].set_xlabel("log$_{10}$ M$_{\odot}$")
        ax[g, h].text(10,0,z_label)
        ax[g,h].text(10,-1,n_gal)
        ax[1, h].tick_params(bottom = True, top = True)
        ax[0, h].tick_params(labelbottom=False) 
        ax[g, 1].tick_params(labelleft=False) 
        ax[g, 2].tick_params(labelleft=False)
        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(-2,3.2)
        q += 1
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(c, cax=cbar_ax)
    cb.set_label('Number Count')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(12, 8)
    plt.savefig("plots/qir_m_z.pdf")
    plt.show()
    
    
    fig, ax = plt.subplots(2, 3)
   # zlst = [0, 0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501]
   # zlst = [0, 0.194738848008908]
    q = 0
    for z in zlst:
        g = q//3
        h = q%3
        fir = np.log10(df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['sfg/q'] == 'sf')&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3.2 )&(df['qir_bress'] > -2)),'lir_w'])
        rad = np.log10(df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['sfg/q'] == 'sf')&(df['mstars_tot'] < 1e12)&(df['qir_bress'] <3.2 )&(df['qir_bress'] > -2)),'bress_rad_lum'])
        z_label = 'z = ' + str(round(z,2))
        mid_mstar_lst,med_qir_lst,low_1_qir_lst,high_1_qir_lst,low_2_qir_lst,high_2_qir_lst,low_3_qir_lst,high_3_qir_lst = med_sig_lines(rad,fir,False,False)
        c = ax[g,h].hexbin(rad,fir,cmap = 'rainbow',mincnt = 10, gridsize = 30,vmin = 10, vmax = 10000)
        ax[g,h].plot(mid_mstar_lst,med_qir_lst,'black')
        ax[g,h].plot(mid_mstar_lst,low_1_qir_lst,'black',linestyle = 'dashed')
        ax[g,h].plot(mid_mstar_lst,high_1_qir_lst,'black',linestyle = 'dashed')
        ax[g,h].plot(mid_mstar_lst,low_2_qir_lst,'black',linestyle = 'dotted')
        ax[g,h].plot(mid_mstar_lst,high_2_qir_lst,'black',linestyle = 'dotted')
        ax[g,h].plot(mid_mstar_lst,low_3_qir_lst,'black',linestyle = 'dashdot')
        ax[g,h].plot(mid_mstar_lst,high_3_qir_lst,'black',linestyle = 'dashdot')

        ax[g, 0].set_ylabel("log$_{10}$ Lir/W")
        ax[1, h].set_xlabel("log$_{10}$ Lrad/W")
        ax[g, h].text(19.5,37,z_label)
        ax[1, h].tick_params(bottom = True, top = True)
        ax[0, h].tick_params(labelbottom=False) 
        ax[g, 1].tick_params(labelleft=False) 
        ax[g, 2].tick_params(labelleft=False)
        ax[g,h].set_xlim(19,22)
        ax[g,h].set_ylim(30,38)
        q += 1
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(c, cax=cbar_ax)
    cb.set_label('Number Count')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(12, 8)
    plt.savefig('plots/lir_lrad_z.pdf')
    plt.show()


def bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist):
    Lsunwatts = 3.846e26
    h = 6.6261e-27 #cm2 g s-1
    d10pc = 3.086e+19
    dfac = 4 * PI * d10pc**2
    qir_bress = []
    bress_rad_lum = []
    bress_lir_total = []
    qir_lst = []
    freefree_lst = []
    sync_lst = []
    ion_phot_lst = []
    
    IncludeSelfAbsorption = False
    
    for z in range(len(zlist)):
        sfr_disk = np.array(hdf5_lir_lst[z][4])
        sfr_disk_2 = []
        sfr_bulge = np.array(hdf5_lir_lst[z][5])
        sfr_bulge_2 = []
        sfr_tot = []
        sfr_tot = sfr_bulge + sfr_disk
        mdisk = np.array(hdf5_lir_lst[z][2])
        mbulge = np.array(hdf5_lir_lst[z][3])
        r_gas_disk_array = np.array(hdf5_lir_lst[z][11]) #half radius of the gas disk [cMpc/h]
        
        print("THis is mdisk")
        print(mdisk)

        ind0 = np.where(np.asarray(mdisk) + np.asarray(mbulge) > 0)
        ms = (mdisk[ind0] + mbulge[ind0])/h0 #in Msun
        sfr_tot = (sfr_tot[ind0])/h0/1e9
        #ionising mag
        lir_total = np.array(lir_total_W_lst[z], dtype=np.float64)
       # ind109 = np.where(lir_total > 10**7 * Lsun)
      #  print("This is the length of sfr before removing 10^9")
      #  print(len(sfr_tot))
      #  lir_total = lir_total[ind109]
      # ms = ms[ind109]
       # sfr_tot= sfr_tot[ind109]
        ind108 = np.where(ms > 0)
        lir_total = lir_total[ind108]
        ms = ms[ind108]
        sfr_tot= sfr_tot[ind108]
        r_gas_disk_array = r_gas_disk_array[ind0]
        r_gas_disk_array = r_gas_disk_array[ind108]
        
        
        
        print("This is length of sfr")
        print(len(sfr_tot))

        n = 0
        
        def tau_factor(m, wave, rgas_disk,nu):
            wave_m = wave * 1e-10 #wavelength in m
            wave_cm = wave_m  * 1e2 #wavelength in cm
            
            T = 1e4 #Temperature in K
            
            freq = (c_speed/ wave) / 1e9 #Frequency in GHz
            
            ne = 10.0**((m + 48.6) / (-2.5)) * dfac * freq * wave_cm #number of electrons
            
            l = (rgas_disk/h/c_speed)*1e6 #distance in pc
            
            tau = 8.2 * 10**(-2) * T**(-1.35) * nu**(-2.1) * (ne**2 * l)
            
            tau = T**(-1.35)*(nu/1.4)**(-2.1) * EM/6e6
            
            return tau

        def ionising_photons(m, wave):
            #m is a vector of AB absolute magnitudes in a band with central wavelength wave
            #wavelength input has to be in angstzrom

            wave_m = wave * 1e-10 #wavelength in m
            wave_cm = wave_m  * 1e2 #wavelength in cm
            freq = c_speed / wave_m #Hz
            hc =  h * (c_speed * 1e2) #h*c in cgs

            lum = 10.0**((m + 48.6) / (-2.5)) * dfac * freq * wave_cm #we want to convert from Luminosity to int(lambda*Lum_lambda*dlambda)
            Q = lum / hc #rate of ionising photons in s^-1.

            return Q

        def freefree_lum(Q, nu):

        #Q is the rate of ionising photos in s^-1
        #nu is the frequency in GHz
        #output in erg/s/Hz

            T = 1e4 #temperature in K
            lum = Q/6.3e25 * (T/1e4)**0.45 * (nu)**(-0.1)

            return lum

        def synchrotron_lum(SFR, nu):

            nu_mw = 0.011148*1.0 #Assumes MW SFR of 1.0
            LNT = 6.1*10**(-2)*(1.49/0.408)**(-0.8) ##2.16 x10^(-2) W/Hz/10^(21)
            ENT = LNT/nu_mw ##1.31 x 10^(23) W/Hz/10^(21)
            ESNR = ENT * 0.06 ##0.0784 x 10^(23)W/Hz/10^(21)
            EEI = 0.94*ENT ##1.23 x 10^(23) W/Hz/10^(21)
            alpha =  0.815661

            comp1 = ESNR * (nu / 1.49)**(-0.5) + (EEI * (nu / 1.49)**(-alpha))
            nuSNCC = SFR * 0.011148
            lum = comp1 * 1e30
            lum = lum * nuSNCC
            
            if(IncludeSelfAbsorption == True):
                for i in (range(100)):
                    print("SELF ABSORPTION IS INCLUDED")

                
                comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(alpha) * np.e**(-tau)
            
            
            return lum


        total_mags_nod = np.array(seds_bands_lst[z][2], dtype=np.float64)
        ion_mag = total_mags_nod[1,:]
     #   ion_mag = ion_mag[ind109]
        ion_mag = ion_mag[ind108]
        q_ionis = ionising_photons(ion_mag, 912.0) #in s^-1
 
        selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15)
        lum_radio = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        freefree = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        sync = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        for i, nu in enumerate(selection_freq):
            lum_radio[i,:] = freefree_lum(q_ionis[:], nu) + synchrotron_lum(sfr_tot[:], nu)

            freefree[i,:] = freefree_lum(q_ionis[:], nu)/1e7
            
            sync[i,:] = synchrotron_lum(sfr_tot[:], nu)/1e7
            

        
        
        qIR_bressan = np.log10(lir_total/3.75e12) - np.log10(lum_radio[3,:]/1e7)
        

        inf_lst = np.where(qIR_bressan == np.inf)
        notinf_lst = np.where(qIR_bressan != np.inf)
        qir_lst.append(qIR_bressan)
        qir_bress.append(np.median(qIR_bressan[notinf_lst]))
        
        bress_rad_lum.append(lum_radio[3][notinf_lst]/1e7)

        bress_lir_total.append(np.median(lir_total[notinf_lst]))
        
        

        
        freefree_lst.append(freefree[3])
        sync_lst.append(sync[3])
           
            
        ion_phot_lst.append(q_ionis)
    
    return qir_lst, qir_bress, bress_rad_lum, bress_lir_total, freefree_lst, sync_lst, ion_phot_lst
    
def median_line(xlist,ylist,xlog,ylog):



    med_y_lst = []
    low_y_lst = []
    high_y_lst = []
    mid_x_lst = []



    if xlog == True:
        xlist = np.log10(np.array(xlist))
    else:
        xlist = np.array(xlist)

    if ylog == True:
        ylist = np.log10(np.array(ylist))
    else:
        ylist = np.array(ylist)

    xbin_lst = list(np.arange(min(xlist),max(xlist),(max(xlist)-min(xlist))/30))

    

    for i in range(len(xbin_lst)-1):
        m = xbin_lst[i]
        n = xbin_lst[i+1]

        x_bins = np.where((xlist > m)&(xlist < n))


        y_bins = ylist[x_bins]

        if len(y_bins) > 9:

            med_y = np.median(y_bins)
            med_y_lst.append(med_y)
            mid_x = (n/2+m/2)
            mid_x_lst.append(mid_x)
            low_y_lst.append(np.percentile(y_bins,16))
            high_y_lst.append(np.percentile(y_bins,84))
        
        else:
            mid_x_lst.append(np.nan)
            med_y_lst.append(np.nan)
            low_y_lst.append(np.nan)
            high_y_lst.append(np.nan)
            continue
    if ylog == True:
        med_y_lst = 10**np.array(med_y_lst)
        low_y_lst = 10**np.array(low_y_lst)
        high_y_lst = 10**np.array(high_y_lst)
    if xlog == True:
        mid_x_lst = 10**np.array(mid_x_lst)

    return np.array(mid_x_lst),np.array(med_y_lst),np.array(low_y_lst),np.array(high_y_lst)

def any_sigma_lines(xlist,ylist,xlog,ylog,sig_up,sig_down):



    med_y_lst = []
    low_y_lst = []
    high_y_lst = []
    mid_x_lst = []



    if xlog == True:
        xlist = np.log10(np.array(xlist))
    else:
        xlist = np.array(xlist)

    if ylog == True:
        ylist = np.log10(np.array(ylist))
    else:
        ylist = np.array(ylist)

    xbin_lst = list(np.arange(min(xlist),max(xlist),(max(xlist)-min(xlist))/30))


    for i in range(len(xbin_lst)-1):
        m = xbin_lst[i]
        n = xbin_lst[i+1]

        x_bins = np.where(((xlist > m)&(xlist < n)))


        y_bins = ylist[x_bins]


        med_y = np.median(y_bins)
        med_y_lst.append(med_y)
        mid_x = (n/2+m/2)
        mid_x_lst.append(mid_x)
        try:
            low_y_lst.append(np.percentile(y_bins,sig_down))
            high_y_lst.append(np.percentile(y_bins,sig_up))
        except:
            low_y_lst.append(np.nan)
            high_y_lst.append(np.nan)
            continue
    if ylog == True:
        med_y_lst = 10**np.array(med_y_lst)
        low_y_lst = 10**np.array(low_y_lst)
        high_y_lst = 10**np.array(high_y_lst)
    if xlog == True:
        mid_x_lst = 10**np.array(mid_x_lst)

    return mid_x_lst,med_y_lst,low_y_lst,high_y_lst

def med_sig_lines(xlist,ylist,xlog,ylog):



    med_y_lst = []
    low_1_y_lst = []
    high_1_y_lst = []
    low_2_y_lst = []
    high_2_y_lst = []
    low_3_y_lst = []
    high_3_y_lst = []
    mid_x_lst = []



    if xlog == True:
        xlist = np.log10(np.array(xlist))
    else:
        xlist = np.array(xlist)

    if ylog == True:
        ylist = np.log10(np.array(ylist))
    else:
        ylist = np.array(ylist)

    xbin_lst = list(np.arange(min(xlist),max(xlist),(max(xlist)-min(xlist))/30))


    for i in range(len(xbin_lst)-1):
        m = xbin_lst[i]
        n = xbin_lst[i+1]

        x_bins = np.where(((xlist > m)&(xlist < n)))


        y_bins = ylist[x_bins]


        med_y = np.median(y_bins)
        med_y_lst.append(med_y)
        mid_x = (n/2+m/2)
        mid_x_lst.append(mid_x)
        try:
            low_1_y_lst.append(np.percentile(y_bins,16))
            high_1_y_lst.append(np.percentile(y_bins,84))
            low_2_y_lst.append(np.percentile(y_bins,2.5))
            high_2_y_lst.append(np.percentile(y_bins,97.5))
            low_3_y_lst.append(np.percentile(y_bins,0.5))
            high_3_y_lst.append(np.percentile(y_bins,99.5))
        except:
            low_1_y_lst.append(np.nan)
            high_1_y_lst.append(np.nan)
            low_2_y_lst.append(np.nan)
            high_2_y_lst.append(np.nan)
            low_3_y_lst.append(np.nan)
            high_3_y_lst.append(np.nan)
            continue
    if ylog == True:
        med_y_lst = 10**np.array(med_y_lst)
        low_y_lst = 10**np.array(low_y_lst)
        high_y_lst = 10**np.array(high_y_lst)
    if xlog == True:
        mid_x_lst = 10**np.array(mid_x_lst)

    return mid_x_lst,med_y_lst,low_1_y_lst,high_1_y_lst,low_2_y_lst,high_2_y_lst,low_3_y_lst,high_3_y_lst

def radio_luminosity_agn(mbh, macc):

    #input mbh has to be in Msun
    #input macc has to be in Msun/yr
    #output luminosity in erg/s
    c_speed = 299792458.0 #in m/s
    c_speed_cm = c_speed * 1e2 #in cm/s
    spin = 0.1
    eta_edd = 4
    Ljet_ADAF = np.zeros ( shape = len(mbh))
    Ljet_td = np.zeros ( shape = len(mbh))
    mdot_norm = np.zeros ( shape = len(mbh))
    Ledd = np.zeros ( shape = len(mbh))
    print(mbh)
    ind = np.where(mbh > 0)
    df_temp = pd.DataFrame(list(zip(mbh,macc)), columns = ['mbh','macc'])
    
    df_temp['ledd'] = 0
    
    df_temp.loc[((df_temp['mbh'] > 0)), 'ledd'] = 1.28e46 * (df_temp['mbh']/1e8) #in erg/s
    
    print(df_temp)
    
    df_temp['mdot_edd'] = 0 
    print(df_temp)    
    df_temp.loc[((df_temp['mbh'] > 0)& (df_temp['macc'] > 0)), 'mdot_edd'] = df_temp['ledd']/ (0.1 * c_speed_cm**2) * 1.586606334841629e-26 #in Msun/yr
    df_temp['mdot_norm'] = df_temp['macc']/df_temp['mdot_edd']
    
    df_temp['Ljet_ADAF'] = 2e45 * (df_temp['mbh']/1e9) * (df_temp['mdot_norm']/0.01) * spin**2 #in erg/s
    df_temp['Ljet_td'] = 2.5e43 * (df_temp['mbh']/1e9)**1.1 * (df_temp['mdot_norm']/0.01)**1.2 * spin**2 #in erg/s
    
    df_temp.loc[((df_temp['mdot_norm'] > eta_edd)), 'mdot_norm'] = eta_edd
        
    
    print(df_temp)
    
    
    Ljet_ADAF = df_temp['Ljet_ADAF']
    Ljet_td = df_temp['Ljet_td']
    mdot_norm = df_temp['mdot_norm']
        
    return Ljet_ADAF, Ljet_td, mdot_norm

def radio_luminosity_per_freq (Ljet_ADAF, Ljet_td, mdot_norm, mbh, nu):

    #here nu has to be rest-frame in GHz
    lum_1p4GHz_adaf = A_adaf * (mbh / 1e9 * mdot_norm / 0.01)**0.42 * Ljet_ADAF
    lum_1p4GHz_td = A_td * (mbh / 1e9)**0.32 * (mdot_norm / 0.01)**(-1.2) * Ljet_td

    freq_hz = nu * 1e9
    lum_nu = (lum_1p4GHz_adaf + lum_1p4GHz_td) * (nu / 1.4)**(-0.7) / freq_hz #in erg/s/Hz
    print("This is lum nu")
    print(lum_nu)
   # return lum_nu, (lum_1p4GHz_adaf + lum_1p4GHz_td)
    return lum_nu


def main(model_dir, outdir, redshift_table, subvols, obsdir):


    plt = common.load_matplotlib()

    file_name = "eagle-rr14-radio-only"
    file_hdf5_sed = "Shark-SED-" + file_name + ".hdf5"

    fields_sed = {'SED/lir_dust': ('disk','bulge_t','total'),}
    fields_sed_bc = {'SED/lir_dust_contribution_bc':('disk', 'total')}
    fields_seds_nodust = {'SED/ab_nodust':('disk','bulge_t','total'),}
    fields_seds_dust = {'SED/ab_dust':('disk','bulge_t','total'),}

  #  fields = {'galaxies': ('mstars_disk', 'mstars_bulge','sfr_disk', 'sfr_burst','type','mgas_metals_bulge', 'mgas_metals_disk')}

    fields = {'galaxies': ('mstars_disk','mstars_bulge','sfr_disk','sfr_burst','type','mgas_metals_bulge','mgas_metals_disk','mgas_bulge','mgas_disk','rgas_disk','rstar_disk','matom_disk','mmol_disk','id_galaxy','m_bh','bh_accretion_rate_hh', 'bh_accretion_rate_sb')}
    
    
    #Bands information:
    #(0): "hst/ACS_update_sep07/wfc_f775w_t81:x", "hst/wfc3/IR/f160w",
    #(2): "F200W_JWST", "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS",
    #(8): "i_SDSS", "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA",
    #(14): "W1_WISE", "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer",
    #(19): "I4_Spitzer", "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
    #(24): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
    #(28): "S500_Herschel", "S850_JCMT", "FUV_Nathan", "Band9_ALMA",
    #(32): "Band8_ALMA", "Band7_ALMA", "Band6_ALMA", "Band4_ALMA",
    #(36): "Band3_ALMA", "BandL_VLA", "BandS_VLA"
    
    
    
    
    LFs_dust     = np.zeros(shape = (len(zlist), len(mbins)))
    print("This is zlist")
    print(zlist)
    
    seds_lir_lst = []
    
    hdf5_lir_lst = []
    
    seds_lir_bc_lst = []
    
    seds_bands_lst = []
    
    seds_nodust_lst = []
    
    lir_total_W_lst = []
    
    Teff_lst = []
    for index, snapshot in enumerate(redshift_table[zlist]):
        print("Will read snapshot %s" % (str(snapshot)))

        hdf5_data = common.read_data(model_dir, snapshot, fields, subvols)

        seds_lir = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed, subvols, file_hdf5_sed)
        
        seds_lir_bc = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed_bc, subvols, file_hdf5_sed)


        seds_nodust = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_nodust, subvols, file_hdf5_sed)
        
        seds_bands = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_dust, subvols, file_hdf5_sed)

        (volh, h0, band_14, band_30, lir_total_W,Teff) = prepare_data(hdf5_data, seds_lir, seds_bands, seds_lir_bc, index, LFs_dust, obsdir)
        
        seds_nodust_lst.append(seds_nodust)
        
        seds_lir_lst.append(seds_lir)
        
        seds_lir_bc_lst.append(seds_lir_bc)
        
        seds_bands_lst.append(seds_bands)

        hdf5_lir_lst.append(hdf5_data)
        
        lir_total_W_lst.append(lir_total_W)
        
        Teff_lst.append(Teff)
        
        
    

    start_time = time.time()
    df = dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst,fields)
    print('df2 takes:')
    print("--- %s seconds ---" % (time.time() - start_time))



  #  rad_size(df)
 #   density(df)
  #  qir_z_plt(df)
   # schr_vs_med(df)
    #fir_lir_mass(df)
    
    #qir_with_m_plt(df)
    
    #q_m_m_z(df)
    
    #qir_vs_qir(df)
    
    #sfr_rad_lum(df)
    #rad_lum_func_plt_z0(df,h0,volh)
  #  rad_lum_func_plt_2(df,h0,volh)

   # hist(df)
   # qir_with_m_plt(df)
    
    #lum_m(df,sf_lst)
  #  qir_metals_mass_bins(df)
  #  metal_hist(df)
    
  #  gas_metal_vs_stellar_mass(df)
    
    #derived_SFR(df)
    
   # qir_v_dust_ff_sync(df)       
    #lo_faro_plots(df)
   # qir_hist(df)
   # GAMA_plots(df)
    #GAMA_plots_old(df)
  #  GAMA_flux_limit(df)
   # met_dist_cent(df)
   # sfr_function(df,h0,volh)
   # qir_plots(df)
   # all_galaxies_gas_disks(df)
   # SFR_z(df)
   # ionising_phot_sfr(df)
  #  rad_lum_func_plt_z0(df,h0,volh)
  #  all_galaxies_gas_disks(df)
   # rad_lum_func_z(df,h0,volh)
#    sf_galaxies_gas_disks(df)
#    delv_plt(df)
 #   qir_m_z(df)
    #qir_mstar_ulirgs(df)
   # sfr_m_hex(df)
   # delv_plt(df)
    #delta_qir(df)
   # extinct_qir_m(df,h0)
   # delv_qir_plts(df)
    rad_lum_func_agn(df,h0,volh)
   # extinct_qir_m_z(df,h0)
  #  qir_mstar(df)
  #  ir_rad_qir_m(df,h0)
    mstars_disk = hdf5_data[2]
    mstars_bulge = hdf5_data[3]
    mstars_tot = []
  #  for i, j in zip(mstars_disk, mstars_bulge):
  #      k = i + j
  #      if k > 0: 
  #          mstars_tot.append(k)
#    radio_convert(h0, zlist,seds_lir_lst,hdf5_lir_lst,seds_lir_bc_lst,seds_bands_lst, lir_total_W_lst)
    def take_log(x,v,h):
        x = x / (v / h**3.0)
        ind = np.where(x > 0)
        x[ind] = np.log10(x[ind])
        return x

    LFs_dust = take_log(LFs_dust, volh, h0)
    plot_lir_lf(plt, outdir, obsdir, LFs_dust, file_name)
    

    
if __name__ == '__main__':
    main(*common.parse_args())
   
    #compile at home: python3 lir_highz_study-dale_z.py -m Shark -s medi-SURS -S ~/SHARK_Out/ -z ~/SHARK_Out/medi-SURS/redshift_list  -v 0 -o ~/SHARK_Out/output/
    #compile on magnus: 