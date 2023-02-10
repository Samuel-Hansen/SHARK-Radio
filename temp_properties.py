
#
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
import common
import utilities_statistics as us
import pandas as pd
import matplotlib as mpl
from collections import OrderedDict


import statistics as stat
zlist=[0]

#zlist = [0, 0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086,8.94312532315157, 9.95650268434316]
#zlist = [0, 1.0, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086, 8.94312532315157, 9.95650268434316]
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

def prepare_data(hdf5_data, seds, seds_bands, seds_bc, index, LFs_dust, obsdir):

    (h0, volh, bh_accretion_rate_hh,bh_accretion_rate_sb,cnfw_subhalo,cooling_rate,descendant_id_galaxy,lambda_subhalo,m_bh,matom_bulge,matom_disk,mean_stellar_age,mgas_bulge,mgas_disk,mgas_metals_bulge,mgas_metals_disk,mhot,mhot_metals,mlost,mlost_metals,mmol_bulge,mmol_disk,mreheated,mreheated_metals,mstars_bulge,mstars_bulge_diskins_assembly,mstars_bulge_mergers_assembly,mstars_burst_diskinstabilities,mstars_burst_mergers,mstars_disk,mstars_metals_bulge,mstars_metals_bulge_diskins_assembly,mstars_metals_bulge_mergers_assembly,mstars_metals_burst_diskinstabilities,mstars_metals_burst_mergers,mstars_metals_disk,mvir_hosthalo,mvir_subhalo,redshift_merger,rgas_bulge,rgas_disk,rstar_bulge,rstar_disk,sfr_burst,sfr_disk,specific_angular_momentum_bulge_gas,specific_angular_momentum_bulge_star,specific_angular_momentum_disk_gas,specific_angular_momentum_disk_gas_atom,specific_angular_momentum_disk_gas_mol,specific_angular_momentum_disk_star,typ,vmax_subhalo,vvir_hosthalo,vvir_subhalo) = hdf5_data
    bin_it = functools.partial(us.wmedians, xbins=xmf)

    lir_disk = seds[0]
    lir_bulge = seds[1]
    lir_total = seds[2] #total IR luminosity in units of Lsun

    lir_cont_bc = seds_bc[1]

    lir_total = np.array(lir_total, dtype=np.float64)
    Tbc = 50.0
    Tdiff = 25.0

    
    Teff = Tbc * lir_cont_bc[0] + Tdiff * (1 - lir_cont_bc[0]) #check if fraction

    #luminosity weight dust temperature

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

    meds_radio = bin_it(x=lir_selected, y=lradio_selected)
  
    return(volh, h0, band_14, band_30, lir_total_W,Teff)

def dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst):
    h0 = float(h0)
    
    lst = ["bh_accretion_rate_hh","bh_accretion_rate_sb","cnfw_subhalo","cooling_rate","descendant_id_galaxy","lambda_subhalo","m_bh","matom_bulge","matom_disk","mean_stellar_age","mgas_bulge","mgas_disk","mgas_metals_bulge","mgas_metals_disk","mhot","mhot_metals","mlost","mlost_metals","mmol_bulge","mmol_disk","mreheated","mreheated_metals","mstars_bulge","mstars_bulge_diskins_assembly","mstars_bulge_mergers_assembly","mstars_burst_diskinstabilities","mstars_burst_mergers","mstars_disk","mstars_metals_bulge","mstars_metals_bulge_diskins_assembly","mstars_metals_bulge_mergers_assembly","mstars_metals_burst_diskinstabilities","mstars_metals_burst_mergers","mstars_metals_disk","mvir_hosthalo","mvir_subhalo","redshift_merger","rgas_bulge","rgas_disk","rstar_bulge","rstar_disk","sfr_burst","sfr_disk","specific_angular_momentum_bulge_gas","specific_angular_momentum_bulge_star","specific_angular_momentum_disk_gas","specific_angular_momentum_disk_gas_atom","specific_angular_momentum_disk_gas_mol","specific_angular_momentum_disk_star","type","vmax_subhalo","vvir_hosthalo","vvir_subhalo"]

    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m
    lstlst = []
    q = 0
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_nodust_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)
    
    len_lst = []
    
    for z in range(len(zlist)):
        idx_disk = lst.index('mstars_disk') + 2
        idx_bulge = lst.index('mstars_bulge') + 2
        stellar_mass_lst = []
        mstars_disk_array = np.array(hdf5_lir_lst[z][idx_disk])
        mstars_bulge_array = np.array(hdf5_lir_lst[z][idx_bulge])
        stellar_mass_array = mstars_disk_array + mstars_bulge_array
        ind0 = np.where(stellar_mass_array > 0)
        stellar_mass_array = stellar_mass_array[ind0]

        len_lst.append(len(stellar_mass_array))
        
    big_lst = []
    q = 0
    for i in zlist:
        j = len_lst[q]
        for k in range(j):
            big_lst.append(i)
        q +=1
    
    df = pd.DataFrame({'z': big_lst})

    



    for i in lst:
        i_lst = []
        for z in range(len(zlist)):
            idx_disk = lst.index('mstars_disk') + 2
            idx_bulge = lst.index('mstars_bulge') + 2
            stellar_mass_lst = []
            mstars_disk_array = np.array(hdf5_lir_lst[z][idx_disk])
            mstars_bulge_array = np.array(hdf5_lir_lst[z][idx_bulge])
            stellar_mass_array = mstars_disk_array + mstars_bulge_array
            ind0 = np.where(stellar_mass_array > 0)
            idx = lst.index(i) + 2
            i_array = np.array(hdf5_lir_lst[z][idx])
            
            i_array = i_array[ind0]
            
            i_lst.append(i_array)

        flat_list = [item for sublist in i_lst for item in sublist]

        df[i] = flat_list
    df['type'] = df['type']/h0
    df["vmax_subhalo"] = df["vmax_subhalo"]
    df["vvir_hosthalo"] = df["vvir_hosthalo"]
    df["vvir_subhalo"] = df["vvir_subhalo"]
    df['lambda_subhalo'] = df['lambda_subhalo']
    df['mean_stellar_age'] = df['mean_stellar_age']
    
    df['rgas_bulge'] = df['rgas_bulge']/h0
    df['rgas_disk'] = df['rgas_disk']/h0
    df['rstar_bulge'] = df['rstar_bulge']/c_speed    
    df['rstar_disk'] = df['rstar_disk']/c_speed
    df['specific_angular_momentum_bulge_gas'] = df['specific_angular_momentum_bulge_gas']/c_speed    
    df['specific_angular_momentum_bulge_star'] = df['specific_angular_momentum_bulge_star']/c_speed    
    df['specific_angular_momentum_disk_gas'] = df['specific_angular_momentum_disk_gas']/c_speed        
    df['specific_angular_momentum_disk_gas_atom'] = df['specific_angular_momentum_disk_gas_atom']/c_speed           
    df['specific_angular_momentum_disk_gas_mol'] = df['specific_angular_momentum_disk_gas_mol']/c_speed     
    df['specific_angular_momentum_disk_star'] = df['specific_angular_momentum_disk_star']/c_speed     
    df['sfr_burst'] = df['sfr_burst']/1e9/h0
    df["sfr_disk"] = df['sfr_disk']/1e9/h0
    df['mstars_disk']/h0 
    df['mstars_bulge']/h0
    df['mstars_tot'] = df['mstars_disk'] + df['mstars_bulge']
    df['sfr'] = df['sfr_burst']+ df["sfr_disk"]
    
    flat_teff_list = [item for sublist in Teff_lst for item in sublist]
    df['teff'] = flat_teff_list
    df['qir'] = [item for sublist in qir_lst_bress for item in sublist]
    df['radio_lum'] = [item for sublist in bress_rad_lum_lst for item in sublist]
    df['freefree'] = [item for sublist in freefree_lst for item in sublist]
    df['sync'] = [item for sublist in sync_lst for item in sublist]
    df['gas_metal'] = ((df['mgas_metals_bulge'] + df['mgas_metals_disk'])/(df['mgas_bulge']+df['mgas_disk']))/0.018
    df['far_infrared_luminosity'] = 10**(df['qir']-np.log10(df['radio_lum']))*3.75*10**12
    df['rad_lum_check'] = df['radio_lum'] - (df['freefree'] + df['sync'])
    df['sfr/sync'] = df['sfr']/df['sync']
    print("This is df[sfr/sync]")
    print(df['sfr/sync'])

    
    df['sfr/q'] = 'q'
    df['sf_test'] = 1

    sf_lst = []
    for z in zlist:
        sfr_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'sfr']
        m_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'mstars_tot']
        
        sfr_z = np.log10(sfr_z)
        m_z = np.log10(m_z)
        
        a,b = np.polyfit(m_z,sfr_z,1)
        df['sf_test'] = abs(np.log10(df['sfr']) - (a * np.log10(df['mstars_tot']) + b))
        df.loc[((df['z'] == z)&(df['sf_test'] < 0.3)), 'sfr/q'] = 'sf'
        
        
        
        sf_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'sfr']
        sf_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'mstars_tot']
        q_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'sfr']
        q_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'mstars_tot'] 
        strz = str(z)
        tit = "z = " + strz[:6]

        x = np.linspace(8, 12, 10)
        y = a * x + b
        y1 = a*x + b + 0.3
        y2 = a*x + b - 0.3

        plt.plot(10**(x),10**(y),color = 'red',label = 'line fit')
        plt.fill_between(x,y1,y2,color = 'red')
        plt.scatter(sf_gal_m, sf_gal_sfr, label = 'Star forming')
        plt.scatter(q_gal_m, q_gal_sfr, label = 'Quenched')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Stellar mass")
        plt.ylabel("SFR")

        plt.legend()
        plt.show()
        sf_lst.append((a,b))
    
    
    
    
    lst2 = ["bh_accretion_rate_hh","bh_accretion_rate_sb","cnfw_subhalo","cooling_rate","descendant_id_galaxy","lambda_subhalo","m_bh","matom_bulge","matom_disk","mean_stellar_age","mgas_bulge","mgas_disk","mgas_metals_bulge","mgas_metals_disk","mhot","mhot_metals","mlost","mlost_metals","mmol_bulge","mmol_disk","mreheated","mreheated_metals","mstars_bulge","mstars_bulge_diskins_assembly","mstars_bulge_mergers_assembly","mstars_burst_diskinstabilities","mstars_burst_mergers","mstars_disk","mstars_metals_bulge","mstars_metals_bulge_diskins_assembly","mstars_metals_bulge_mergers_assembly","mstars_metals_burst_diskinstabilities","mstars_metals_burst_mergers","mstars_metals_disk","mvir_hosthalo","mvir_subhalo","redshift_merger","rgas_bulge","rgas_disk","rstar_bulge","rstar_disk","sfr_burst","sfr_disk","specific_angular_momentum_bulge_gas","specific_angular_momentum_bulge_star","specific_angular_momentum_disk_gas","specific_angular_momentum_disk_gas_atom","specific_angular_momentum_disk_gas_mol","specific_angular_momentum_disk_star","type","vmax_subhalo","vvir_hosthalo","vvir_subhalo",'mstars_tot','sfr','teff','qir','radio_lum','freefree','sync','gas_metal','far_infrared_luminosity','rad_lum_check','sfr/sync']
    
    df2 = pd.DataFrame({'parms': lst2})
    
    med_lst_all = []
    
    for i in lst2:

        med = np.median(df[i])
        med_lst_all.append(med)
        
    med_lst_cold = []
    
    for i in lst2:
        med = np.median(df.loc[(df['teff']< 35),i])
        med_lst_cold.append(med)
        
    med_lst_hot = []
    
    for i in lst2:
        med = np.median(df.loc[(df['teff'] > 45),i])
        med_lst_hot.append(med)
        
    all_qir = []
    
    for i in lst2:
        med = np.median(df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)),i])
        all_qir.append(med)
        
        
    big_qir= []
    
    for i in lst2:
        med = np.median((df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)&(df['qir']>2.5)&(df['qir']<3)),i]))
        big_qir.append(med)
    
                        
    low_qir = []
    
    for i in lst2:
        med = np.median((df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)&(df['qir']< 1)&(df['qir']> 0)),i]))
        low_qir.append(med)
        
    all_qir_sf = []
    
    for i in lst2:
        med = np.median(df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)&(df['sfr/q']=='sf')),i])
        all_qir_sf.append(med)
        
        
    big_qir_sf= []
    
    for i in lst2:
        med = np.median((df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)&(df['qir']>2.5)&(df['sfr/q']=='sf')&(df['qir']<3)),i]))
        big_qir_sf.append(med)
    
                        
    low_qir_sf = []
    
    for i in lst2:
        med = np.median((df.loc[((df['mstars_tot']<3e8)&(df['mstars_tot']>1e8)&(df['qir']< 1)&(df['sfr/q']=='sf')&(df['qir']> 0)),i]))
        low_qir_sf.append(med)       
        
    

    df2['qir_all'] = all_qir
    df2['qir>2.5_all'] = big_qir
    df2['qir<1_all'] = low_qir
    df2['%_diff_(qir>2.5_all-qir<1_all)/qir>2.5_all'] = (df2['qir>2.5_all']-df2['qir<1_all'])/df2['qir>2.5_all']
    df2['qir_all_sf'] = all_qir_sf
    df2['qir>2.5_sf'] = big_qir_sf
    df2['qir<1_sf'] = low_qir_sf
    df2['%_diff_(qir>2.5_sf-qir<1_sf)/qir>2.5_sf'] = (df2['qir>2.5_sf']-df2['qir<1_sf'])/df2['qir>2.5_sf']
    
    df2['qir_all_%(qir_all_sf/qir_all)'] = df2['qir_all_sf']/df2['qir_all']
    df2['qir>2.5_%_diff_(qir>2.5_all_sf-qir>2.5_all)/qir>2.5_all_sf'] = (df2['qir>2.5_sf']-df2['qir>2.5_all'])/df2['qir>2.5_sf']
    df2['qir<1_%_diff_(qir<1_all_sf-qir<1_all)/qir<1_all_sf'] = (df2['qir<1_sf']-df2['qir<1_all'])/df2['qir<1_sf']   
    
                        
                        
    df2.to_csv('teff_medians10.csv')
 #   print(df2)
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
    lst = ["bh_accretion_rate_hh","bh_accretion_rate_sb","cnfw_subhalo","cooling_rate","descendant_id_galaxy","lambda_subhalo","m_bh","matom_bulge","matom_disk","mean_stellar_age","mgas_bulge","mgas_disk","mgas_metals_bulge","mgas_metals_disk","mhot","mhot_metals","mlost","mlost_metals","mmol_bulge","mmol_disk","mreheated","mreheated_metals","mstars_bulge","mstars_bulge_diskins_assembly","mstars_bulge_mergers_assembly","mstars_burst_diskinstabilities","mstars_burst_mergers","mstars_disk","mstars_metals_bulge","mstars_metals_bulge_diskins_assembly","mstars_metals_bulge_mergers_assembly","mstars_metals_burst_diskinstabilities","mstars_metals_burst_mergers","mstars_metals_disk","mvir_hosthalo","mvir_subhalo","redshift_merger","rgas_bulge","rgas_disk","rstar_bulge","rstar_disk","sfr_burst","sfr_disk","specific_angular_momentum_bulge_gas","specific_angular_momentum_bulge_star","specific_angular_momentum_disk_gas","specific_angular_momentum_disk_gas_atom","specific_angular_momentum_disk_gas_mol","specific_angular_momentum_disk_star","type","vmax_subhalo","vvir_hosthalo","vvir_subhalo"]

    IncludeSelfAbsorption = False
    
    for z in range(len(zlist)):
        idx_disk = lst.index('mstars_disk') + 2
        idx_bulge = lst.index('mstars_bulge') + 2
        idx_sdisk = lst.index('sfr_burst') + 2
        idx_sbulge = lst.index('sfr_disk') + 2
        
        sfr_disk = np.array(hdf5_lir_lst[z][idx_sdisk])
        sfr_disk_2 = []
        sfr_bulge = np.array(hdf5_lir_lst[z][idx_sbulge])
        sfr_bulge_2 = []
        sfr_tot = []
        sfr_tot = sfr_bulge + sfr_disk
        mdisk = np.array(hdf5_lir_lst[z][idx_disk])
        mbulge = np.array(hdf5_lir_lst[z][idx_bulge])
    #    r_gas_disk_array = np.array(hdf5_lir_lst[z][11]) #half radius of the gas disk [cMpc/h]
        


        ind0 = np.where(np.asarray(mdisk) + np.asarray(mbulge) > 0)
        ms = (mdisk[ind0] + mbulge[ind0])/h0 #in Msun
        sfr_tot = (sfr_tot[ind0])/h0/1e9
        #ionising mag
        lir_total = np.array(lir_total_W_lst[z], dtype=np.float64)
       # ind109 = np.where(lir_total > 10**7 * Lsun)

      #  lir_total = lir_total[ind109]
      # ms = ms[ind109]
       # sfr_tot= sfr_tot[ind109]
        ind108 = np.where(ms > 0)
        lir_total = lir_total[ind108]
        ms = ms[ind108]
        sfr_tot= sfr_tot[ind108]
      #  r_gas_disk_array = r_gas_disk_array[ind0]
       # r_gas_disk_array = r_gas_disk_array[ind108]
        
        


        n = 0
        
        def tau_factor(m, wave, rgas_disk,nu):
            wave_m = wave * 1e-10 #wavelength in m
            wave_cm = wave_m  * 1e2 #wavelength in cm
            
            T = 1e4 #Temperature in K
            
            freq = (c_speed/ wave) / 1e9 #Frequency in GHz
            
            ne = 10.0**((m + 48.6) / (-2.5)) * dfac * freq * wave_cm #number of electrons
            
            l = (rgas_disk/h/c_speed)*1e6 #distance in pc
            
            tau = 8.2 * 10**(-2) * T**(-1.35) * nu**(-2.1) * (ne**2 * l)
            
            return tau

        def ionising_photons(m, wave):
            #m is a vector of AB absolute magnitudes in a band with central wavelength wave
            #wavelength input has to be in angstrom

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
            lum = Q/6.3e32 * (T/1e4)**0.45 * (nu)**(-0.1)

            return lum

        def synchrotron_lum(SFR, nu):

        #SFR in Msun/yr
        #nu is the frequency in GHz
            #output in erg/s/Hz

            ENT = 1.44 #1.44 ir 1.38
            ESNR = 0.06 * ENT
            alpha = 0.8

            comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(-alpha)
            nuSNCC = SFR * 0.0219157 #0.0095
            lum = comp1 * 1e30
            lum = lum * nuSNCC
            
            if(IncludeSelfAbsorption == True):
                for i in (range(100)):

                
                    comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(alpha) * np.e**(-tau)
            
            
            return lum


        total_mags_nod = np.array(seds_bands_lst[z][2], dtype=np.float64) #no_dust
        ion_mag = total_mags_nod[1,:]
     #   ion_mag = ion_mag[ind109]
        ion_mag = ion_mag[ind108]
        q_ionis = ionising_photons(ion_mag, 912.0) #in s^-1
 
        selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15)
        lum_radio = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        freefree = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        sync = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        for i, nu in enumerate(selection_freq):

            freefree[i,:] = freefree_lum(q_ionis[:], nu)/1e7
            
            sync[i,:] = synchrotron_lum(sfr_tot[:], nu)/1e7
            
            lum_radio[i,:] = freefree[i,:] + sync[i,:]

        
        
        qIR_bressan = np.log10(lir_total/3.75e12) - np.log10(lum_radio[3,:])
        

        inf_lst = np.where(qIR_bressan == np.inf)
        notinf_lst = np.where(qIR_bressan != np.inf)
        qir_lst.append(qIR_bressan)
        qir_bress.append(np.median(qIR_bressan[notinf_lst]))
        
        bress_rad_lum.append(lum_radio[3][notinf_lst])

        bress_lir_total.append(np.median(lir_total[notinf_lst]))

        
        freefree_lst.append(freefree[3])
        sync_lst.append(sync[3])

    
    return qir_lst, qir_bress, bress_rad_lum, bress_lir_total, freefree_lst, sync_lst
    
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

        x_bins = np.where(((xlist > m)&(xlist < n)))


        y_bins = ylist[x_bins]


        med_y = np.median(y_bins)
        med_y_lst.append(med_y)
        mid_x = (n/2+m/2)
        mid_x_lst.append(mid_x)
        try:
            low_y_lst.append(np.percentile(y_bins,16))
            high_y_lst.append(np.percentile(y_bins,84))
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


def main(model_dir, outdir, redshift_table, subvols, obsdir):


    plt = common.load_matplotlib()

    file_name = "eagle-rr14-radio-only"
    file_hdf5_sed = "Shark-SED-" + file_name + ".hdf5"

    fields_sed = {'SED/lir_dust': ('disk','bulge_t','total'),}
    fields_sed_bc = {'SED/lir_dust_contribution_bc':('disk', 'total')}
    fields_seds_nodust = {'SED/ab_nodust':('disk','bulge_t','total'),}
    fields_seds_dust = {'SED/ab_dust':('disk','bulge_t','total'),}

  #  fields = {'galaxies': ('mstars_disk', 'mstars_bulge','sfr_disk', 'sfr_burst','type','mgas_metals_bulge', 'mgas_metals_disk')}

    fields = {'galaxies': ("bh_accretion_rate_hh","bh_accretion_rate_sb","cnfw_subhalo","cooling_rate","descendant_id_galaxy","lambda_subhalo","m_bh","matom_bulge","matom_disk","mean_stellar_age","mgas_bulge","mgas_disk","mgas_metals_bulge","mgas_metals_disk","mhot","mhot_metals","mlost","mlost_metals","mmol_bulge","mmol_disk","mreheated","mreheated_metals","mstars_bulge","mstars_bulge_diskins_assembly","mstars_bulge_mergers_assembly","mstars_burst_diskinstabilities","mstars_burst_mergers","mstars_disk","mstars_metals_bulge","mstars_metals_bulge_diskins_assembly","mstars_metals_bulge_mergers_assembly","mstars_metals_burst_diskinstabilities","mstars_metals_burst_mergers","mstars_metals_disk","mvir_hosthalo","mvir_subhalo","redshift_merger","rgas_bulge","rgas_disk","rstar_bulge","rstar_disk","sfr_burst","sfr_disk","specific_angular_momentum_bulge_gas","specific_angular_momentum_bulge_star","specific_angular_momentum_disk_gas","specific_angular_momentum_disk_gas_atom","specific_angular_momentum_disk_gas_mol","specific_angular_momentum_disk_star","type","vmax_subhalo","vvir_hosthalo","vvir_subhalo")}

    
    
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

    
    seds_lir_lst = []
    
    hdf5_lir_lst = []
    
    seds_lir_bc_lst = []
    
    seds_bands_lst = []
    
    seds_nodust_lst = []
    
    lir_total_W_lst = []
    
    Teff_lst = []
    for index, snapshot in enumerate(redshift_table[zlist]):
      #  print("Will read snapshot %s" % (str(snapshot)))
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
        
        
    


  
    df,sf_lst = dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst)
  #  rad_size(df)
 #   density(df)
  #  qir_z_plt(df)
   # schr_vs_med(df)
    #fir_lir_mass(df)
    
    #qir_with_m_plt(df)
    
    #q_m_m_z(df)
    
    #qir_vs_qir(df)
    
    #sfr_rad_lum(df)
    
   # rad_lum_func_plt(df)

   # hist(df)
   # qir_with_m_plt(df)
    
    #lum_m(df,sf_lst)
  #  qir_metals_mass_bins(df)
  #  metal_hist(df)
    
  #  gas_metal_vs_stellar_mass(df)
    
    #derived_SFR(df)
    
   # qir_v_dust_ff_sync(df)       
    #lo_faro_plots(df)
    
    #GAMA_plots(df)
  #  GAMA_flux_limit(df)
    met_dist_cent(df)
   # qir_plots(df)
    
   # SFR_z(df)
    
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
    
    #upload to magnus: scp temp_properties.py shansen@magnus.pawsey.org.au:/scratch/pawsey0119/shansen       
    #download file from magnus: scp shansen@magnus.pawsey.org.au:/scratch/pawsey0119/shansen/teff_medians3.csv .  