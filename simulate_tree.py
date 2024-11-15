"""
 Copyright (C) 2024  Konstantinos Kritos <kkritos1@jhu.edu>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from global_constants import *
from merger_tree_functions import *
from simulation_functions import *
from merger_remnant import *
import argparse
from astropy.cosmology import z_at_value
import astropy.units as u
import pickle

parser = argparse.ArgumentParser(description='Simulate Tree input parameters')

parser.add_argument('-q', '--minimum_merger_mass_ratio', type=float, metavar=' ', default=0, help='minimum merger mass ratio')
parser.add_argument('-M', '--stellar_galaxy_mass_threshold', type=float, metavar=' ', default=1.0E7, help='Minimum galaxy stellar mass')
parser.add_argument('-Q', '--major_merger_mass_ratio_threshold', type=float, metavar=' ', default=0.10, help='Major merger mass ratio threshold')
parser.add_argument('-S', '--seed_number', type=int, metavar=' ', default=34563589, help='Seed number')
parser.add_argument('-i', '--gal_id', type=int, metavar=' ', default=0, help='Galaxy id')
parser.add_argument('-t', '--gas_expulsion_timescale', type=float, metavar=' ', default=100.0, help='gas expulsion timescale')
parser.add_argument('-g', '--max_gas_fraction_parameter', type=float, metavar=' ', default=1.0, help='maximum fraction of the NSC initial total mass in gas that did not fragment into stars')
parser.add_argument('-s', '--max_star_fraction_parameter', type=float, metavar=' ', default=1.0, help='maximum fraction of the galaxy mass into stars of the NSC')

args = parser.parse_args()

q_min = args.minimum_merger_mass_ratio
Mnsc = args.stellar_galaxy_mass_threshold
q_thres = args.major_merger_mass_ratio_threshold
seed = args.seed_number
gal_id = args.gal_id
tge = args.gas_expulsion_timescale
maxgasfrac = args.max_gas_fraction_parameter
maxstarsfrac = args.max_star_fraction_parameter

if __name__ == "__main__":
    
    print('seed=', seed)
    
    # initialize pseudorandom number generator:
    np.random.seed(seed)
    
    # get the merger tree:
    selected_mergers = None
    if gal_id == 0:
        while selected_mergers is None:
            try:
                # get galaxy's base ID:
                gal_id = np.random.choice(all_galaxies)
                print(gal_id)
                # load corresponding merger tree:
                merger_tree = load_merger_tree(gal_id)
                # get list of mergers:
                merger_list = get_merger_list(merger_tree)
                # filter merger list:
                selected_mergers = select_mergers(merger_list, gal_id, q_min=q_min, Mnsc=Mnsc)
            except:
                selected_mergers = None
    else:
        try:
            # load corresponding merger tree:
            merger_tree = load_merger_tree(gal_id)
            # get list of mergers:
            merger_list = get_merger_list(merger_tree)
            # filter merger list:
            selected_mergers = select_mergers(merger_list, gal_id, q_min=q_min, Mnsc=Mnsc)
        except: # terminate program
            exit()
            
    # number of merger nodes (the last node corresponds to the final time, so it's not a genuine node):
    N_nodes = selected_mergers['merger_time'].size - 1
    
    # sort indices in chronological order:
    indices_sorted = np.argsort(selected_mergers['merger_time'].value)
    
    selected_mergers_ordered = {
        'primary_level': selected_mergers['primary_level'].value[indices_sorted], 
        'secondary_level': selected_mergers['secondary_level'].value[indices_sorted], 
        'mass_ratio': selected_mergers['mass_ratio'].value[indices_sorted], 
        't_formation': selected_mergers['t_formation'].value[indices_sorted], 
        'merger_time': selected_mergers['merger_time'].value[indices_sorted], 
    }
    
    print(selected_mergers)
    # initialize BH tree:
    BH_tree = {
        'mass': np.zeros(N_nodes+1), 
        'spin': np.zeros(N_nodes+1), 
        'level': np.unique(np.concatenate((selected_mergers_ordered['primary_level'], 
                                                  selected_mergers_ordered['secondary_level']))), 
        'status': np.zeros(N_nodes+1)
    }
    
    # initialize NSC tree:
    NSC_tree = {
        'N_star': np.zeros(N_nodes+1), 
        'r_h': np.zeros(N_nodes+1), 
        'level': np.unique(np.concatenate((selected_mergers_ordered['primary_level'], 
                                                  selected_mergers_ordered['secondary_level'])))
    }
    
    BH_mergers = []
    
    ejected_bhs = []
    
    nuclear_star_clusters = []
    
    # iterate over all nodes of the merger tree in chronological order:
    for i in range(0, N_nodes):
        
        # read primary and secondary levels of the merger node:
        pl, sl = selected_mergers_ordered['primary_level'][i], selected_mergers_ordered['secondary_level'][i]
        
        # get current merger time:
        tm = selected_mergers_ordered['merger_time'][i] * 1.0E3
        
        # get time of current primary branch's previous node (either merger or formation):
        mask_p = ((selected_mergers['primary_level']==pl) | (selected_mergers['secondary_level']==pl)) & (selected_mergers['merger_time'] < tm/1.0E3)
        try:
            tp = selected_mergers[mask_p]['merger_time'].max() * 1.0E3
            Ip = 1 # indicator if primary branch had previous merger (=0 no, =1 yes)
        except: # no previous merger
            tp = selected_mergers[selected_mergers['secondary_level'] == pl]['t_formation'].value[0] * 1.0E3
            Ip = 0 # indicator if primary branch had previous merger (=0 no, =1 yes)
            
        # get time of current secondary branch's previous node (either merger or formation):
        mask_s = ((selected_mergers['primary_level']==sl) | (selected_mergers['secondary_level']==sl)) & (selected_mergers['merger_time'] < tm/1.0E3)
        try:
            ts = selected_mergers[mask_s]['merger_time'].max() * 1.0E3
            Is = 1 # indicator if secondary branch had previous merger (=0 no, =1 yes)
        except: # no previous merger
            ts = selected_mergers[selected_mergers['secondary_level'] == sl]['t_formation'].value[0] * 1.0E3
            Is = 0 # indicator if secondary branch had previous merger (=0 no, =1 yes)
            
        print('i=', i, 'pl=', pl, 'sl=', sl, 'tm=', tm/1e3, 'tp=', tp/1e3, 'ts=', ts/1e3, 'Ip=', Ip, 'Is=', Is)
        
        m_star0 = mean_stellar_mass(0.08, 150, 0.08, 150) # initial average stellar mass
        
        # simulate primary branch:
        if Ip == 0: # the primary galaxy did not have a previous merger episode
            
            u = np.random.rand() * maxstarsfrac # fraction of galaxy's stellar mass locked in the NSC initially
            fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
            
            N0p = u * Mnsc / m_star0
            rh0p = 10**np.random.uniform(-1, 1)
            Mgas0p = u * fg * Mnsc / (1 - fg)
            MH0p = 10.0
            SH0p = 0.0
            
        elif Ip == 1: # the primary galaxy had a previous merger episode
            
            fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
            
            N0p = NSC_tree['N_star'][NSC_tree['level']==pl][0]
            rh0p = NSC_tree['r_h'][NSC_tree['level']==pl][0]
            Mgas0p = N0p * m_star0 * fg / (1 - fg)
            MH0p = BH_tree['mass'][BH_tree['level']==pl][0]
            SH0p = BH_tree['spin'][BH_tree['level']==pl][0]
            
        # simulate secondary branch:
        if Is == 0: # the primary galaxy did not have a previous merger episode
            
            u = np.random.rand() * maxstarsfrac # fraction of galaxy's stellar mass locked in the NSC initially
            fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
            
            N0s = u * Mnsc / m_star0
            rh0s = 10**np.random.uniform(-1, 1)
            Mgas0s = u * fg * Mnsc / (1 - fg)
            MH0s = 10.0
            SH0s = 0.0
            
        elif Is == 1: # the primary galaxy had a previous merger episode
            
            fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
            
            N0s = NSC_tree['N_star'][NSC_tree['level']==sl][0]
            rh0s = NSC_tree['r_h'][NSC_tree['level']==sl][0]
            Mgas0s = N0s * m_star0 * fg / (1 - fg)
            MH0s = BH_tree['mass'][BH_tree['level']==sl][0]
            SH0s = BH_tree['spin'][BH_tree['level']==sl][0]
            
        # simulate primary nuclear star cluster:
        cluster_p = simulate_cluster(N0=N0p, rh0=rh0p, Mgas0=Mgas0p, mBH0=10.0, MH0=MH0p, SH0=SH0p, tmax=(tm - tp), dt=1, tge=tge, seed=np.random.randint(999999999))
        cluster_p['T_0'] = tm * np.ones(len(cluster_p['t']))
        nuclear_star_clusters.append(cluster_p)
        
        # simulate secondary nuclear star cluster:
        cluster_s = simulate_cluster(N0=N0s, rh0=rh0s, Mgas0=Mgas0s, mBH0=10.0, MH0=MH0s, SH0=SH0s, tmax=(tm - ts), dt=1, tge=tge, seed=np.random.randint(999999999))
        cluster_s['T_0'] = tm * np.ones(len(cluster_s['t']))
        nuclear_star_clusters.append(cluster_s)
        
        BH_tree['mass'][BH_tree['level']==pl] = cluster_p['M_H'][-1] # update BH mass of primary level
        BH_tree['spin'][BH_tree['level']==pl] = cluster_p['S_H'][-1] # update BH spin of primary level
        BH_tree['status'][BH_tree['level']==pl] = 1 # update BH status of primary level (nuclear BH)
        
        BH_tree['mass'][BH_tree['level']==sl] = cluster_s['M_H'][-1] # update BH mass of secondary level
        BH_tree['spin'][BH_tree['level']==sl] = cluster_s['S_H'][-1] # update BH spin of secondary level
        BH_tree['status'][BH_tree['level']==sl] = 1 # update BH status of secondary level (nuclear BH)
        
        # NSC merger episode:
        N_new = cluster_p['N_star'][-1] + cluster_s['N_star'][-1]
        R_new = N_new**2 / (cluster_p['N_star'][-1]**2 / cluster_p['r_h'][-1] + cluster_s['N_star'][-1]**2 / cluster_s['r_h'][-1])
        
        # update primary level NSC:
        NSC_tree['N_star'][NSC_tree['level']==pl] = N_new
        NSC_tree['r_h'][NSC_tree['level']==pl] = R_new
        
        fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
        Mgas0_new = N_new * m_star0 * fg / (1 - fg)
        
        # check if the halo-halo merger is major:
        if selected_mergers_ordered['mass_ratio'][i] > q_thres: # central NSCs of both halos merge
            
            # check if both halos have nuclear BHs:
            if BH_tree['status'][BH_tree['level']==pl]==1 and \
               BH_tree['status'][BH_tree['level']==sl]==1: # both halos have nuclear BHs; BH-BH merger occurs
                
                # read binary parameters:
                mP = BH_tree['mass'][BH_tree['level']==pl] # BH mass of primary halo
                sP = BH_tree['spin'][BH_tree['level']==pl] # BH spin of primary halo
                mS = BH_tree['mass'][BH_tree['level']==sl] # BH mass of secondary halo
                sS = BH_tree['spin'][BH_tree['level']==sl] # BH spin of secondary halo
                qB = min(mP, mS) / max(mP, mS) # binary mass ratio
                
                # draw seed number:
                seed = np.random.randint(999999999)
                
                # compute binary merger remnant properties:
                mR = remnant_mass(m1=mP, m2=mS, chi1=sP, chi2=sS, seed=seed) # remnant mass
                sR = remnant_spin(q=qB, chi1=sP, chi2=sS, seed=seed) # remnant spin
                vR = remnant_kick(q=qB, chi1=sP, chi2=sS, seed=seed) # GW kick velocity
                
                v_esc = np.sqrt(kappa * G_Newt * (N_new * m_star0 + Mgas0_new) / R_new) # escape velocity from nucleus of merged galaxy
                
                BH_tree['mass'][BH_tree['level']==pl] = mR # update BH mass of primary level
                BH_tree['spin'][BH_tree['level']==pl] = sR # update BH spin of primary level
                
                # check if remnant BH is ejected from the nucleus:
                if vR < v_esc: # BH retained in the nucleus
                    
                    BH_tree['status'][BH_tree['level']==pl] = 1 # update BH status of primary level (nuclear BH)
                    
                else: # BH ejected from the nucleus; becomes wandering BH
                    
                    BH_tree['status'][BH_tree['level']==pl] = -1 # update BH status of primary level (free BH)
                    
                    ejected_bhs.append(np.array([mR[0], sR[0], vR[0], v_esc, tm]))
                    
                # BH of secondary level merged:
                BH_tree['mass'][BH_tree['level']==sl] = 0.0 # update BH mass of secondary level
                BH_tree['spin'][BH_tree['level']==sl] = 0.0 # update BH spin of secondary level
                BH_tree['status'][BH_tree['level']==sl] = 0 # # update BH status of secondary level (merged BH)
                
                BH_mergers.append(np.array([seed, tm, mP[0], mS[0], sP[0], sS[0], mR[0], sR[0], vR[0], v_esc]))
                
            else: # not both halos have nuclear BHs; no BH merger occurs
                
                if BH_tree['status'][BH_tree['level']==sl]==1:
                    
                    # BH of secondary level deposited into the primary nucleus:
                    BH_tree['mass'][BH_tree['level']==pl] = BH_tree['mass'][BH_tree['level']==sl] # update BH mass of primary level
                    BH_tree['spin'][BH_tree['level']==pl] = BH_tree['spin'][BH_tree['level']==sl] # update BH spin of primary level
                    BH_tree['status'][BH_tree['level']==pl] = 1 # # update BH status of primary level
                    
                else:
                    
                    pass
                    
        else: # the smaller galaxy dissolves in the halo of the larger galaxy
            
            if BH_tree['status'][BH_tree['level']==sl] == 1:
                
                # the BH of the secondary halo becomes off-nuclear but trapped in the core of an ultra-compact dwarf:
                BH_tree['status'][BH_tree['level']==sl] = 2 # (satellite BH)
                
    # simulate final branch:
    
    if N_nodes==0: # check if the galaxy did not have merger history
        
        # formation time:
        tm = selected_mergers[0]['t_formation'] * 1.0E3
        
        m_star0 = mean_stellar_mass(0.08, 150, 0.08, 150) # initial average stellar mass
        
        u = np.random.rand() * maxstarsfrac # fraction of galaxy's stellar mass locked in the NSC initially
        fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
        
        N0p = u * Mnsc / m_star0
        rh0p = 10**np.random.uniform(-1, 1)
        Mgas0p = u * fg * Mnsc / (1 - fg)
        MH0p = 10.0
        SH0p = 0.0
        
    else:
        
        fg = np.random.rand() * maxgasfrac # fraction of NSC's initial total mass in gas that did not fragment into stars
        
        N0p = NSC_tree['N_star'][NSC_tree['level']==pl][0]
        rh0p = NSC_tree['r_h'][NSC_tree['level']==pl][0]
        Mgas0p = N0p * m_star0 * fg / (1 - fg)
        MH0p = BH_tree['mass'][BH_tree['level']==pl][0]
        SH0p = BH_tree['spin'][BH_tree['level']==pl][0]
        
    # final time:
    t_final = selected_mergers[0]['merger_time'] * 1.0E3
    
    # simulate primary nuclear star cluster:
    cluster_p = simulate_cluster(N0=N0p, rh0=rh0p, Mgas0=Mgas0p, mBH0=10.0, MH0=MH0p, SH0=SH0p, tmax=(t_final - tm), dt=1, tge=tge, seed=np.random.randint(999999999))
    cluster_p['T_0'] = tm * np.ones(len(cluster_p['t']))
    nuclear_star_clusters.append(cluster_p)
    
    BH_tree['mass'][BH_tree['level']==1] = cluster_p['M_H'][-1] # update BH mass of primary level
    BH_tree['spin'][BH_tree['level']==1] = cluster_p['S_H'][-1] # update BH spin of primary level
    BH_tree['status'][BH_tree['level']==1] = 1 # update BH status of primary level (nuclear BH)
    
    # update primary level NSC:
    NSC_tree['N_star'][NSC_tree['level']==1] = cluster_p['N_star'][-1]
    NSC_tree['r_h'][NSC_tree['level']==1] = cluster_p['r_h'][-1]
    
    with open('BH_tree.txt', 'w') as f:
        
        for i in range(N_nodes+1):
            
            # primary galaxy's final stellar mass:
            M_gal = selected_mergers['primary_mass'][selected_mergers['secondary_level']==1].value[0] if BH_tree['level'][i]==1 else 0.0
            
            # primary galaxy's NSC mass and radius:
            M_NSC = NSC_tree['N_star'][0] if NSC_tree['level'][i]==1 else 0.0
            R_NSC = NSC_tree['r_h'][0] if NSC_tree['level'][i]==1 else 0.0
            
            f.write(str(BH_tree['mass'][i]) + ' ' + str(BH_tree['spin'][i]) + ' ' + str(BH_tree['level'][i]) + ' ' + str(BH_tree['status'][i]) + ' ' + str(M_gal) + ' ' + str(M_NSC) + ' ' + str(R_NSC) + ' ' + str(N_nodes))
            
            f.write('\n') # new line
            
    with open('BH_mergers.txt', 'w') as g:
        
        for j in range(len(BH_mergers)):
            
            g.write(str(BH_mergers[j][0]) + ' ' + str(BH_mergers[j][1]) + ' ' + str(BH_mergers[j][2]) + ' ' + str(BH_mergers[j][3]) + ' ' + str(BH_mergers[j][4]) + ' ' + str(BH_mergers[j][5]) + ' ' + str(BH_mergers[j][6]) + ' ' + str(BH_mergers[j][7]) + ' ' + str(BH_mergers[j][8]) + ' ' + str(BH_mergers[j][9]))
            
            g.write('\n')
            
    with open('ejected_bhs.txt', 'w') as g:
        
        for l in range(len(ejected_bhs)):
            
            g.write(str(ejected_bhs[l][0]) + ' ' + str(ejected_bhs[l][1]) + ' ' + str(ejected_bhs[l][2]) + ' ' + str(ejected_bhs[l][3]) + ' ' + str(ejected_bhs[l][4]))
            
            g.write('\n')
            
    with open('nuclear_star_clusters.pkl', 'wb') as h:
        
        pickle.dump(nuclear_star_clusters, h)
        
# end of file.