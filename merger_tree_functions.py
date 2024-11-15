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

import numpy as np
from astropy.io import ascii
from astropy import table
import csv

base_output=925
time=ascii.read('./output_step_redshift.txt')
z_base=float(time['z'][time['output']==base_output])

filename = f'./galaxy_ids_by_mass_{base_output}.csv'
with open(filename) as f:
    reader=csv.reader(f)
    next(reader)
    all_galaxies=np.array([row[0] for row in reader]).astype('int')

def find_target_galaxy(merger_tree, gal_id):
    try:
        branch=merger_tree['level']==gal_id
        target_id=merger_tree[branch][0]['child_id']
        target_output=merger_tree[branch][0]['output_index']+1
        mask=np.logical_and(merger_tree['gal_id']==target_id,merger_tree['output_index']==target_output)
        
        return merger_tree[mask]
    except:
        mask=np.logical_and(merger_tree['gal_id']==target_id,merger_tree['output_index']==(target_output-1))
        return merger_tree[mask]
    
def load_merger_tree(base_id):
    tree_file_name=f'./merger_trees/merger_tree_o925_gal{str(base_id).zfill(4)}.csv'
    return ascii.read(tree_file_name)

def get_merger_list(merger_tree):
    # This function collects a list of all mergers within the merger tree
    # merger_tree: Table loaded via load_merger_tree
    merger_list=table.Table(names=('primary_level','secondary_level','primary_mass','secondary_mass',
                                   'merger_output_index','merger_time','secondary_mass_max'),
                      dtype=('i4','i4','f4','f4','i4','f4','f4'))
    branches=list(set(merger_tree['level']))
    branches.sort()
    for branch in branches[1:]:
        mask=merger_tree['level']==branch
        main=find_target_galaxy(merger_tree,branch)
        merger_list.add_row([main['level'],branch,main['Mgal'],merger_tree[mask]['Mgal'][0],
                             main['output_index'],main['time'],merger_tree[mask]['Mgal'].max()])
    merger_list.add_column(merger_list['secondary_mass']/merger_list['primary_mass'],name='merger_ratio')
    return merger_list

def select_mergers(merger_list,base_id,q_min=0,Mnsc=0):
    # This function filters out mergers that meet the supplied criteria, and computes the time at which 
    # the conditions for NSC formation are met. Write files in ./NSC_trees 
    # merger_list: the merger table returned by get_merger_list
    # base_id: The id of the galaxy, used to name the file
    # q_min: The minimum mass ratio at merger for a branch to be retained
    # Mnsc: The minimum galaxy stellar mass for a branch to be retained. Also the mass at which the NSC forms
    selected_mergers=table.Table(names=('primary_level','secondary_level','primary_mass','secondary_mass',
                                  'merger_output_index','merger_time','secondary_mass_max','mass_ratio'),
                      dtype=('i4','i4','f4','f4','i4','f4','f4','f4'))
    
    merger_tree = load_merger_tree(base_id) # ADDED REQUIRED LINE
    
    selected_mergers.add_row([1,1,merger_tree['Mgal'].max(),merger_tree['Mgal'].max(),
                          merger_tree['output_index'].max(),merger_tree['time'].max(),
                          merger_tree['output_index'].max(),1])
    
    m_new=[merger for merger in merger_list[merger_list['primary_level']==1] if 
           (merger['merger_ratio']>q_min and merger['secondary_mass_max']>Mnsc)]
    m_ids=[merger['secondary_level'] for merger in m_new]
    #print(len(m_new),'top level mergers')
    i=0
    while len(m_new)>0:
        i+=1
        [selected_mergers.add_row(merger) for merger in m_new]
        m_new=[merger for merger in merger_list if (merger['primary_level'] in m_ids and 
                                                    (merger['merger_ratio']>q_min and 
                                                     merger['secondary_mass_max']>Mnsc))]
        m_ids=[merger['secondary_level'] for merger in m_new]
        
    #Add time when NSC is being created
    times=[]
    for branch in selected_mergers['secondary_level']:
        branch_mask=merger_tree['level']==branch
        t_ini=merger_tree[branch_mask]['time'][merger_tree[branch_mask]['Mgal']>Mnsc][-1]
        times.append(t_ini)
            
    selected_mergers.add_column(np.array(times),name='t_formation')
    selected_mergers.remove_columns(['secondary_mass_max'])
    
    return(selected_mergers)

# end of file.