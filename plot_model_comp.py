import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil

folder = "./dynamic_GLMiHMM_crossvals/infos/"
doubles = "./dynamic_GLMiHMM_crossvals/infos_extra/"

# Define conditions
condition_names = [
    {'path': '/summarised_sessions/0_3/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_35/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_25/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_2/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_15/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_4/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.06},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.12},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.02},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.01},
    {'path': '/summarised_sessions/0_3 no dur/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.24},
    {'path': '/summarised_sessions/0_3 1 state/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 full dur/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 1 state/', 'variance': 0.},
    {'path': '/summarised_sessions/0_3 her gam/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 herer gam/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 ler alp/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 lerer alp/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 ler lowest dur/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_25 new setup/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 new setup/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3 new setup/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 new setup betgam/', 'variance': 0.04},  # better gamma
    {'path': '/summarised_sessions/0_3 mix/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_25 new setup long dur/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_3 new setup/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_4 new setup/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_2 new setup/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 new setup/', 'variance': 0.06},
    {'path': '/summarised_sessions/0_25 best setup/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_25 best denser/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best low gam/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best high alp/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best low alp/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 sndpla denser/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best/', 'variance': 0.06},
    {'path': '/summarised_sessions/0_25 best/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_2 best/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_3 best/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best moretime/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best morestate/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best wsls/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_25 best 3state/', 'variance': 0.},
    {'path': '/summarised_sessions/0_25 best 15state/', 'variance': 0.}
]

# no-shows: 'NYU-21', 'ZFM-04308' 
subject_names = ['CSHL045', 'CSHL047', 'CSHL049', 'CSHL051', 'CSHL052', 'CSHL053', 'CSHL054', 'CSHL055', 'CSHL058', 'CSHL059', 'CSHL060', 'CSHL_007', 'CSHL_014', 'CSHL_015',
           'CSHL_020', 'CSH_ZAD_001', 'CSH_ZAD_011', 'CSH_ZAD_017', 'CSH_ZAD_019', 'CSH_ZAD_022', 'CSH_ZAD_024', 'CSH_ZAD_025', 'CSH_ZAD_026', 'CSH_ZAD_029', 'DY_008',
           'DY_009', 'DY_010', 'DY_011', 'DY_013', 'DY_014', 'DY_016', 'DY_018', 'DY_020', 'KS014', 'KS015', 'KS016', 'KS017', 'KS019', 'KS021', 'KS022', 'KS023',
           'KS042', 'KS043', 'KS044', 'KS045', 'KS046', 'KS051', 'KS052', 'KS055', 'KS084', 'KS086', 'KS091', 'KS094', 'KS096', 'MFD_05', 'MFD_06', 'MFD_07', 'MFD_08',
           'MFD_09', 'NR_0017', 'NR_0019', 'NR_0020', 'NR_0021', 'NR_0024', 'NR_0027', 'NR_0028', 'NR_0029', 'NR_0031', 'NYU-06', 'NYU-11', 'NYU-12', 'NYU-27',
           'NYU-30', 'NYU-37', 'NYU-39', 'NYU-40', 'NYU-45', 'NYU-46', 'NYU-47', 'NYU-48', 'NYU-65', 'PL015', 'PL016', 'PL017', 'PL024', 'PL030', 'PL031', 'PL033',
           'PL034', 'PL035', 'PL037', 'PL050', 'SWC_021', 'SWC_022', 'SWC_023', 'SWC_038', 'SWC_039', 'SWC_042', 'SWC_043', 'SWC_052', 'SWC_053', 'SWC_054', 'SWC_058',
           'SWC_060', 'SWC_061', 'SWC_065', 'SWC_066', 'UCLA005', 'UCLA006', 'UCLA011', 'UCLA012', 'UCLA014', 'UCLA015', 'UCLA017', 'UCLA030', 'UCLA033', 'UCLA034',
           'UCLA035', 'UCLA036', 'UCLA037', 'UCLA044', 'UCLA048', 'UCLA049', 'UCLA052', 'ZFM-01576', 'ZFM-01577', 'ZFM-01592', 'ZFM-01935', 'ZFM-01936', 'ZFM-01937',
           'ZFM-02368', 'ZFM-02369', 'ZFM-02370', 'ZFM-02372', 'ZFM-02373', 'ZFM-05236', 'ZM_1897', 'ZM_1898', 'ZM_2240', 'ZM_2241', 'ZM_2245', 'ZM_3003',
           'ibl_witten_13', 'ibl_witten_14', 'ibl_witten_16', 'ibl_witten_17', 'ibl_witten_18', 'ibl_witten_19', 'ibl_witten_20', 'ibl_witten_25', 'ibl_witten_26',
           'ibl_witten_27', 'ibl_witten_29', 'ibl_witten_32']

subjects_and_nums = [(subject, num) for subject in subject_names for num in range(2)]

condition_subs = {i: subjects_and_nums.copy() for i in range(len(condition_names))}

conditions = [
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_35/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_2/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_15/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_4/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.06 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.12 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.02 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.01 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' in x and x['dur'] == 'no') and x['n_states'] == 15,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.24 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' in x and x['dur'] == 'no') and x['n_states'] == 1,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' in x and x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 700 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' in x and x['dur'] == 'no') and x['n_states'] == 1,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.1 and x['gamma_b_0'] == 10 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.001 and x['alpha_b_0'] == 1000,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and x['dur_params']['r_support'][0] == 2 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    
    # 112 - 32
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(2, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_4/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_2/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.06 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.1 and x['gamma_b_0'] == 10 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.001 and x['alpha_b_0'] == 1000,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.06 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_2/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 905)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 18 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25_wsls/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 3 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
]


# Initialize dictionaries to store results
cvlls = {i: [] for i in range(len(conditions))}
full_cvlls = {i: [] for i in range(len(conditions))}
subjects = {i: [] for i in range(len(conditions))}

# Process files
non_cross_vals = []

compare = [32, 39, 40, 37, 38, 12]
compare_res = [np.zeros(308), np.zeros(308), np.zeros(308), np.zeros(308), np.zeros(308), np.zeros(308)]

for fol in [folder, doubles]:
    for file in tqdm(os.listdir(fol)):

        infos = json.load(open(fol + file, 'r'))
        subject = infos['subject']
        if not infos['cross_val']:
            non_cross_vals.append(file)
            continue
        if infos['n_samples'] == 10:
            continue
        assert infos['n_samples'] in [12000, 10000]
        # if subject == "KS014":
        #     print(file)

        if infos['cross_val_type'] == 'lenca':
            os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_lenca/" + file)
            continue

        if infos['cross_val_num'] not in [0, 1]:
            continue

        for i, condition in enumerate(conditions):
            if conditions[i](infos):
                if subject in subjects[i] and np.mean(infos['cross_val_preds'][-4000:]) in cvlls[i]:
                    # print(file)
                    os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
                    break
                else:
                    if i == 32:
                        shutil.copy(fol + file, "./dynamic_GLMiHMM_crossvals/infos_32/" + file)
                    if (subject, infos['cross_val_num']) not in condition_subs[i]:
                        os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_exists_with_different_cross_val/" + file)
                    else:
                        cvlls[i].append(np.mean(infos['cross_val_preds'][-4000:]))
                        full_cvlls[i].append(np.convolve(infos['cross_val_preds'], np.ones(4000)/4000, mode='valid'))
                        subjects[i].append(subject)
                        condition_subs[i].remove((subject, infos['cross_val_num']))

                        if i in compare:
                            compare_res[compare.index(i)][subjects_and_nums.index((subject, infos['cross_val_num']))] += np.mean(infos['cross_val_preds'][-4000:])
                    break

        else:
            print("fits no schema: " + infos['file_name'])
            quit()

# Plot boxplots
plt.figure(figsize=(16, 8))
labels = [f"{cond['variance']} var {cond['path'].split('/')[-2]}" for cond in condition_names]
plt.boxplot([100 * np.exp(cvlls[i]) for i in range(len(conditions))], labels=labels)
plt.ylabel("Exponentiated log likelihood on heldout", fontsize=16)
plt.xticks(range(1, len(conditions) + 1), labels)
plt.savefig("boxplot.png")
plt.close()

from scipy.stats import sem

print([len(cvlls[x]) for x in cvlls])

def comp_perfs(a, b, second_model):
    mask = np.logical_and(a != 0, b != 0)

    plt.figure(figsize=(16, 9))
    plt.scatter(a[mask], b[mask])
    plt.plot([a[mask].min(), a[mask].max()], [a[mask].min(), a[mask].max()], 'k')
    plt.xlabel("best model cvll", fontsize=22)
    plt.ylabel(second_model, fontsize=22)

plt.close()

# comp_perfs(compare_res[0], compare_res[1], "decay=0.2")
# plt.tight_layout()
# plt.savefig("comp_1")
# plt.show()

# comp_perfs(compare_res[0], compare_res[2], "decay=0.3")
# plt.tight_layout()
# plt.savefig("comp_2")
# plt.close()

# comp_perfs(compare_res[0], compare_res[3], "var=0.06")
# plt.tight_layout()
# plt.savefig("comp_3")
# plt.close()

# comp_perfs(compare_res[0], compare_res[4], "var=0.03")
# plt.tight_layout()
# plt.savefig("comp_4")
# plt.close()

# plot lines of the different means
plt.figure(figsize=(16, 8))
vals = []
for i, label in enumerate(labels):
    plt.errorbar(i, np.mean(cvlls[i]), yerr=sem(cvlls[i]) / 2, fmt='o', label=label)
    vals.append(np.mean(cvlls[i]))
    print(i, label, np.mean(cvlls[i]))
plt.ylabel("Exponentiated log likelihood on heldout", fontsize=16)
# plt.ylim(np.sort(vals)[-2] - 0.006, np.sort(vals)[-2] + 0.001)
plt.xticks(range(len(conditions)), labels, rotation=45)
plt.savefig("means.png")
plt.show()

# some filled up arrays only go to 10000
full_cvlls = {x: np.array([full_cvlls[x]]).mean(0) for x in full_cvlls}

for i, label in enumerate(labels):
    plt.plot(full_cvlls[i], label=label)
plt.legend()
plt.savefig("temp.png")
plt.close()

quit()

# Plot histograms
for i, label in enumerate(labels):
    plt.hist(np.exp(cvlls[i]), bins=np.linspace(0.5, 1), alpha=0.3, label=label)

for i, label in enumerate(labels):
    plt.axvline(np.mean(np.exp(cvlls[i])), label=f'{label} mean')

plt.legend()
plt.show()
