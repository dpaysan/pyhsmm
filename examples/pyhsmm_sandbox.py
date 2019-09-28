# coding: utf-8

# # PyHSMM Sandbox
# ---

# ---
#
# This notebook serves to provide a sandbox environment to experiment with the pyhsmm package and hence Bayesian inference for Hidden Semi-Markov Models.

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhsmm.basic.distributions import PoissonDuration, NegativeBinomialDuration
from pyhsmm.models import WeakLimitHDPHSMM, HSMM
from pybasicbayes.distributions.gaussian import Gaussian
from pybasicbayes.distributions.multinomial import Categorical
from pybasicbayes.distributions.mixturedistribution import DistributionMixture
import pickle
from pyhsmm.util.eaf_processing import to_eaf
import random

np.random.seed(42)
random.seed(42)

# ---
#
# ## Single Sequence Experiments
#
# First, we will consider a single sequence to perform inference on. This should yield in the frequentist setting the same results as those obtained by using the hsmmlearn package.
#
# ### Data  Input

# We first read in the symbol sequence being our observations. To this end we have stored the symbol sequence for the run 692 as a csv file. Note that the zSymbols refer to the respective combination of the categorical observables as defined in the iterate_v3 config yaml.

# In[3]:

import sys

sys.path.append('/home/daniel/PycharmProjects/virtamed')

"""
the following data files are given in the obs_records dictionary and the 
decoded_df object:

'Data/2018_01_18_13_45_22_022/2018_01_18_13_45_22_022.eaf'
'Data/2018_01_18_12_57_05_692/2018_01_18_12_57_05_692_1.eaf'
'Data/2018_01_18_12_42_56_973/2018_01_18_12_42_56_973.eaf'
'Data/2018_01_18_12_47_42_257/2018_01_18_12_47_42_257.eaf'
'Data/2018_01_18_12_51_23_562/2018_01_18_12_51_23_562.eaf'
'Data/2018_01_18_13_01_17_229/2018_01_18_13_01_17_229.eaf'
"""

pickle_in = open("./data_dict_20092019_150210.pkl", "rb")
data_dict = pickle.load(pickle_in)

obs_list = data_dict["obs"]
records = data_dict["records"]

decode_dfs = data_dict["dec_dfs"]

# ### Definition of initial parameters

# In[7]:


Nmax = None
obs_dim = 4

state_list = ["DX", "place_tool", "cutting_loop", "coag_loop", "clear_view", "handle_chips"]
trans_matrix = [
    [0.000, 0.760, 0.010, 0.010, 0.210, 0.010],  # DX
    [0.010, 0.000, 0.430, 0.430, 0.120, 0.010],  # place tool
    [0.010, 0.450, 0.000, 0.250, 0.250, 0.040],  # cutting loop
    [0.010, 0.530, 0.210, 0.000, 0.200, 0.050],  # coag loop
    [0.010, 0.450, 0.200, 0.250, 0.000, 0.050],  # clear view
    [0.010, 0.060, 0.015, 0.015, 0.900, 0.000]  # handle chips
]

# TODO: add prior for transition matrix functionality in transitions.


# Initial state distribution prior
pi_0 = [0.990, 0.002, 0.002, 0.002, 0.002, 0.002]

# Parameters for the duration distribution (Gamma prior for a Poisson distribution)
alpha_0s = [480, 100.0, 50.0, 100.0, 100, 400.0]
beta_0s = [10, 10, 10, 10, 10, 10]

# # Parameter for the duration distribution (Beta and Gamma prior for the NegativeBinomial distribution)
#
# # Beta prior for p
# alpha_0s = [0.9982, 0.9992, 0.99982, 0.99982, 0.997, 0.9988]
# beta_0s = []
# for alpha_0 in alpha_0s:
#     beta_0s.append(1 - alpha_0)
#
# p_s = np.array([0.9982, 0.9992, 0.99982, 0.99982, 0.97, 0.9988])
# r_s = np.array([0.811, 0.160, 0.0056, 0.0056, 0.09, 0.3604])
#
# # Gamma prior for r
# theta_0s = np.array([0.811, 0.160, 0.09, 0.0056, 0.09, 0.3604])
# k_0s = np.array([1, 1, 1, 1, 1, 1])

# Parameters for the Gaussian emission distribution of the continuous observables
mu_0s = [[-0.00153264, -0.04537947, -0.02625606],
         [0.01677362, -0.05283357, -0.05239786],
         [-0.00229492, 0.02923637, 0.03890725],
         [0.20052574, 0.04940714, -0.07683169],
         [-0.01527346, -0.02078213, 0.02857407],
         [-0.06600041, 0.16879629, 0.05148763]]

sigma_0s = [[[4.72930231e-01, 2.43618309e-02, 6.44149858e-03],
             [2.43618309e-02, 1.59940769e-01, 4.41889207e-02],
             [6.44149858e-03, 4.41889207e-02, 2.99600381e-01]],

            [[1.32391372e-01, 1.68389855e-02, 1.34488169e-02],
             [1.68389855e-02, 1.07246191e-01, 3.01391374e-02],
             [1.34488169e-02, 3.01391374e-02, 1.76619888e-01]],

            [[6.00011833e-02, -1.12897590e-04, -7.41909106e-04],
             [-1.12897590e-04, 5.14267897e-02, -1.24109720e-03],
             [-7.41909106e-04, -1.24109720e-03, 7.34008519e-02]],

            [[1.87182998e-01, 8.10472330e-03, -3.87681125e-02],
             [8.10472330e-03, 5.12437949e-02, -1.92794279e-03],
             [-3.87681125e-02, -1.92794279e-03, 6.75327843e-02]],

            [[2.94168504e-01, -2.66189531e-02, -3.09086863e-03],
             [-2.66189531e-02, 1.38492736e-01, -1.36803256e-02],
             [-3.09086863e-03, -1.36803256e-02, 1.46398883e-01]],

            [[2.68716849e-01, -2.42227143e-01, 1.49313662e-01],
             [-2.42227143e-01, 1.15183681e+00, 7.76968373e-02],
             [1.49313662e-01, 7.76968373e-02, 7.02026769e-01]]]

kappa_0 = 100
nu_0 = obs_dim+3

alpha_0 = 10

weights = [[0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788,
            0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788,
            0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788, 0.00077788,
            0.00077788, 0.00077788, 0.00077788, 0.30247754, 0.67418615],
           [0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00761228,
            0.01091094, 0.00169162, 0.03078745, 0.00169162, 0.00169162, 0.00169162, 0.0017762, 0.00169162, 0.00169162,
            0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.00169162, 0.17787364, 0.21711917, 0.04643491,
            0.07443119, 0.00169162, 0.00169162, 0.00169162, 0.394147],
           [0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.10657194,
            0.12433393, 0.0339698, 0.54795737, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405,
            0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.0044405, 0.02664298, 0.0044405, 0.02397869,
            0.02553286, 0.0044405, 0.0044405, 0.0044405, 0.0044405],
           [0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165,
            0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.0394657, 0.1180935,
            0.00607165, 0.32088646, 0.00607165, 0.00607165, 0.00607165, 0.00607165, 0.04948391, 0.02823315, 0.00637523,
            0.28567092, 0.00607165, 0.00607165, 0.00607165, 0.00607165],
           [0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00415103,
            0.00164904, 0.01063346, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727,
            0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.00113727, 0.6899238, 0.11850335, 0.11332878,
            0.03337882, 0.00113727, 0.00113727, 0.00113727, 0.00113727],
           [0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664,
            0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664,
            0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00407664, 0.00611496, 0.59315124, 0.00407664,
            0.28251121, 0.00407664, 0.00407664, 0.00407664, 0.00407664]]

# After setting the parameters and implementing a MixtureDistribution in the pybasicbayes
# package we can start defining our HSMM.

# In[13]:


dur_distns = []
gaussians = []
categoricals = []
mixtures = []
alpha_0s = np.array(alpha_0s)
beta_0s = np.array(beta_0s)
mu_0s = np.array(mu_0s)
sigma_0s = np.array(sigma_0s)
weights = np.array(weights)

dist_obs_map = [0, 0, 0, 1]

for state in range(len(state_list)):
    dur_distns.append(PoissonDuration(alpha_0=alpha_0s[state], beta_0=beta_0s[state]))
    # dur_distns.append(NegativeBinomialDuration(r=r_s[state], p=p_s[state], k_0=k_0s[state], theta_0=theta_0s[state],
    #                                           alpha_0=alpha_0s[state],
    #                                           beta_0=beta_0s[state]))
    gaussians.append(Gaussian(mu_0=mu_0s[state], sigma_0=sigma_0s[state], kappa_0=kappa_0, nu_0=nu_0))
    categoricals.append(Categorical(weights=weights[state, :], K=weights.shape[1], alpha_0=alpha_0))
    mixtures.append(DistributionMixture(distv=[gaussians[-1], categoricals[-1]], dist_obs_map=dist_obs_map))

distv = [gaussians, categoricals]

tmat = [
    [0.000, 0.760, 0.010, 0.010, 0.210, 0.010],  # DX
    [0.010, 0.000, 0.430, 0.430, 0.120, 0.010],  # place tool
    [0.010, 0.450, 0.000, 0.250, 0.250, 0.040],  # cutting loop
    [0.010, 0.530, 0.210, 0.000, 0.200, 0.050],  # coag loop
    [0.010, 0.450, 0.200, 0.250, 0.000, 0.050],  # clear view
    [0.010, 0.060, 0.015, 0.015, 0.900, 0.000]  # handle chips
]

pi_0 = [0.990, 0.002, 0.002, 0.002, 0.002, 0.002]

posteriormodel = HSMM(obs_distns=mixtures, dur_distns=dur_distns, trans_matrix=tmat,
                      pi_0=pi_0, alpha=50., init_state_concentration=10.)
# posteriormodel.add_data(np.array(obs_list[10]))
# posteriormodel.add_data(np.array(obs_list[11]))
# posteriormodel.add_data(np.array(obs_list[12]))

for i in range(10, len(np.array(obs_list))):
    posteriormodel.add_data(np.array(obs_list[i]))

from pyhsmm.util.text import progprint_xrange

print('Gibbs sampling for initialization')

for idx in progprint_xrange(0):
    posteriormodel.resample_model()

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Initial Fit')
plt.savefig('pyhsmm_init_fit.png')

print('EM')

#likes = posteriormodel.MAP_EM_fit(maxiter=20)
likes = posteriormodel.EM_fit(maxiter=500)
state_seq = posteriormodel.stateseqs[0]
states = ["DX", "place_tool", "cutting_loop", "coag_loop", "clear_view", "handle_chips"]
eaf_file = "/home/daniel/PycharmProjects/virtamed/Data/2018_01_18_12_57_05_692/2018_01_18_12_57_05_692_1.eaf"
to_eaf(state_seq, decode_dfs[10], states, eaf_file, output_dir=".")

state_seq = posteriormodel.stateseqs[1]
eaf_file = "/home/daniel/PycharmProjects/virtamed/Data/2018_01_18_12_42_56_973/2018_01_18_12_42_56_973.eaf"
to_eaf(state_seq, decode_dfs[11], states, eaf_file, output_dir=".")

state_seq = posteriormodel.stateseqs[2]
eaf_file = "/home/daniel/PycharmProjects/virtamed/Data/2018_01_18_12_47_42_257/2018_01_18_12_47_42_257.eaf"
to_eaf(state_seq, decode_dfs[12], states, eaf_file, output_dir=".")

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Fit after BWA')
plt.savefig('pyhsmm_bwa_fit.png')

plt.figure()
plt.title('Log-Likelihood during BWA')
plt.plot(likes)
plt.savefig('pyhsmm_ll_conv.png')

plt.show()
