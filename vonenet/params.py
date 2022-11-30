import numpy as np
from .utils import sample_dist
import scipy.stats as stats


def generate_gabor_param(features, seed=0, rand_flag=False, sf_corr=0, sf_max=9, sf_min=0):
    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        print('Uniform gabor parameters')
        ori_bins1 = np.array([0, 180])
        ori_dist1 = np.array([1])
        ori_bins2 = np.array([0, 180])
        ori_dist2 = np.array([1])
        ori_bins3 = np.array([0, 180])
        ori_dist3 = np.array([1])
        ori_bins4 = np.array([0, 180])
        ori_dist4 = np.array([1])

        nx_bins = np.array([0.1, 10 ** 0.2])
        nx_dist = np.array([1])

        ny_bins = np.array([0.1, 10 ** 0.2])
        ny_dist = np.array([1])

        # sf_bins = np.array([0.5, 8])
        # sf_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        sfmax_ind = np.where(sf_bins < sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind + 1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()
    else:
        print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins1 = np.array([-22.5, 22.5])
        ori_dist1 = np.array([50])
        ori_dist1 = ori_dist1 / ori_dist1.sum()

        ori_bins2 = np.array([67.5, 90])
        ori_dist2 = np.array([49])
        ori_dist2 = ori_dist2 / ori_dist2.sum()

        ori_bins3 = np.array([112.5, 135])
        ori_dist3 = np.array([77])
        ori_dist3 = ori_dist3 / ori_dist3.sum()

        ori_bins4 = np.array([157.5, 180])
        ori_dist4 = np.array([54])
        ori_dist4 = ori_dist4 / ori_dist4.sum()

        # Schiller 1976
        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([[2., 0., 1., 0., 0.],
                                 [8., 9., 4., 1., 0.],
                                 [1., 2., 19., 17., 3.],
                                 [0., 0., 1., 7., 4.],
                                 [0., 0., 0., 0., 0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([4, 4, 8, 25, 32, 26, 28, 12])

        sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind + 1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()

    phase = sample_dist(phase_dist, phase_bins, features)
    ori_theta1 = sample_dist(ori_dist1, ori_bins1, features)
    ori_theta2 = sample_dist(ori_dist2, ori_bins2, features)
    ori_theta3 = sample_dist(ori_dist3, ori_bins3, features)
    ori_theta4 = sample_dist(ori_dist4, ori_bins4, features)
    ori_theta1[ori_theta1 < 0] = ori_theta1[ori_theta1 < 0] + 180
    ori_theta2[ori_theta2 < 0] = ori_theta2[ori_theta2 < 0] + 180
    ori_theta3[ori_theta3 < 0] = ori_theta3[ori_theta3 < 0] + 180
    ori_theta4[ori_theta4 < 0] = ori_theta4[ori_theta4 < 0] + 180

    if rand_flag:
        sf = sample_dist(sf_dist, sf_bins, features, scale='log2')
        nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
        ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
    else:

        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:, 0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10 ** nx

        ny_samp = np.random.rand(features)
        ny = np.zeros(features)
        for samp_ind, nx_samp in enumerate(nx):
            bin_id = np.argwhere(nx_bins < nx_samp)[-1]
            ny[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                     np.log10(ny_bins))
        ny = 10 ** ny

        sf = np.interp(samps_cdf[:, 1], np.hstack(([0], sf_dist.cumsum())), np.log2(sf_bins))
        sf = 2 ** sf

    return sf, ori_theta1, ori_theta2, ori_theta3, ori_theta4, phase, nx, ny
