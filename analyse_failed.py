import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


plot_for_each_cluster = False
heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
n_clusters = 50
# queue = pd.read_csv('failed_mmij2008_1.csv')
# n_records = queue.shape[0]

if plot_for_each_cluster:
    ax = plt.subplots(2, 2, sharex=True)[1].reshape(-1)

li_f, li_s = [], []
for i in range(4):
    s = pd.read_csv('succeeded{}.csv'.format(i))
    if plot_for_each_cluster:
        n_records = s.shape[0]
        for j in range(n_records):
            ax[i].plot(s.iloc[j]['vw010':'vw600'], heights, alpha=.01, color='C2')
    li_s.append(s)

    s = pd.read_csv('succeeded{}_att2.csv'.format(i))
    if plot_for_each_cluster:
        n_records = s.shape[0]
        for j in range(n_records):
            ax[i].plot(s.iloc[j]['vw010':'vw600'], heights, alpha=.01, color='C2')
    li_s.append(s)

    f = pd.read_csv('failed{}_att2.csv'.format(i))
    if plot_for_each_cluster:
        n_records = f.shape[0]
        for j in range(n_records):
            ax[i].plot(f.iloc[j]['vw010':'vw600'], heights, alpha=.02, color='C3')
    li_f.append(f)

plt.figure()

succeeded = pd.concat(li_s, axis=0, ignore_index=True)
n_records = succeeded.shape[0]
for i in range(n_records):
    plt.plot(succeeded.iloc[i]['vw010':'vw600'], heights, alpha=.01, color='C2')

failed = pd.concat(li_f, axis=0, ignore_index=True)
n_records = failed.shape[0]
for i in range(n_records):
    plt.plot(failed.iloc[i]['vw010':'vw600'], heights, alpha=.02, color='C3')

for i in range(3):
    for k in ['vw100', 'vw200', 'vw300']:
        idx_cut_out = failed[failed[k] > 15][k].idxmin()
        plt.plot(failed.loc[idx_cut_out, 'vw010':'vw600'], heights)
        failed.drop(idx_cut_out, inplace=True)
        idx_cut_in = failed[failed[k] < 15][k].idxmax()
        plt.plot(failed.loc[idx_cut_in, 'vw010':'vw600'], heights)
        failed.drop(idx_cut_in, inplace=True)

# cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(queue.loc[:, 'vw010':'vw600'])

# # Determine how much samples belong to each cluster.
# freq = np.zeros(n_clusters)
# for l in cluster_model.labels_:  # Labels: Index of the cluster each sample belongs to.
#     freq[l] += 100. / n_records

# for i in range(n_clusters):
#     plt.plot(cluster_model.cluster_centers_[i, :], heights)


def plot_2d_histograms(data, power_curve, ax=None, height=100):
    h, aoa_edges, cl_edges = np.histogram2d(data['vw{0:03.0f}'.format(height)], data['mcp']*1e-3, [np.linspace(0, 26, 50), 50])[:3] #, [bins['vw'], bins['mcp']]
    x, y = np.meshgrid(aoa_edges, cl_edges)
    h_masked = np.ma.masked_where(h == 0, h)
    ax.pcolormesh(x, y, h_masked.T)  #, vmax=vmax)

    ax.plot(power_curve['vw{0:03.0f}'.format(height)], power_curve['mcp']*1e-3, color='C1')

    ax.set_ylabel('Mean cycle power [kW]')
    ax.set_xlabel('$v_{{w,{0:03.0f}m}}$ [m/s]'.format(height))
    ax.set_xlim([0, None])

df_power_curve = pd.read_csv('opt_res_log_profile.csv')

fig, ax = plt.subplots(3, 1, sharex=True)
plot_2d_histograms(succeeded, df_power_curve, ax[0], height=150)
plot_2d_histograms(succeeded, df_power_curve, ax[1], height=200)
plot_2d_histograms(succeeded, df_power_curve, ax[2], height=300)


plt.show()



