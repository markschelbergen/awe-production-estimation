import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import add_panel_labels

plot_for_each_cluster = False
heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
n_clusters = 50
# queue = pd.read_csv('failed_mmij2008_1.csv')
# n_records = queue.shape[0]

if plot_for_each_cluster:
    ax = plt.subplots(2, 3, sharex=True)[1].reshape(-1)

li_s = []
for loc in ['mmca']:  #, 'mmij']:
    for i in range(4):
        s = pd.read_csv('opt_res_{}/succeeded{}.csv'.format(loc, i))
        if plot_for_each_cluster:
            n_records = s.shape[0]
            for j in range(n_records):
                ax[i].plot(s.iloc[j]['vw010':'vw600'], heights, alpha=.01, color='C2')
        li_s.append(s)

loc = 'mmca'

s = pd.read_csv('opt_res_{}/succeeded_att2.csv'.format(loc))
if plot_for_each_cluster:
    n_records = s.shape[0]
    for j in range(n_records):
        ax[4].plot(s.iloc[j]['vw010':'vw600'], heights, alpha=.01, color='C2')
li_s.append(s)
succeeded = pd.concat(li_s, axis=0, ignore_index=True)

li_f = []
# for i in range(4):
#     f = pd.read_csv('opt_res_mmij/failed{}.csv'.format(i))
#     li_f.append(f)
li_f.append(pd.read_csv('opt_res_{}/failed_att2.csv'.format(loc)))

failed = pd.concat(li_f, axis=0, ignore_index=True)

if plot_for_each_cluster:
    n_records = failed.shape[0]
    for j in range(n_records):
        ax[4].plot(failed.iloc[j]['vw010':'vw600'], heights, alpha=.02, color='C3')

plt.figure()

n_records = succeeded.shape[0]
for i in range(n_records):
    plt.plot(succeeded.iloc[i]['vw010':'vw600'], heights, alpha=.01, color='C2')

n_records = failed.shape[0]
for i in range(n_records):
    plt.plot(failed.iloc[i]['vw010':'vw600'], heights, alpha=.02, color='C3')

print("No. of wind profiles above cut-out: {}".format((failed['vw200'] > 15).sum()))

for i in range(3):
    for k in ['vw100', 'vw200', 'vw300']:
        try:
            idx_cut_out = failed[failed[k] > 15][k].idxmin()
            plt.plot(failed.loc[idx_cut_out, 'vw010':'vw600'], heights)
            failed.drop(idx_cut_out, inplace=True)
        except ValueError:
            pass
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


def plot_2d_distr(data, power_curve, ax=None, height=100):
    # h, aoa_edges, cl_edges = np.histogram2d(data['vw{0:03.0f}'.format(height)], data['mcp']*1e-3, [np.linspace(0, 26, 50), 50])[:3] #, [bins['vw'], bins['mcp']]
    # x, y = np.meshgrid(aoa_edges, cl_edges)
    # h_masked = np.ma.masked_where(h == 0, h)
    # ax.pcolormesh(x, y, h_masked.T)  #, vmax=vmax)

    speed_bin_edges = np.arange(0, 30, 1)
    speed_bin_center = (speed_bin_edges[1:] + speed_bin_edges[:-1]) / 2
    data['speed_bin'] = np.digitize(data['vw{0:03.0f}'.format(height)], speed_bin_center)

    perc5 = []
    perc25 = []
    perc75 = []
    perc95 = []
    mean = []
    for i_bin in range(len(speed_bin_center)):
        mcps_in_bin = data.loc[data['speed_bin'] == i_bin, "mcp"].values
        if mcps_in_bin.shape[0] > 10:
            p5, p25, p75, p95 = np.percentile(mcps_in_bin, [5, 25, 75, 95])
            perc5.append(p5)
            perc25.append(p25)
            perc75.append(p75)
            perc95.append(p95)
        else:
            perc5.append(np.nan)
            perc25.append(np.nan)
            perc75.append(np.nan)
            perc95.append(np.nan)
        if mcps_in_bin.shape[0] > 0:
            mean.append(np.mean(mcps_in_bin))
        else:
            mean.append(np.nan)

    # ax.fill_between(speed_bin_center, np.array(perc25)*1e-3, np.array(perc75)*1e-3, alpha=0.5, color='C0', label='25-75th percentile')
    ax.fill_between(speed_bin_center, np.array(perc5)*1e-3, np.array(perc95)*1e-3, alpha=0.3, color='C0', label='5-95th percentile')
    # ax.fill_between(speed_bin_center, np.array(perc75)*1e-3, np.array(perc95)*1e-3, alpha=0.3, color='C4')
    ax.plot(speed_bin_center, np.array(mean)*1e-3, color='C0', label='Mean')

    ax.plot(power_curve['vw{0:03.0f}'.format(height)], power_curve['mcp']*1e-3, color='k', label='Log')

    ax.set_ylabel('Mean cycle power [kW]')
    ax.set_xlabel('$v_{{w,{0:03.0f}m}}$ [m/s]'.format(height))

df_power_curve = pd.read_csv('opt_res_{}/opt_res_{}1.csv'.format(loc, loc))

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[5, 4.5])
plt.subplots_adjust(left=.171, bottom=.121, right=.983, top=.87, hspace=.274)
ax[0].set_ylim([0, 20])
ax[0].set_xlim([0, 27])
# plot_2d_histograms(succeeded, df_power_curve, ax[0], height=150)
plot_2d_distr(succeeded, df_power_curve, ax[0], height=200)
plot_2d_distr(succeeded, df_power_curve, ax[1], height=300)
ax[0].legend(bbox_to_anchor=(.15, 1.03, .7, .2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
for a in ax: a.grid()
add_panel_labels(ax, .2)

plt.show()



