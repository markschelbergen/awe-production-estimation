import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from qsm import NormalisedWindTable1D
from cycle_optimizer import OptimizerCycleKappa
from kitev3 import sys_props_v3


n_nbrs = 5
gtol = 1e-4

queue = pd.read_csv('failed2.csv')
queue = queue[queue.id == 'mmij1200902400']

pool = pd.read_csv('succeeded2.csv')

# heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
# col_names_pool = ['id'] + ['vw{0:03.0f}'.format(h) for h in heights] + ['x{:02d}'.format(i) for i in range(34)]
# pool = pool[col_names_pool]


nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(pool.loc[:, 'vw010':'vw600'].values)
distances, neighbor_ids = nbrs.kneighbors(queue.loc[:, 'vw010':'vw600'].values)
i_next_opt = np.argmin(distances[:, 0])

next_opt = queue.iloc[i_next_opt]
vw = list(next_opt['vw010':'vw600'])
env_state = NormalisedWindTable1D()
env_state.normalised_wind_speeds = vw
env_state.set_reference_wind_speed(1.)
axp = env_state.plot_wind_profile()

print("Profile:", next_opt['id'])
print("Mean wind speed: {:.1f} m/s".format(np.mean(vw)))
oc = OptimizerCycleKappa(sys_props_v3, env_state)
n_x = len(oc.x0_real_scale)
ax = plt.subplots(3, (n_x+1)//3+1)[1].reshape(-1)

x_sol = np.empty((n_x, n_nbrs))
x_sol[:] = np.nan
h_sol = np.empty((oc.N_EQ_CONS, n_nbrs))
h_sol[:] = np.nan
g_sol = np.empty((oc.N_INEQ_CONS, n_nbrs))
g_sol[:] = np.nan
mcps = []
for i_attempt in range(n_nbrs):
    starting_point = pool.iloc[neighbor_ids[i_next_opt, i_attempt]]
    print("Attempt {}: starting from {}.".format(i_attempt+1, starting_point['id']))
    axp.plot(starting_point['vw010':'vw600'], env_state.heights, '--')

    x0 = starting_point['x00':'x33']
    oc.x0_real_scale = x0.values
    try:
        oc.optimize(maxiter=300)
        x_sol[:, i_attempt] = oc.x_opt_real_scale
        if oc.op_res['success'] or oc.op_res['exit_mode'] == 9:
            cycle_res = oc.eval_point()
            mcps.append(cycle_res['mean_cycle_power'])
            cons = oc.eval_point(relax_errors=True)[1]
            eq_cons = cons[:oc.N_EQ_CONS]
            h_sol[:, i_attempt] = eq_cons
            ineq_cons = cons[oc.N_EQ_CONS:]
            g_sol[:, i_attempt] = ineq_cons
            if np.max(np.abs(eq_cons)) < gtol and np.min(ineq_cons) > -gtol:
                print("Optimization succeeded")
            else:
                print("CONSTRAINTS VIOLATED")
            print("- Max. abs. equality constraints:", np.max(np.abs(eq_cons)))
            print("- Min. inequality constraint:", np.min(ineq_cons))
        else:
            mcps.append(np.nan)
            print("Optimization failed")
    except Exception as e:
        mcps.append(np.nan)
        print(e)

ax[0].bar(range(n_nbrs), mcps, color='C1')
ax[0].set_ylabel('Mean cycle power [W]')
for i, xs in enumerate(x_sol):
    ax[i+1].bar(range(n_nbrs), xs)
    ax[i+1].set_ylabel(oc.OPT_VARIABLE_LABELS[i])

ax = plt.subplots(3, oc.N_EQ_CONS//3+1)[1].reshape(-1)
for i, hs in enumerate(h_sol):
    ax[i].bar(range(n_nbrs), hs, color=np.where(np.abs(hs) > gtol, 'C3', 'C0'))
    ax[i].set_ylabel(oc.EQ_CONS_LABELS[i])

ax = plt.subplots(4, oc.N_INEQ_CONS//4+1)[1].reshape(-1)
for i, gs in enumerate(g_sol):
    ax[i].bar(range(n_nbrs), gs, color=np.where(gs < -gtol, 'C3', 'C0'))
    ax[i].set_ylabel(oc.INEQ_CONS_LABELS[i])

plt.show()