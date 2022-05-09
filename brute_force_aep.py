import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from qsm import NormalisedWindTable1D
from cycle_optimizer import OptimizerCycleKappa
from kitev3_10mm_tether import sys_props_v3


expand_pool = True


def xopt2series(x_opt):
    record = {}
    for i, xi in enumerate(x_opt):
        record['x{:02d}'.format(i)] = x_opt[i]
    return pd.Series(record)


queue = pd.read_csv('wind_profiles_mmij_2008.csv')
n_records = queue.shape[0]

log = pd.read_csv('opt_res_log_profile.csv')
log.insert(0, 'id', np.mean(log.loc[:, 'vw010':'vw600'], axis=1).apply(lambda vw: 'log{:011.1f}'.format(vw)))
llj = pd.read_csv('opt_res_llj_profile.csv')
llj.insert(0, 'id', np.mean(llj.loc[:, 'vw010':'vw600'], axis=1).apply(lambda vw: 'llj{:011.1f}'.format(vw)))
pool = pd.concat([log, llj])

col_names = list(pool) + ['mcp', 'start_id']
col_names.insert(1, 'time')
succeeded = pd.DataFrame(columns=col_names)

col_names.remove('mcp')
failed = pd.DataFrame(columns=col_names)

success_counter = 0
failed_counter = 0
for i in range(n_records):
    nbrs = NearestNeighbors(n_neighbors=1).fit(pool.loc[:, 'vw010':'vw600'].values)
    distances, neighbor_ids = nbrs.kneighbors(queue.loc[:, 'vw010':'vw600'].values)
    i_next_opt = np.argmin(distances[:, 0])
    starting_point = pool.iloc[neighbor_ids[i_next_opt, 0]]

    next_opt = queue.iloc[i_next_opt]
    vw = list(next_opt['vw010':'vw600'])
    print("Mean wind speed: {:.1f} - starting from {}.".format(np.mean(vw), starting_point['id']))

    env_state = NormalisedWindTable1D()
    env_state.normalised_wind_speeds = vw
    env_state.set_reference_wind_speed(1.)

    oc = OptimizerCycleKappa(sys_props_v3, env_state)
    x0 = starting_point['x00':'x33']
    oc.x0_real_scale = x0.values
    try:
        oc.optimize(maxiter=300)
        if oc.op_res['success']:
            cycle_res = oc.eval_point()
        success = oc.op_res['success']
    except Exception as e:
        print(e)
        success = False

    if success:
        success_record = pd.concat([next_opt, xopt2series(oc.x_opt_real_scale)])
        success_record['mcp'] = cycle_res['mean_cycle_power']
        success_record['start_id'] = starting_point['id']
        success_counter += 1
        succeeded = succeeded.append(success_record, ignore_index=True)
        if expand_pool:
            pool = pool.append(success_record[list(pool)], ignore_index=True)
    else:
        failed_record = pd.concat([next_opt, x0])
        failed_record['start_id'] = starting_point['id']
        failed = failed.append(failed_record, ignore_index=True)

        # env_state.plot_wind_profile()
        # plt.plot(starting_point['vw010':'vw600'], env_state.heights, '--', color='C{}'.format(failed_counter%10))
        # plt.show()

        failed_counter += 1

    queue.drop(queue.index[i_next_opt], inplace=True)
    print("{}/{} optimizations succeeded".format(success_counter, i+1))

    # if failed_counter == 1:
    #     break

succeeded.to_csv('succeeded.csv', index=False)
failed.to_csv('failed.csv', index=False)


