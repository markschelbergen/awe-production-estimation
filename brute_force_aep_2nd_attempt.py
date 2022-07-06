import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from qsm import NormalisedWindTable1D
from cycle_optimizer import OptimizerCycleKappa
from kitev3 import sys_props_v3

loc = 'mmij'
expand_pool = True
n_nbrs = 5
gtol = 1e-4

li_f, li_s = [], []
for i in range(4):
    s = pd.read_csv('opt_res_{}/succeeded{}.csv'.format(loc, i))
    li_s.append(s)

    f = pd.read_csv('opt_res_{}/failed{}.csv'.format(loc, i))
    li_f.append(f)
pool = pd.concat(li_s, axis=0, ignore_index=True)
queue = pd.concat(li_f, axis=0, ignore_index=True)
n_records = queue.shape[0]

col_names = list(pool) + ['attempt']
succeeded = pd.DataFrame(columns=col_names)

heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
col_names_pool = ['id'] + ['vw{0:03.0f}'.format(h) for h in heights] + ['x{:02d}'.format(i) for i in range(34)]
pool = pool[col_names_pool]

col_names.remove('mcp')
col_names.remove('attempt')
failed_attempts = pd.DataFrame(columns=col_names)

col_names_fa = ['id'] + ['vw{0:03.0f}'.format(h) for h in heights]
failed_altogether = pd.DataFrame(columns=col_names_fa)

success_counter = 0
failed_counter = 0

for i in range(n_records):
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(pool.loc[:, 'vw010':'vw600'].values)
    distances, neighbor_ids = nbrs.kneighbors(queue.loc[:, 'vw010':'vw600'].values)
    i_next_opt = np.argmin(distances[:, 0])

    next_opt = queue.iloc[i_next_opt]
    vw = list(next_opt['vw010':'vw600'])
    vw[0] = 0
    h = [0., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
    env_state = NormalisedWindTable1D(h, vw)

    print()
    print("#"*10, i, "#"*10)
    print("Profile:", next_opt['id'])
    print("Mean wind speed: {:.1f} m/s".format(np.mean(vw)))
    oc = OptimizerCycleKappa(sys_props_v3, env_state)
    for i_attempt in range(n_nbrs):
        starting_point = pool.iloc[neighbor_ids[i_next_opt, i_attempt]]
        print("Attempt {}: starting from {}.".format(i_attempt+1, starting_point['id']))

        x0 = starting_point['x00':'x33']
        oc.x0_real_scale = x0.values
        try:
            oc.optimize(maxiter=300)
            if oc.op_res['success'] or oc.op_res['exit_mode'] == 9:
                cycle_res = oc.eval_point()
                cons = oc.eval_point(relax_errors=True)[1]
                eq_cons = cons[:oc.N_EQ_CONS]
                ineq_cons = cons[oc.N_EQ_CONS:]
                if np.max(np.abs(eq_cons)) < gtol and np.min(ineq_cons) > -gtol:
                    success = True
                else:
                    success = False
                    print("CONSTRAINTS VIOLATED")
                print("- Max. abs. equality constraints:", np.max(np.abs(eq_cons)))
                print("- Min. inequality constraint:", np.min(ineq_cons))
            else:
                success = False
        except Exception as e:
            print(e)
            success = False

        if success:
            success_record = next_opt.copy()
            success_record['x00':'x33'] = oc.x_opt_real_scale
            success_record['start_id'] = starting_point['id']
            success_record['mcp'] = cycle_res['mean_cycle_power']
            success_record['attempt'] = i_attempt+1
            succeeded = succeeded.append(success_record, ignore_index=True)
            if expand_pool:
                pool = pool.append(success_record[list(pool)], ignore_index=True)
            break
        else:
            failed_record = next_opt.copy()
            failed_record['x00':'x33'] = x0
            failed_record['start_id'] = starting_point['id']
            failed_attempts = failed_attempts.append(failed_record, ignore_index=True)
            # if oc.op_res is not None and oc.op_res['exit_mode'] == 9:
            #     break

    if success:
        success_counter += 1
    else:
        failed_counter += 1
        failed_record = next_opt[col_names_fa]
        failed_altogether = failed_altogether.append(failed_record, ignore_index=True)

    queue.drop(queue.index[i_next_opt], inplace=True)
    print("{}/{}/{} optimizations succeeded".format(success_counter, i+1, n_records))

succeeded.to_csv('opt_res_{}/succeeded_att2.csv'.format(loc), index=False)
failed_attempts.to_csv('opt_res_{}/failed_attempts.csv'.format(loc), index=False)
failed_altogether.to_csv('opt_res_{}/failed_att2.csv'.format(loc), index=False)

