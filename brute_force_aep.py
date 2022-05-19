import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from qsm import NormalisedWindTable1D
from cycle_optimizer import OptimizerCycleKappa
from kitev3_10mm_tether import sys_props_v3
from multiprocessing import Process, cpu_count


expand_pool = True
gtol = 1e-4


def divide_data_in_batches(n_clusters=4):
    data = pd.read_csv('wind_profiles_mmij_2008.csv')
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(data.loc[:, 'vw010':'vw600'])
    for i in range(n_clusters):
        mask = cluster_model.labels_ == i
        print("{} records go in cluster {}".format(np.sum(mask), i))
        data[mask].to_csv('wind_profiles_mmij_2008_cluster{}.csv'.format(i), index=False)


def xopt2series(x_opt):
    record = {}
    for i, xi in enumerate(x_opt):
        record['x{:02d}'.format(i)] = x_opt[i]
    return pd.Series(record)


def perform_opts(i_cluster):
    queue = pd.read_csv('wind_profiles_mmij_2008_cluster{}.csv'.format(i_cluster))
    n_records = queue.shape[0]

    log = pd.read_csv('opt_res_log_profile.csv').loc[:, 'vw010':'x33']
    log.insert(0, 'id', np.mean(log.loc[:, 'vw010':'vw600'], axis=1).apply(lambda vw: 'log{:011.1f}'.format(vw)))
    llj = pd.read_csv('opt_res_llj_profile.csv').loc[:, 'vw010':'x33']
    llj.insert(0, 'id', np.mean(llj.loc[:, 'vw010':'vw600'], axis=1).apply(lambda vw: 'llj{:011.1f}'.format(vw)))
    pool = pd.concat([log, llj])

    col_names = list(pool) + ['mcp', 'start_id', 'exit_mode']
    col_names.insert(1, 'time')
    succeeded = pd.DataFrame(columns=col_names)

    col_names.remove('mcp')
    col_names.remove('exit_mode')
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
        print()
        print("Mean wind speed: {:.1f} - starting from {}.".format(np.mean(vw), starting_point['id']))

        env_state = NormalisedWindTable1D()
        env_state.normalised_wind_speeds = vw
        env_state.set_reference_wind_speed(1.)

        oc = OptimizerCycleKappa(sys_props_v3, env_state)
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
            else:
                success = False
        except Exception as e:
            print(e)
            success = False

        if success:
            success_record = pd.concat([next_opt, xopt2series(oc.x_opt_real_scale)])
            success_record['mcp'] = cycle_res['mean_cycle_power']
            success_record['start_id'] = starting_point['id']
            success_record['exit_mode'] = oc.op_res['exit_mode']
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

    succeeded.to_csv('succeeded{}.csv'.format(i_cluster), index=False)
    failed.to_csv('failed{}.csv'.format(i_cluster), index=False)


def run_all():
    processes = []
    for i in range(4):
        p = Process(target=perform_opts, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    perform_opts(0)
    # divide_data_in_batches()
    # run_all()
    # print(cpu_count())