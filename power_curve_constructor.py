import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy
from math import pi
import pickle
from itertools import cycle

from qsm import SteadyStateError, OperationalLimitViolation, PhaseError
from utils import flatten_dict
from cycle_optimizer import OptimizerCycle3
from kitepower_kites import sys_props_v2
from qsm import LogProfile, Cycle, Environment, KiteKinematics, SteadyState

from cycle_optimizer import OptimizerCycle3
from kitepower_kites import sys_props_v3
from qsm import LogProfile, Cycle

def plot_optimal_trajectories(kite_positions_dict, title=None):
    # plot trajectory bounds
    phi = np.linspace(0, 2*pi/3, 40)
    # minimum tether length (at end of reel-in phase) circle
    r_circle = 200
    x_circle = np.cos(phi) * r_circle
    z_circle = np.sin(phi) * r_circle
    plt.plot(x_circle, z_circle, 'k--', linewidth=1)

    # plot lower bound elevation
    x_elev_lb = np.linspace(0, 400, 40)
    z_elev_lb = np.tan(25*pi/180)*x_elev_lb
    plt.plot(x_elev_lb, z_elev_lb, 'k--', linewidth=1)

    for v, kite_positions in kite_positions_dict.items():
        # Plot x vs. y of trajectory.
        x_traj = [kp.x for kp in kite_positions['trajectory']]
        try:
            z_traj = [kp.z for kp in kite_positions['trajectory']]
        except AttributeError:
            z_traj = [np.sin(kp.elevation_angle)*kp.straight_tether_length for kp in kite_positions['trajectory']]
        plt.plot(x_traj, z_traj, label=kite_positions['label'])
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.grid()
    plt.gca().set_aspect('equal')
    if title:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


class PowerCurveConstructor:
    def __init__(self, wind_speeds):
        self.wind_speeds = wind_speeds

        self.x_opts = []
        self.x0 = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []
        self.system_properties = None

        self.opt_variable_labels = []
        self.opt_variable_bounds = []

        self.run_labels = []

    def set_labels_bounds_and_system(self, power_optimizer):
        self.opt_variable_labels = power_optimizer.OPT_VARIABLE_LABELS
        self.opt_variable_bounds = power_optimizer.bounds_real_scale
        self.system_properties = power_optimizer.system_properties

    def run_optimization(self, wind_speed, power_optimizer, x0, sweep_for_wind_speeds=None):
        #TODO: check if necessary to run multiple optimizations using different starting points
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        if x0 is not None:
            print("x0:", x0)
            power_optimizer.x0_real_scale = copy(x0)
        x_opt = power_optimizer.optimize()
        self.x0[-1].append(copy(x0))

        if sweep_for_wind_speeds is not None and sweep_for_wind_speeds[0] <= wind_speed <= sweep_for_wind_speeds[1]:
            print("Sweeping design space")
            power_optimizer.perform_local_sensitivity_analysis()
        print("x_opt:", x_opt)
        self.x_opts[-1].append(x_opt)

        self.optimization_details[-1].append(power_optimizer.op_res)
        try:
            cons, kpis = power_optimizer.eval_point()
            sim_successful = True
        except (SteadyStateError, OperationalLimitViolation, PhaseError) as e:
            print("Error occurred while evaluating the resulting optimal point: {}".format(e))
            cons, kpis = power_optimizer.eval_point(relax_errors=True)
            sim_successful = False
        print("cons:", cons)
        self.constraints[-1].append(cons)
        kpis['sim_successful'] = sim_successful
        self.performance_indicators[-1].append(kpis)
        return x_opt, sim_successful

    def run(self, power_optimizer, x0_start=None, dx0=None, run_label=None, sweep_for_wind_speeds=None):
        if x0_start is not None:
            assert len(x0_start) == len(power_optimizer.x0_real_scale), "Length first starting point is incorrect."
        if dx0 is not None:
            assert len(dx0) == len(power_optimizer.x0_real_scale), "Length starting point derivative is incorrect."

        self.run_labels.append(run_label)
        self.x_opts.append([]), self.x0.append([]), self.optimization_details.append([])
        self.constraints.append([])  #, self.violated_constraint_ids.append([]), self.active_constraint_ids.append([])
        self.performance_indicators.append([])
        x_opt_last = None

        for i, v in enumerate(self.wind_speeds):
            if x_opt_last is None:
                x0_next = x0_start
            elif dx0 is not None:
                x0_next = x_opt_last + dx0*(v - v_last)
            else:
                x0_next = x_opt_last

            print("[{}] Processing v={:.1f}m/s".format(i, v))
            x_opt, sim_successful = self.run_optimization(v, power_optimizer, x0_next, sweep_for_wind_speeds)
            if sim_successful:
                x_opt_last = x_opt
                v_last = v

    def run_predefined_sequence(self, optimization_sequence, x0_start=None, run_label=None, sweep_for_wind_speeds=None):
        self.run_labels.append(run_label)
        self.x_opts.append([]), self.x0.append([]), self.optimization_details.append([])
        self.constraints.append([]), self.performance_indicators.append([])
        x_opt_last = None

        wind_speed_treshold_sequence = iter(sorted(list(optimization_sequence)))
        active_wind_speed_treshold = next(wind_speed_treshold_sequence)

        for i, v in enumerate(self.wind_speeds):
            if v > active_wind_speed_treshold:
                active_wind_speed_treshold = next(wind_speed_treshold_sequence)
            power_optimizer = optimization_sequence[active_wind_speed_treshold]['power_optimizer']
            vary_x0 = optimization_sequence[active_wind_speed_treshold].get('vary_x0', True)
            dx0 = optimization_sequence[active_wind_speed_treshold].get('dx0', None)

            if x0_start is not None:
                assert len(x0_start) == len(power_optimizer.x0_real_scale), "Length starting point is incorrect."
            if dx0 is not None:
                assert len(dx0) == len(power_optimizer.x0_real_scale), "Length starting point derivative is incorrect."

            if not vary_x0 or x_opt_last is None:
                x0_next = x0_start
            elif dx0 is not None:
                if isinstance(vary_x0, list):
                    assert len(vary_x0) == len(power_optimizer.x0_real_scale), "Length vary_x0 is incorrect."
                    x0_next0 = x0_start
                    x0_next1 = x_opt_last + dx0*(v - v_last)
                    x0_next = np.where(vary_x0, x0_next1, x0_next0)
                else:
                    x0_next = x_opt_last + dx0*(v - v_last)
            elif isinstance(vary_x0, list):
                assert len(vary_x0) == len(power_optimizer.x0_real_scale), "Length vary_x0 is incorrect."
                x0_next0 = x0_start
                x0_next1 = x_opt_last
                x0_next = np.where(vary_x0, x0_next1, x0_next0)
            else:
                x0_next = x_opt_last

            print("[{}] Processing v={:.2f}m/s".format(i, v))
            try:
                x_opt, sim_successful = self.run_optimization(v, power_optimizer, x0_next, sweep_for_wind_speeds)
            except (OperationalLimitViolation, SteadyStateError, PhaseError):
                try:
                    x_opt, sim_successful = self.run_optimization(v+1e-2, power_optimizer, x0_next, sweep_for_wind_speeds)
                    self.wind_speeds[i] = v+1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError):
                    self.wind_speeds = self.wind_speeds[:i]
                    print("Wind speeds array reduced, {:.1f} m/s is the max. wind speed analysed.".format(self.wind_speeds[-1]))
                    break

            if sim_successful:
                x_opt_last = x_opt
                v_last = v

    def run_predefined_sequence2(self, optimization_sequence, starting_points, run_label=None, sweep_for_wind_speeds=None):
        self.run_labels.append(run_label)
        self.x_opts.append([]), self.x0.append([]), self.optimization_details.append([])
        self.constraints.append([]), self.performance_indicators.append([])

        wind_speed_treshold_sequence = iter(sorted(list(optimization_sequence)))
        active_wind_speed_treshold = next(wind_speed_treshold_sequence)

        for i, (v, x0) in enumerate(zip(self.wind_speeds, starting_points)):
            if v > active_wind_speed_treshold:
                active_wind_speed_treshold = next(wind_speed_treshold_sequence)
            power_optimizer = optimization_sequence[active_wind_speed_treshold]['power_optimizer']

            print("[{}] Processing v={:.2f}m/s".format(i, v))
            self.run_optimization(v, power_optimizer, x0, sweep_for_wind_speeds)

    def recalc_performance_indicators(self, power_optimizer):  # written for testing reasons
        performance_indicators = []

        for v, x_opt in zip(self.wind_speeds, self.x_opts[-1]):
            if None not in x_opt:
                power_optimizer.environment_state.set_reference_wind_speed(v)
                performance_indicators.append(power_optimizer.eval_point(x_real_scale=x_opt))
        return performance_indicators

    def plot_optimal_trajectories(self, run_id=-1, wind_speed_ids=None, create_fig=True, norm_v=None):
        if create_fig:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            title = self.run_labels[run_id]
        else:
            title = None

        op_res = self.performance_indicators[run_id]
        kite_positions_dict = {}
        if wind_speed_ids is None:
            if len(self.wind_speeds) > 8:
                wind_speed_ids = [int(a) for a in np.linspace(0, len(self.wind_speeds)-1, 6)]
            else:
                wind_speed_ids = range(len(self.wind_speeds))
        for i in wind_speed_ids:
            v = self.wind_speeds[i]
            kpis = op_res[i]
            if kpis is not None:
                kite_positions_dict[v] = {'trajectory': kpis['kinematics']}
                if norm_v is None:
                    kite_positions_dict[v]['label'] = "$v_{100m}$="+"{:.1f} ".format(v) + "m s$^{-1}$"
                else:
                    kite_positions_dict[v]['label'] = "$\hat{v}_{100m}=$"+"{:.1f}".format(v/norm_v)
            else:
                print("Can not plot trajectory for {} m/s wind speed.".format(v))

        plot_optimal_trajectories(kite_positions_dict, title)

    def eval_plots(self, power_optimizer, run_id=-1, wind_speed_id=0):
        power_optimizer.environment_state.set_reference_wind_speed(self.wind_speeds[wind_speed_id])
        power_optimizer.eval_fun(self.x_opts[run_id][wind_speed_id], True, scale_x=False)

    def plot_sensitivity(self, power_optimizer, run_id=-1, wind_speed_id=0, param_id=None):
        power_optimizer.environment_state.set_reference_wind_speed(self.wind_speeds[wind_speed_id])

        power_optimizer.x0_real_scale = self.x_opts[run_id][wind_speed_id]
        tmp = power_optimizer.reduce_x
        if param_id:
            power_optimizer.reduce_x = [param_id]
        else:
            power_optimizer.reduce_x = None
        power_optimizer.perform_local_sensitivity_analysis()
        power_optimizer.reduce_x = tmp

    def plot_efficiency_sensitivity(self, power_optimizer, run_id=-1, wind_speed_id=0, param_id=0):
        power_optimizer.environment_state.set_reference_wind_speed(self.wind_speeds[wind_speed_id])

        power_optimizer.x0_real_scale = self.x_opts[run_id][wind_speed_id]
        tmp = power_optimizer.reduce_x
        power_optimizer.reduce_x = [param_id]
        power_optimizer.sweep_design_space_for_efficiency()
        power_optimizer.reduce_x = tmp

    def plot_optimization_results(self, run_id=-1):
        assert self.x_opts, "No optimization results available for plotting."
        xo, x0 = self.x_opts[run_id], self.x0[run_id]
        cons = self.constraints[run_id]
        kpis, opt_details = self.performance_indicators[run_id], self.optimization_details[run_id]
        try:
            performance_indicators = next(list(flatten_dict(kpi)) for kpi in kpis if kpi is not None)
        except StopIteration:
            performance_indicators = []

        fig, ax = plt.subplots(max([len(xo[0]), 6]), 2, sharex=True)  #, num="overview")
        plt.suptitle(self.run_labels[run_id])

        wind_speed_range = [self.wind_speeds[0], self.wind_speeds[-1]]

        for i in range(len(xo[0])):
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in xo], label='x_opt')
            # Plot starting points.
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in x0], 'o', markerfacecolor='None', label='x0')

            ax[i, 0].grid()
            if self.opt_variable_labels:
                label = self.opt_variable_labels[i]
                ax[i, 0].set_ylabel(label)

                # Plot the actual reeling factors, could differ from the design variables in case of force controlled
                # steady state.
                if label == "Reel-out\nfactor [-]" and "reeling_factor_out" in performance_indicators:
                    ax[i, 0].plot(self.wind_speeds, [kpi['reeling_factor_out'] if kpi else None for kpi in kpis], '.',
                                  label='actual rf')
                elif label == "Reel-in\nfactor [-]" and "reeling_factor_in" in performance_indicators:
                    ax[i, 0].plot(self.wind_speeds, [kpi['reeling_factor_in'] if kpi else None for kpi in kpis], '.')
                elif label == "Elevation angle\nend [rad]":
                    ax[i, 0].plot(self.wind_speeds, [a[i]-a[i-1] for a in x0], ':')
            else:
                ax[i, 0].set_ylabel("x[{}]".format(i))
            if self.opt_variable_bounds:
                ax[i, 0].plot(wind_speed_range, [self.opt_variable_bounds[i][0]]*2, 'k--')
                ax[i, 0].plot(wind_speed_range, [self.opt_variable_bounds[i][1]]*2, 'k--')
        ax[0, 0].legend()

        ax[0, 1].plot(self.wind_speeds, [od['nit'] for od in opt_details])

        opts_failed = list(zip(*[(v, od['nit']) for v, od in zip(self.wind_speeds, opt_details) if not od['success']]))
        if opts_failed:
            v_opts_failed = opts_failed[0]
            ax[0, 1].plot(v_opts_failed, opts_failed[1], 'x', label='opt failed')

        sim_failed = list(zip(*[(v, od['nit']) for v, od, kpi in zip(self.wind_speeds, opt_details, kpis) if not kpi['sim_successful']]))
        if sim_failed:
            ax[0, 1].plot(sim_failed[0], sim_failed[1], 'x', label='sim failed')

        ax[0, 1].grid()
        ax[0, 1].legend()
        ax[0, 1].set_ylabel('Optimization iterations [-]')

        cons_treshold_magenta = -.1
        power = [kpi['average_power']['cycle'] if kpi and kpi['sim_successful'] and all([a >= cons_treshold_magenta for a in c])
                 else None for kpi, c in zip(kpis, cons)]
        ax[1, 1].plot(self.wind_speeds, power)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Mean power [W]')

        # Plot tether force related performance indicators.
        for key in performance_indicators:
            if 'max_tether_force' in key:
                ax[2, 1].plot(self.wind_speeds, [flatten_dict(kpi)[key] if kpi and key in flatten_dict(kpi) else None for kpi in kpis],
                            label=key.replace('tether_force_', ''))
        relevant_limits = ['tether_force_max_limit', 'tether_force_min_limit']
        for rl in relevant_limits:
            lim = getattr(self.system_properties, rl)
            ax[2, 1].plot(wind_speed_range, [lim]*2, 'k--')  # label=rl
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('Tether force [N]')
        ax[2, 1].legend(loc=2)
        ax[2, 1].annotate('Violation occurring before\nswitch to force controlled',
                          xy=(0.05, 0.10), xycoords='axes fraction')

        # Plot reeling speed related performance indicators.
        for key in performance_indicators:
            if 'reeling_speed' in key and 'average' not in key:
                ax[3, 1].plot(self.wind_speeds, [flatten_dict(kpi).get(key, None) if kpi else None for kpi in kpis],
                              label=key.replace('reeling_speed_', ''))
        relevant_limits = ['reeling_speed_min_limit', 'reeling_speed_max_limit']
        for rl in relevant_limits:
            lim = getattr(self.system_properties, rl)
            ax[3, 1].plot(wind_speed_range, [lim]*2, 'k--')  # label=rl
        ax[3, 1].grid()
        ax[3, 1].set_ylabel('Reeling speed [m/s]')
        ax[3, 1].legend(loc=2)

        # Plot constraint matrix.
        try:
            n_cons = len(next(c for c in cons if c is not None))
        except StopIteration:
            n_cons = 0
        cons_flattened = []
        for subcons in cons:
            subcons = subcons.tolist()
            if subcons:
                cons_flattened += subcons
            else:
                cons_flattened += [0]*n_cons

        cons_treshold_magenta = -.1
        cons_treshold_red = -1e-6
        cons_matrix = np.array(cons_flattened).reshape((-1, n_cons)).transpose()
        cons_matrix_mod = np.where(cons_matrix < cons_treshold_magenta, -2, 0)
        cons_matrix_mod = np.where((cons_matrix >= cons_treshold_magenta) & (cons_matrix < cons_treshold_red), -1, cons_matrix_mod)
        cons_matrix_mod = np.where((cons_matrix >= cons_treshold_red) & (cons_matrix < 1e-3), 1, cons_matrix_mod)
        cons_matrix_mod = np.where(cons_matrix == 0., 0, cons_matrix_mod)
        cons_matrix_mod = np.where(cons_matrix >= 1e-3, 2, cons_matrix_mod)

        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im_cons1 = ax[4, 1].matshow(cons_matrix_mod, cmap=cmap, vmin=-2, vmax=2,
                                    extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        ax[4, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[4, 1].set_yticklabels(range(n_cons))
        ax[4, 1].set_ylabel('Constraint id\'s')

        # Set colorbar
        ax_pos = ax[4, 1].get_position()
        h_cbar = ax_pos.y1 - ax_pos.y0
        w_cbar = .012
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = np.arange(-2+4/10., 2., 4/5.)
        cbar_ticks_labels = ['<-.1', '<0', '0', '~0', '>0']
        cbar = fig.colorbar(im_cons1, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        im_cons2_limits = [-.1, .1]
        im_cons2 = ax[5, 1].matshow(cons_matrix, vmin=im_cons2_limits[0], vmax=im_cons2_limits[1], cmap=mpl.cm.YlGnBu_r,
                                    extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        ax[5, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[5, 1].set_yticklabels(range(n_cons))
        ax[5, 1].set_ylabel('Constraint id\'s')

        # Set colorbar
        ax_pos = ax[5, 1].get_position()
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = im_cons2_limits[:]
        cbar_ticks_labels = [str(v) for v in cbar_ticks]
        if im_cons2_limits[0] < np.min(cons_matrix) < im_cons2_limits[1]:
            cbar_ticks.insert(1, np.min(cons_matrix))
            cbar_ticks_labels.insert(1, 'min: {:.2E}'.format(np.min(cons_matrix)))
        if im_cons2_limits[0] < np.max(cons_matrix) < im_cons2_limits[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, 'max: {:.2E}'.format(np.max(cons_matrix)))
        cbar = fig.colorbar(im_cons2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds [m/s]')
        ax[-1, 1].set_xlabel('Wind speeds [m/s]')
        ax[0, 0].set_xlim([wind_speed_range[0], wind_speed_range[1]])

    def comparison_plot(self, plot_vars=None, plot_kpi=None):
        assert self.x_opts, "No optimization results available for plotting."
        if plot_vars is None:
            plot_vars = max([len(xo[0]) for xo in self.x_opts])
        if isinstance(plot_vars, int):
            plot_n_vars = plot_vars
            plot_vars = range(plot_vars)
        else:
            plot_n_vars = len(plot_vars)

        if plot_kpi:
            n_plots = plot_n_vars+3
        else:
            n_plots = plot_n_vars+2

        _, ax = plt.subplots(n_plots, 1, sharex=True)
        color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        line_style_cycler = cycle(["-", "--", "-.", ":"])

        first_iter = True  # case of first iter is used as reference case
        power_curve_ref = [kpi0['average_power']['cycle'] if kpi0 else None for kpi0 in self.performance_indicators[0]]
        for xo, kpis, l in zip(self.x_opts, self.performance_indicators, self.run_labels):
            line_color = next(color_cycler)
            line_style = '.'+next(line_style_cycler)
            # Plot the resulting optimal points.
            for i, var_id in enumerate(plot_vars):
                if '[rad]' in self.opt_variable_labels[var_id]:
                    plot_values = [a[var_id]*180/pi if a[var_id] else a[var_id] for a in xo]
                else:
                    plot_values = [a[var_id] for a in xo]
                ax[i].plot(self.wind_speeds, plot_values, line_style, color=line_color, label=l)
            # Plot the power curves.
            ax[plot_n_vars].plot(self.wind_speeds, [kpi['average_power']['cycle'] if kpi and kpi['sim_successful'] else None for kpi in kpis], line_style, color=line_color)
            if not first_iter:
                diff_power = [kpi['average_power']['cycle']-p_ref if kpi and p_ref else None for kpi, p_ref in zip(kpis, power_curve_ref)]
                ax[plot_n_vars+1].plot(self.wind_speeds, diff_power, line_style, color=line_color)

            if plot_kpi:
                line_style_cycler_kpi = cycle(["-", "--", "-.", ":"])
                plot_kpi_keys = list(kpis[0][plot_kpi])
                for pkk in plot_kpi_keys:
                    kpi_values = [kpi[plot_kpi][pkk] if kpi else None for kpi in kpis]
                    ax[plot_n_vars+2].plot(self.wind_speeds, kpi_values, next(line_style_cycler_kpi), color=line_color)
                if first_iter:
                    ax[plot_n_vars+2].legend(plot_kpi_keys)
            first_iter = False

        for i, var_id in enumerate(plot_vars):
            if self.opt_variable_labels:
                label = self.opt_variable_labels[var_id].replace('[rad]', '[deg]')
                ax[i].set_ylabel(label)
            else:
                ax[i].set_ylabel("x[{}]".format(var_id))
        ax[plot_n_vars].set_ylabel('Mean power [W]')
        ax[plot_n_vars+1].set_ylabel('Mean power\ndifference [W]')
        if plot_kpi:
            ax[plot_n_vars+2].set_ylabel(plot_kpi)

        for a in ax: a.grid()
        ax[0].legend()
        ax[-1].set_xlabel('Wind speed [m/s]')

    def export_results(self, file_name):
        export_dict = self.__dict__
        for k, v in export_dict.items():
            if isinstance(v, np.ndarray):
                export_dict[k] = v.tolist()
        with open(file_name, 'wb') as f:
            pickle.dump(export_dict, f)

    def import_results(self, file_name):
        with open(file_name, 'rb') as f:
            import_dict = pickle.load(f)
        for k, v in import_dict.items():
            setattr(self, k, v)


def cycle_v2_ref():
    # Tried to reconstruct the pumping cycle in the Rolf's paper, however did not succeed. Eventhough the trends are
    # similar
    env_state = LogProfile()
    env_state.wind_speed_ref = 5.9
    env_state.h_ref = 6.
    env_state.h_0 = 0.07
    print("Wind speed at 100 m:", env_state.calculate_wind(100))
    print("Wind speed at 139 m:", env_state.calculate_wind(139))

    # env_state = Environment(10.1, 1.225)

    control_settings = {
        'retraction': ('tether_force_ground', 750.),
        'transition': ('tether_force_ground', 750.),
        'traction': ('tether_force_ground', 3069),
    }

    # kin0 = KiteKinematics(straight_tether_length=385, azimuth_angle=0., elevation_angle=26.6*np.pi/180., course_angle=np.pi)
    # env_state.calculate(kin0.z)
    # sys_props_v2.update(kin0.straight_tether_length, False)
    # ss0 = SteadyState()
    # ss0.control_settings = ('tether_force_ground', 750.)
    # ss0.find_state(sys_props_v2, env_state, kin0, print_details=True)
    # print(ss0.reeling_speed)
    #
    # ss0.control_settings = ('reeling_speed', ss0.reeling_speed)
    # ss0.find_state(sys_props_v2, env_state, kin0, print_details=True)
    # print(ss0.tether_force_ground)

    cycle = Cycle(control_settings)

    cycle.elevation_angle_traction = 26.6*np.pi/180.
    cycle.tether_length_start_retraction = 385
    cycle.tether_length_end_retraction = 234
    cycle.traction_phase.azimuth_angle = 10.6 * np.pi / 180.
    cycle.traction_phase.course_angle = 96.4 * np.pi / 180.
    cycle.transition_phase.kite_powered = False
    cycle.run_simulation(sys_props_v2, env_state)

    cycle.trajectory_plot(steady_state_markers=True)
    print(min([kin.straight_tether_length for kin in cycle.kinematics]))
    print(max([kin.straight_tether_length for kin in cycle.kinematics]))
    cycle.time_plot(['straight_tether_length', 'z'])
    cycle.time_plot(['reeling_speed', 'tether_force_ground'])
    cycle.time_plot(['straight_tether_length', 'tether_force_ground',
                     'kite_speed', 'power_ground', 'apparent_wind_speed', 'wind_speed', 'z'])

    res_ref = {
        'average_power': {
            'cycle': cycle.average_power,
            'in': cycle.retraction_phase.average_power,
            'trans': cycle.transition_phase.average_power,
            'out': cycle.traction_phase.average_power,
        },
        'duration': {
            'cycle': cycle.duration,
            'in': cycle.retraction_phase.duration,
            'trans': cycle.transition_phase.duration,
            'out': cycle.traction_phase.duration,
        },
    }
    print("###")
    for k in ['in', 'trans', 'out', 'cycle']:
        print(k, "Power [kW]:", res_ref['average_power'][k]*1e-3)
        print("Time [s]:", res_ref['duration'][k])
    plt.show()
    oc = OptimizerCycle3(sys_props_v2, env_state, True, False, reduce_x=(0, 1, 2, 3))
    oc.x0_real_scale = [3600., 750., 30.*np.pi/180., 150, 230]
    x_opt = oc.optimize()
    print(x_opt)
    cons, kpis = oc.eval_point()
    print("###")
    for k in ['in', 'trans', 'out', 'cycle']:
        print("Power [kW]:", kpis['average_power'][k]*1e-3)
        print("Time [s]:", kpis['duration'][k])
    oc.eval_point(True)
    plt.show()


def cycle_v3_ref():
    env_state = LogProfile()
    env_state.wind_speed_ref = 10.
    env_state.h_ref = 100.

    theta_out = 25*np.pi/180.
    phi_out = 10 * np.pi / 180.
    chi_out = 100 * np.pi / 180.
    le0 = 200
    le1 = 400
    dle = le1 - le0
    f_in = 750.
    f_out = 3000.
    cycle_settings = {
        'cycle': {
            'tether_length_start_retraction': le1,
            'tether_length_end_retraction': le0,
            'elevation_angle_traction': theta_out,
        },
        'retraction': {
            'control': ('tether_force_ground', f_in),
        },
        'transition': {
            'control': ('reeling_speed', 0., f_in, f_out),
            'time_step': 0.25,
        },
        'traction': {
            'control': ('tether_force_ground', f_out),
            'azimuth_angle': phi_out,
            'course_angle': chi_out,
        },
    }

    cycle = Cycle(cycle_settings)
    cycle.run_simulation(sys_props_v3, env_state)

    cycle.trajectory_plot(steady_state_markers=True)
    print(min([kin.straight_tether_length for kin in cycle.kinematics]))
    print(max([kin.straight_tether_length for kin in cycle.kinematics]))
    cycle.time_plot(['straight_tether_length', 'reeling_speed', 'tether_force_ground'])

    res_ref = {
        'average_power': {
            'cycle': cycle.average_power,
            'in': cycle.retraction_phase.average_power,
            'trans': cycle.transition_phase.average_power,
            'out': cycle.traction_phase.average_power,
        },
        'duration': {
            'cycle': cycle.duration,
            'in': cycle.retraction_phase.duration,
            'trans': cycle.transition_phase.duration,
            'out': cycle.traction_phase.duration,
        },
    }
    print("###")
    for k in ['in', 'trans', 'out', 'cycle']:
        print(k, "Power [kW]:", res_ref['average_power'][k]*1e-3)
        print("Time [s]:", res_ref['duration'][k])

    oc = OptimizerCycle3(cycle_settings, sys_props_v3, env_state, True, False, reduce_x=(0, 1, 2, 3, 4))
    oc.x0_real_scale = [f_out, f_in, theta_out, dle, le0]
    print(oc.x0_real_scale)
    x_opt = oc.optimize()
    print(x_opt)
    cons, kpis = oc.eval_point()
    print("###")
    for k in ['in', 'trans', 'out', 'cycle']:
        print("Power [kW]:", kpis['average_power'][k]*1e-3)
        print("Time [s]:", kpis['duration'][k])
    oc.eval_point(True)
    plt.show()


def power_curve_v3():
    from qsm import NormalisedWindTable1D, SteadyState, KiteKinematics, Cycle, LogProfile, TractionPhasePattern, TractionPhaseHybrid
    from cycle_optimizer import OptimizerCycle3
    from kitepower_kites import sys_props_v3
    import pandas as pd

    log_profile = True
    if not log_profile:
        df = pd.read_csv('wind_resource/profile1.csv', sep=";")
        env_state = NormalisedWindTable1D()
        env_state.heights = list(df['h [m]'])
        env_state.normalised_wind_speeds = list((df['u1 [-]']**2 + df['v1 [-]']**2)**.5)
    else:
        env_state = LogProfile()
    env_state.set_reference_wind_speed(9.)

    theta_out = 25*np.pi/180.
    #TODO: decide on angles
    phi_out = 13 * np.pi / 180.
    chi_out = 100 * np.pi / 180.
    le0 = 200
    le1 = 400
    dle = le1 - le0
    f_in = 200.
    f_out = 1500
    cycle_settings = {
        'cycle': {
            'tether_length_start_retraction': le1,
            'tether_length_end_retraction': le0,
            'elevation_angle_traction': theta_out,
            'traction_phase': TractionPhaseHybrid,
        },
        'retraction': {
            # 'control': ('tether_force_ground', f_in),
        },
        'transition': {
            # 'control': ('reeling_speed', 0., f_in, f_out),
            'time_step': 0.25,
        },
        'traction': {
            # 'control': ('tether_force_ground', f_out),
            'azimuth_angle': phi_out,
            'course_angle': chi_out,
        },
    }

    op_cycle01 = OptimizerCycle3(cycle_settings, sys_props_v3, env_state, True, False, reduce_x=(0, 1,))
    op_cycle0123 = OptimizerCycle3(cycle_settings, sys_props_v3, env_state, True, False, reduce_x=(0, 1, 2, 3))
    construct_power_curve = True
    if False:
        # v = 4
        # x0 = [1171.3099564866843, 300.00000000023783, 0.4363323129985824, 200.0, 200.0]
        # while True:
        #     print("Processing v={:.2f}m/s".format(v))
        #     op_cycle01.environment_state.set_reference_wind_speed(v)
        #     op_cycle01.x0_real_scale = copy(x0)
        #     x_opt = op_cycle01.optimize()
        #     try:
        #         cons, kpis = op_cycle01.eval_point()
        #         print("x_opt:", x_opt)
        #     except:
        #         print("Did not converge to feasible solution for latter wind speed.")
        #         break
        #     x0 = x_opt
        #     v -= .25

        # v = 23
        # x0 = [4999.999999999997, 2819.640575754534, 0.9055261203127993, 200.0, 200.0]
        # while True:
        #     print("Processing v={:.2f}m/s".format(v))
        #     op_cycle012.environment_state.set_reference_wind_speed(v)
        #     op_cycle012.x0_real_scale = copy(x0)
        #     x_opt = op_cycle012.optimize()
        #     try:
        #         cons, kpis = op_cycle012.eval_point()
        #         print("x_opt:", x_opt)
        #     except:
        #         print("Did not converge to feasible solution for latter wind speed.")
        #         break
        #     x0 = x_opt
        #     v += .25

        v = 19
        x0 = [4999.999999999995, 1651.5847896791208, 0.992842320539557, 200.0, 200.0]
        print("Processing v={:.2f}m/s".format(v))
        op_cycle012.environment_state.set_reference_wind_speed(v)
        op_cycle012.x0_real_scale = copy(x0)
        x_opt = op_cycle012.optimize()
        print(x_opt)
        op_cycle012.perform_local_sensitivity_analysis()
    elif construct_power_curve:
        if not log_profile:
            v_cut_out = 25
        else:
            v_cut_out = 24
        v_cut_out = 20
        wind_speeds = np.linspace(6.1, 18.5, 13)
        x0 = [f_out, f_in, theta_out, dle, le0]
        x0 = [2500, 300., 0.45, 200.0, 200.0]

        pc = PowerCurveConstructor(wind_speeds)
        pc.set_labels_bounds_and_system(op_cycle0123)

        # Full design variables for all wind speeds.
        op_seq_cycle3 = {
            13: {'power_optimizer': op_cycle0123, 'dx0': np.array([0., 0., 0., 0., 0.])},
            np.inf: {'power_optimizer': op_cycle0123, 'dx0': np.array([0., 0., 0., 0., 0.])},  #np.array([0., 0., 0.07, 0., 0.]
        }
        pc.run_predefined_sequence(op_seq_cycle3, x0, run_label='full cycle')

        pc.plot_optimization_results()
        pc.plot_optimal_trajectories()

        plt.figure()
        ncwp = [kpis['n_crosswind_patterns'] for kpis in pc.performance_indicators[-1]]
        f_out_min = [kpis['min_tether_force']['out'] for kpis in pc.performance_indicators[-1]]

        for x in pc.x_opts[-1]:
            print("[{:.0f}, {:.0f}, {:.2f}, {:.1f}, {:.1f}],".format(*x))

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(wind_speeds, ncwp)
        ax[1].plot(wind_speeds, f_out_min)


        # x0 = [5000.000000000001, 4999.529000766381, 0.8198555052734466, 200.0, 200.0]
        # wind_speeds = np.arange(18, 14, -1)
        #
        # pc = PowerCurveConstructor(wind_speeds)
        # pc.set_labels_bounds_and_system(op_cycle01)
        #
        # # Full design variables for all wind speeds.
        # op_seq_cycle3 = {
        #     np.inf: {'power_optimizer': op_cycle012, 'dx0': np.array([0., 0., 0., 0., 0.])},
        # }
        # pc.run_predefined_sequence(op_seq_cycle3, x0, run_label='full cycle')
        #
        # pc.plot_optimization_results()
        # pc.plot_optimal_trajectories()

        # wind_speeds = np.arange(13, 26)
        # x0 = [5000.000000000001, 1385.7964349337492, 0.45, 200.0, 200.0]
        #
        # pc = PowerCurveConstructor(wind_speeds)
        # pc.set_labels_bounds_and_system(op_cycle01)
        #
        # # Full design variables for all wind speeds.
        # op_seq_cycle3 = {
        #     13: {'power_optimizer': op_cycle01, 'dx0': np.array([0., 0., 0., 0., 0.])},
        #     14: {'power_optimizer': op_cycle012, 'dx0': np.array([0., 0., 0., 0., 0.])},
        #     15: {'power_optimizer': op_cycle012, 'dx0': np.array([0., 0., 0.25, 0., 0.])},
        #     np.inf: {'power_optimizer': op_cycle012, 'dx0': np.array([0., 0., 0.09, 0., 0.])},
        # }
        # pc.run_predefined_sequence(op_seq_cycle3, x0, run_label='full cycle')
        #
        # pc.plot_optimization_results()
        # pc.plot_optimal_trajectories()

        # v = 19
        # x_opt = [4999.999999999995, 2432.532418961368, 0.7495600180133208, 200.0, 200.0]
        # cwps = []
        # f_in = np.linspace(300, 5000, 50)
        # for f in f_in:  #zip([18], [[4999.999999999999, 2327.9375535001113, 0.7081265737275835, 200.0, 200.0]]):
        #     env_state.set_reference_wind_speed(v)
        #     cycle_settings = {
        #         'cycle': {
        #             'elevation_angle_traction': x_opt[2],
        #             'tether_length_start_retraction': x_opt[3]+x_opt[4],
        #             'tether_length_end_retraction': x_opt[4],
        #             'traction_phase': TractionPhaseHybrid,
        #         },
        #         'retraction': {
        #             'control': ('tether_force_ground', f),
        #         },
        #         'transition': {
        #             'control': ('reeling_speed', 0., f, x_opt[0]),
        #             'time_step': 0.25,
        #         },
        #         'traction': {
        #             'control': ('tether_force_ground', x_opt[0]),
        #             'azimuth_angle': phi_out,
        #             'course_angle': chi_out,
        #         },
        #     }
        #     cycle = Cycle(cycle_settings)
        #     cycle.run_simulation(sys_props_v3, env_state, {'enable_steady_state_errors': False})
        #     cwps.append(cycle.traction_phase.n_crosswind_patterns)
        #     # cycle.time_plot(('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground'),
        #     #     ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]'))
        #     # cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
        #     #                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
        #     #                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
        #     # cycle.time_plot(('straight_tether_length', 'reeling_speed', 'x', 'y', 'z'),
        #     #                 ('r [m]', r'$\dot{\mathrm{r}}$ [m/s]', 'x [m]', 'y [m]', 'z [m]'))
        #     # cycle.trajectory_plot3d()
        #     # print(cycle.retraction_phase.energy)
        #
        # plt.figure()
        # plt.plot((f_in), cwps)  #*180./np.pi

        # kpis = pc.performance_indicators[0]
        # cons = pc.constraints[0]
        # v_pc, p_pc = zip(*[(v, kpi['average_power']['cycle']) for kpi, v, c in zip(kpis, pc.wind_speeds, cons) if kpi and kpi['sim_successful'] and all([a >= -1e-6 for a in c])])
        # print(v_pc)
        # print(p_pc)
    elif True:
        v = 4.36
        x = [1544.205012763466, 199.9999999997906, 0.4363323129985824, 200.0, 200.0]
        v = 25.25
        x = [2992.358659799002, 2606.6493866322085, 0.9055261203127993, 200.0, 200.0]
        v = 25
        x = [4999.999999999997, 2819.640575754534, 0.9055261203127993, 200.0, 200.0]

        op_cycle012.environment_state.set_reference_wind_speed(v)
        op_cycle012.x0_real_scale = x
        # cons, kpis = op_cycle012.eval_point()
        op_cycle012.eval_point(True, relax_errors=True)
        plt.show()
    else:
        v = 25.921
        beta = 1.004
        x = [4971.247013479256, 2648.1564946215904, beta, 200.0, 200.0]
        x = [4975, 2648.1564946215904, beta, 200.0, 200.0]

        op_cycle012.environment_state.set_reference_wind_speed(v)
        op_cycle012.x0_real_scale = x
        cons, kpis = op_cycle012.eval_point()
        print(kpis['max_tether_force']['out'])
        op_cycle012.eval_point(True, relax_errors=True)
        plt.show()

        env_state.set_reference_wind_speed(v)
        cycle_settings = {
            'cycle': {
                'elevation_angle_traction': beta,
                'tether_length_start_retraction': x[3]+x[4],
                'tether_length_end_retraction': x[4],
            },
            'retraction': {
                'control': ('tether_force_ground', x[1]),
            },
            'transition': {
                'control': ('reeling_speed', 0., x[1], x[0]),
                'time_step': 0.25,
            },
            'traction': {
                'control': ('tether_force_ground', x[0]),
                'azimuth_angle': phi_out,
                'course_angle': chi_out,
            },
        }
        # cycle = Cycle(cycle_settings)
        # cycle.run_simulation(sys_props_v3, env_state, {'enable_steady_state_errors': False})
        # id_fail = 16
        # print(cycle.traction_phase.kinematics[id_fail].__dict__)

        le = 343.08011417660805  #cycle.traction_phase.kinematics[id_fail].straight_tether_length
        kappas = []
        betas = np.linspace(25*np.pi/180., np.pi/2, 45)
        for b in betas:
            kin0 = KiteKinematics(straight_tether_length=le, elevation_angle=b,
                                  azimuth_angle=phi_out,
                                  course_angle=chi_out)
            env_state.calculate(kin0.z)
            sys_props_v3.update(kin0.straight_tether_length, True)
            ss0 = SteadyState()
            ss0.control_settings = ('tether_force_ground', x[0])
            try:
                ss0.find_state(sys_props_v3, env_state, kin0)
                kappas.append(ss0.kinematic_ratio)
            except:
                kappas.append(None)
        plt.plot(betas*180./np.pi, kappas)
        plt.axvline(1.005*180./np.pi, color='k', linestyle='--')

    plt.show()


def blabla():
    wind_speeds = np.linspace(6.1, 18.5, 13)
    starting_points = [
        [2957, 300, 0.45, 200.0, 200.0],
        [4230, 300, 0.45, 200.0, 200.0],
        [5000, 545, 0.45, 200.0, 200.0],
        [5000, 789, 0.45, 200.0, 200.0],
        [5000, 989, 0.45, 200.0, 200.0],
        [5000, 1185, 0.45, 200.0, 200.0],
        [5000, 1455, 0.45, 200.0, 200.0],
        [5000, 1717, 0.46, 200.0, 200.0],
        [5000, 1829, 0.52, 200.0, 200.0],
        [5000, 2031, 0.57, 200.0, 200.0],
        [5000, 2072, 0.71, 200.0, 200.0],
        [5000, 1974, 0.87, 200.0, 200.0],
        [5000, 1658, 60*np.pi/180., 200.0, 200.0],
    ]
    #     [2957, 300, 0.45, 200.0, 200.0],
    #     [4208, 300, 0.45, 200.0, 200.0],
    #     [5000, 537, 0.45, 200.0, 200.0],
    #     [5000, 778, 0.45, 200.0, 200.0],
    #     [5000, 976, 0.45, 200.0, 200.0],
    #     [5000, 1170, 0.45, 200.0, 200.0],
    #     [5000, 1411, 0.45, 200.0, 200.0],
    #     [5000, 1642, 0.45, 200.0, 200.0],
    #     [5000, 1815, 0.51, 200.0, 200.0],
    #     [5000, 1955, 0.56, 200.0, 200.0],
    #     [5000, 2086, 0.68, 200.0, 200.0],
    #     [5000, 2019, 0.84, 200.0, 200.0],
    #     [5000, 1689, 0.99, 200.0, 200.0],
    #     [5000, 1344, 70*np.pi/180, 200.0, 200.0],
    # ]

    from qsm import LogProfile, TractionPhaseHybrid
    from cycle_optimizer import OptimizerCycle3
    from kitepower_kites import sys_props_v3

    env_state = LogProfile()
    env_state.set_reference_wind_speed(9.)

    theta_out = 25*np.pi/180.
    #TODO: decide on angles
    phi_out = 13 * np.pi / 180.
    chi_out = 100 * np.pi / 180.
    le0 = 200
    le1 = 400
    cycle_settings = {
        'cycle': {
            'tether_length_start_retraction': le1,
            'tether_length_end_retraction': le0,
            'elevation_angle_traction': theta_out,
            'traction_phase': TractionPhaseHybrid,
        },
        'retraction': {
            # 'control': ('tether_force_ground', f_in),
        },
        'transition': {
            # 'control': ('reeling_speed', 0., f_in, f_out),
            'time_step': 0.25,
        },
        'traction': {
            # 'control': ('tether_force_ground', f_out),
            'azimuth_angle': phi_out,
            'course_angle': chi_out,
        },
    }
    op_cycle0123 = OptimizerCycle3(cycle_settings, sys_props_v3, env_state, True, False, reduce_x=(0, 1, 2, 3), apply_ineq_cons=1)

    pc = PowerCurveConstructor(wind_speeds)
    pc.set_labels_bounds_and_system(op_cycle0123)

    # Full design variables for all wind speeds.
    op_seq_cycle3 = {
        np.inf: {'power_optimizer': op_cycle0123, 'dx0': np.array([0., 0., 0., 0., 0.])},
    }
    pc.run_predefined_sequence2(op_seq_cycle3, starting_points, run_label='full cycle')

    pc.plot_optimization_results()
    pc.plot_optimal_trajectories(wind_speed_ids=[10, 11, 12])
    plt.show()


if __name__ == "__main__":
    power_curve_v3()
    blabla()
