from pyOpt import Optimization, SLSQP
from scipy import optimize as op
import numpy as np
import matplotlib.pyplot as plt

from qsm import DuoCycle, OptCycle


class OptimizerError(Exception):
    pass


def read_slsqp_output_file(print_details=True):
    """Read relevant information from pyOpt's output file for the SLSQP algorithm."""
    i_iter = 0
    with open('SLSQP.out') as f:
        for line in f:
            if line[:11] == "     ITER =":
                i_iter += 1
                x_iter = []
                while True:
                    line = next(f)
                    xi = line.strip()
                    if not xi:
                        break
                    elif line[:38] == "        NUMBER OF FUNC-CALLS:  NFUNC =":
                        nfunc_line = line
                        ngrad_line = next(f)
                        break
                    else:
                        x_iter.append(float(xi))
                if print_details:
                    print("Iter {}: x=".format(i_iter) + str(x_iter))
            elif line[:38] == "        NUMBER OF FUNC-CALLS:  NFUNC =":
                nfunc_line = line
                ngrad_line = next(f)
                break

    nit = i_iter
    try:
        nfev = nfunc_line.split()[5]
    except:
        nfev = nfunc_line.split()[4][1:]
    try:
        njev = ngrad_line.split()[5]
    except:
        njev = ngrad_line.split()[4][1:]

    return nit, nfev, njev


def convert_optimization_result(op_sol, nit, nfev, njev, print_details, iprint):
    """Write pyOpt's optimization results to the same format as the output of SciPy's minimize function."""
    op_res = {
        'x': [v.value for v in op_sol._variables.values()],
        'success': op_sol.opt_inform['value'] == 0,
        'message': op_sol.opt_inform['text'],
        'fun': op_sol._objectives[0].value,
        'nit': nit,
        'nfev': nfev,
        'njev': njev,
    }
    if print_details:
        print("{}    (Exit mode {})".format(op_res['message'], op_sol.opt_inform['value']))
        print("            Current function value: {}".format(op_res['fun']))
        if iprint:
            print("            Iterations: {}".format(nit))
            print("            Function evaluations: {}".format(nfev))
            print("            Gradient evaluations: {}".format(njev))

    return op_res


class Optimizer:
    """Class collecting useful functionalities for solving an optimization problem and evaluating the results using
    different settings and, thereby, enabling assessing the effect of these settings."""
    def __init__(self, x0_real_scale, bounds_real_scale, scaling_x, n_ineq_cons, n_eq_cons, system_properties, environment_state):
        assert isinstance(x0_real_scale, np.ndarray)
        assert isinstance(bounds_real_scale, np.ndarray)
        assert isinstance(scaling_x, np.ndarray)

        # Simulation side conditions.
        self.system_properties = system_properties
        self.environment_state = environment_state

        # Optimization configuration.
        self.use_library = 'pyopt'  # Either 'pyopt' or 'scipy' can be opted. pyOpt is in general faster, however more
        # cumbersome to install.

        self.scaling_x = scaling_x  # Scaling the optimization variables will affect the optimization. In general, a
        # similar search range is preferred for each variable.
        self.x0_real_scale = x0_real_scale  # Optimization starting point.
        self.bounds_real_scale = bounds_real_scale  # Optimization variables bounds defining the search space.
        self.n_ineq_cons = n_ineq_cons
        self.n_eq_cons = n_eq_cons

        # Settings inferred from the optimization configuration.
        self.x0 = None  # Scaled starting point.
        self.x_opt_real_scale = None  # Optimal solution for the optimization vector.

        # Optimization operational attributes.
        self.x_last = None  # Optimization vector used for the latest evaluation function call.
        self.obj = None  # Value of the objective/cost function of the latest evaluation function call.
        self.cons = None  # Values of the inequality constraint functions of the latest evaluation function call.
        self.x_progress = []  # Evaluated optimization vectors of every conducted optimization iteration - only tracked
        # when using Scipy.

        # Optimization result.
        self.op_res = None  # Result dictionary of optimization function.

    def clear_result_attributes(self):
        """Clear the inferred optimization settings and results before re-running the optimization."""
        self.x0 = None
        self.x_last = None
        self.obj = None
        self.cons = None
        self.x_progress = []
        self.x_opt_real_scale = None
        self.op_res = None

    def eval_point(self, relax_errors=False, x_real_scale=None):
        """Evaluate simulation results using the provided optimization vector. Uses either the optimization vector
        provided as argument, the optimal vector, or the starting point for the simulation."""
        if x_real_scale is None:
            if self.x_opt_real_scale is not None:
                x_real_scale = self.x_opt_real_scale
            else:
                x_real_scale = self.x0_real_scale
        res = self.eval_fun(x_real_scale, scale_x=False, relax_errors=relax_errors)
        return res

    def eval_fun_pyopt(self, x, *args):
        """PyOpt's implementation of SLSQP can produce NaN's in the optimization vector or contain values that violate
        the bounds."""
        if np.isnan(x).any():
            raise OptimizerError("Optimization vector contains NaN's.")

        x_full = x

        bounds_adhered = (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e6).all() and \
                         (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e6).all()
        if not bounds_adhered:
            raise OptimizerError("Optimization bounds violated.")

        obj, cons = self.eval_fun(x_full, *args)
        return obj, [-c for c in cons], 0

    def obj_fun(self, x, *args):
        """Scipy's implementation of SLSQP uses separate functions for the objective and constraints. Since the
        objective and constraints result from the same simulation, a work around is provided to prevent running the same
        simulation twice."""
        x_full = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.cons = self.eval_fun(x_full, *args)
            self.x_last = x_full.copy()

        return self.obj

    def cons_fun(self, x, return_i=-1, *args):
        """Scipy's implementation of SLSQP uses separate functions for the objective and every constraint. Since the
        objective and constraints result from the same simulation, a work around is provided to prevent running the same
        simulation twice."""
        x_full = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.cons = self.eval_fun(x_full, *args)
            self.x_last = x_full.copy()

        if return_i > -1:
            return self.cons[return_i]
        else:
            return self.cons

    def callback_fun_scipy(self, x):
        """Function called when using Scipy for every optimization iteration (does not include function calls for
        determining the gradient)."""
        if np.isnan(x).any():
            raise OptimizerError("Optimization vector contains nan's.")
        self.x_progress.append(x.copy())

    def optimize(self, *args, maxiter=30, iprint=1):
        """Perform optimization."""
        self.clear_result_attributes()
        # Construct scaled starting point and bounds
        self.x0 = self.x0_real_scale*self.scaling_x
        bounds = self.bounds_real_scale.copy()
        bounds[:, 0] = bounds[:, 0]*self.scaling_x
        bounds[:, 1] = bounds[:, 1]*self.scaling_x

        starting_point = self.x0

        print_details = True
        ftol, eps = 1e-6, 1e-6
        if self.use_library == 'scipy':
            con = {
                'type': 'ineq',  # g_i(x) >= 0
                'fun': self.cons_fun,
            }
            cons = []
            for i in range(self.n_ineq_cons):
                cons.append(con.copy())
                cons[-1]['args'] = (i, *args)

            options = {
                'disp': print_details,
                'maxiter': maxiter,
                'ftol': ftol,
                'eps': eps,
                'iprint': iprint,
            }
            self.op_res = dict(op.minimize(self.obj_fun, starting_point, args=args, bounds=bounds, method='SLSQP',
                                           options=options, callback=self.callback_fun_scipy, constraints=cons))
        elif self.use_library == 'pyopt':
            op_problem = Optimization('Pumping cycle power', self.eval_fun_pyopt)
            op_problem.addObj('f')

            x_range = range(len(self.x0))
            for i_x, xi0, b in zip(x_range, starting_point, bounds):
                op_problem.addVar('x{}'.format(i_x), 'c', lower=b[0], upper=b[1], value=xi0)

            for i_c in range(self.n_eq_cons):
                op_problem.addCon('g{}'.format(i_c), 'e')

            for i_c in range(self.n_eq_cons, self.n_ineq_cons+self.n_eq_cons):
                op_problem.addCon('g{}'.format(i_c), 'i')

            # grad = Gradient(op_problem, sens_type='FD', sens_mode=sens_mode, sens_step=eps)
            # f0, g0, _ = self.eval_fun_pyopt(starting_point)
            # grad_fun = lambda f, g: grad.getGrad(starting_point, {}, [f], g)
            # dff0, dgg0 = grad_fun(f0, g0)
            # if np.any(dff0 == 0.):
            #     print("!!! Gradient contains zero component !!!")

            optimizer = SLSQP()
            optimizer.setOption('IPRINT', iprint)  # -1 - None, 0 - Screen, 1 - File
            optimizer.setOption('MAXIT', maxiter)
            optimizer.setOption('ACC', ftol)

            optimizer(op_problem, sens_type='FD', sens_step=eps, *args)
            op_sol = op_problem.solution(0)

            if iprint == 1:
                nit, nfev, njev = read_slsqp_output_file(print_details)
            else:
                nit, nfev, njev = 0, 0, 0

            self.op_res = convert_optimization_result(op_sol, nit, nfev, njev, print_details, iprint)
        else:
            raise ValueError("Invalid library provided.")

        res_x = self.op_res['x']
        self.x_opt_real_scale = res_x/self.scaling_x

        return self.x_opt_real_scale


class OptimizerCycleDuo(Optimizer):
    """Tether force controlled cycle optimizer. Zero reeling speed is used as setpoint for transition phase."""
    OPT_VARIABLE_LABELS = [
        "kappa_out",
        "kappa_in",
        "Reel-out\nforce [N]",
        "Reel-in\nforce [N]",
        "Reel-out\nelevation\nangle [rad]",
        "Tether length [m]",
    ]
    X0_REAL_SCALE_DEFAULT = np.array([8, 2, 4000, 500, 30*np.pi/180., 180])
    SCALING_X_DEFAULT = np.array([1, 1, 1e-4, 1e-4, 1, 1e-3])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [0, 15],
        [0, 15],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [25 * np.pi / 180, 60. * np.pi / 180.],
        [150, 250],
    ])
    N_INEQ_CONS = 0
    N_EQ_CONS = 2

    def __init__(self, system_properties, environment_state):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[2, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        bounds[3, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)

    def eval_fun(self, x, scale_x=True, relax_errors=True):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        cycle = DuoCycle(self.system_properties, self.environment_state)
        cycle_res = cycle.run_simulation(*x_real_scale, relax_errors=relax_errors)
        mcp, v_out, v_in, l2d_error_out, l2d_error_in, elevation_rate_in = cycle_res

        # Prepare the simulation by updating simulation parameters.
        env_state = self.environment_state
        env_state.calculate(100.)
        power_wind_100m = .5 * env_state.air_density * env_state.wind_speed ** 3

        # Determine optimization objective and constraints.
        obj = -mcp/power_wind_100m/self.system_properties.kite_projected_area

        eq_cons = np.array([l2d_error_out, l2d_error_in])

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit

        speed_violation_traction = v_out - speed_min_limit
        ineq_cons_traction_min_speed = speed_violation_traction / speed_min_limit + 1e-6
        speed_violation_traction = v_out - speed_max_limit
        ineq_cons_traction_max_speed = -speed_violation_traction / speed_max_limit + 1e-6

        speed_violation_retraction = v_in - speed_min_limit
        ineq_cons_retraction_min_speed = speed_violation_retraction / speed_min_limit + 1e-6
        speed_violation_retraction = v_in - speed_max_limit
        ineq_cons_retraction_max_speed = -speed_violation_retraction / speed_max_limit + 1e-6

        ineq_cons = np.array([ineq_cons_traction_min_speed, ineq_cons_traction_max_speed,
                              ineq_cons_retraction_min_speed, ineq_cons_retraction_max_speed])

        return obj, np.hstack([eq_cons, ineq_cons])


class OptimizerCycle(Optimizer):
    """Tether force controlled cycle optimizer. Zero reeling speed is used as setpoint for transition phase."""
    N_POINTS_PER_PHASE = [5, 10]
    OPT_VARIABLE_LABELS = [
        "Duration reel-out [s]",
        "Duration reel-in [s]",
        "Reel-out\nelevation\nangle [rad]",
        "Minimum tether length [m]",
    ]
    OPT_VARIABLE_LABELS = np.append(OPT_VARIABLE_LABELS,
                                    ["Reel-out\nforce {} [N]".format(i+1) for i in range(N_POINTS_PER_PHASE[0])] +
                                    ["Reel-in\nforce {} [N]".format(i+1) for i in range(N_POINTS_PER_PHASE[1])])

    X0_REAL_SCALE_DEFAULT = np.array([30, 20, 30*np.pi/180., 150])
    X0_REAL_SCALE_DEFAULT = np.append(X0_REAL_SCALE_DEFAULT, np.hstack((np.ones(N_POINTS_PER_PHASE[0])*4000,
                                                                        np.ones(N_POINTS_PER_PHASE[1])*1500)))
    SCALING_X_DEFAULT = np.array([1e-2, 1e-2, 1, 1e-3])
    SCALING_X_DEFAULT = np.append(SCALING_X_DEFAULT, np.ones(np.sum(N_POINTS_PER_PHASE))*1e-4)
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [1, 120],
        [1, 60],
        [25*np.pi/180, 60.*np.pi/180.],
        [200, 250],
    ])
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.empty((np.sum(N_POINTS_PER_PHASE), 2))*np.nan, axis=0)
    N_INEQ_CONS = np.sum(N_POINTS_PER_PHASE)*2
    N_EQ_CONS = 1

    def __init__(self, system_properties, environment_state):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[4:4+np.sum(self.N_POINTS_PER_PHASE), :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)

    def eval_fun(self, x, scale_x=True, relax_errors=True):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        cycle = OptCycle(self.system_properties, self.environment_state)
        cycle.n_points_per_phase = self.N_POINTS_PER_PHASE
        forces_out = x_real_scale[4:4+self.N_POINTS_PER_PHASE[0]]
        forces_in = x_real_scale[4+self.N_POINTS_PER_PHASE[0]:4+np.sum(self.N_POINTS_PER_PHASE)]
        cycle_res = cycle.run_simulation(*x_real_scale[:4], forces_out, forces_in, relax_errors=relax_errors)
        if not relax_errors:
            return cycle_res
        mcp, v_out, v_in, tether_length_end, l2d_errors_out, l2d_errors_in = cycle_res

        # Prepare the simulation by updating simulation parameters.
        env_state = self.environment_state
        env_state.calculate(100.)
        power_wind_100m = .5 * env_state.air_density * env_state.wind_speed ** 3

        # Determine optimization objective and constraints.
        obj = -mcp/power_wind_100m/self.system_properties.kite_projected_area

        eq_cons = np.array([tether_length_end-x_real_scale[3]])

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit

        ineq_cons = []
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = v_out[i] - speed_min_limit
            ineq_cons.append(speed_violation_traction / speed_min_limit + 1e-6)
            speed_violation_traction = v_out[i] - speed_max_limit
            ineq_cons.append(-speed_violation_traction / speed_max_limit + 1e-6)
        for i in range(self.N_POINTS_PER_PHASE[1]):
            speed_violation_retraction = -v_in[i] - speed_min_limit
            ineq_cons.append(speed_violation_retraction / speed_min_limit + 1e-6)
            speed_violation_retraction = -v_in[i] - speed_max_limit
            ineq_cons.append(-speed_violation_retraction / speed_max_limit + 1e-6)

        return obj, np.hstack([eq_cons, np.array(ineq_cons)])


class OptimizerCycleKappa(Optimizer):
    """Tether force controlled cycle optimizer. Zero reeling speed is used as setpoint for transition phase."""
    N_POINTS_PER_PHASE = [5, 10]
    OPT_VARIABLE_LABELS = [
        "Duration reel-out [s]",
        "Duration reel-in [s]",
        "Reel-out\nelevation\nangle [rad]",
        "Minimum tether length [m]",
    ]
    OPT_VARIABLE_LABELS = OPT_VARIABLE_LABELS + \
                          ["Reel-out\nforce {} [N]".format(i + 1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Reel-in\nforce {} [N]".format(i + 1) for i in range(N_POINTS_PER_PHASE[1])] + \
                          ["Kinematics\nratios out".format(i + 1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Kinematics\nratios in".format(i + 1) for i in range(N_POINTS_PER_PHASE[1])]

    X0_REAL_SCALE_DEFAULT = np.array([30, 20, 30 * np.pi / 180., 170])
    X0_REAL_SCALE_DEFAULT = np.hstack([X0_REAL_SCALE_DEFAULT, np.ones(N_POINTS_PER_PHASE[0]) * 2000,
                                       np.ones(N_POINTS_PER_PHASE[1]) * 1500, np.ones(N_POINTS_PER_PHASE[0]) * 4,
                                       np.ones(N_POINTS_PER_PHASE[1]) * 1.5])
    SCALING_X_DEFAULT = np.array([1e-2, 1e-2, 1, 1e-3])
    SCALING_X_DEFAULT = np.hstack([SCALING_X_DEFAULT, np.ones(np.sum(N_POINTS_PER_PHASE)) * 1e-4,
                                   np.ones(np.sum(N_POINTS_PER_PHASE))])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [1, 120],
        [1, 60],
        [25*np.pi/180, 70.*np.pi/180.],
        [150, 250],
    ])
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.empty((np.sum(N_POINTS_PER_PHASE), 2))*np.nan, axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[0, 15]] * np.sum(N_POINTS_PER_PHASE)), axis=0)
    N_INEQ_CONS = np.sum(N_POINTS_PER_PHASE)*3 #+ N_POINTS_PER_PHASE[1] - 1
    N_EQ_CONS = 1 + np.sum(N_POINTS_PER_PHASE)
    EQ_CONS_LABELS = ['Tether length periodicity'] + \
                     ['L/D error out {}'.format(i) for i in range(N_POINTS_PER_PHASE[0])] + \
                     ['L/D error in {}'.format(i) for i in range(N_POINTS_PER_PHASE[1])]

    def __init__(self, system_properties, environment_state):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[4:4+np.sum(self.N_POINTS_PER_PHASE), :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)

    def eval_fun(self, x, scale_x=True, relax_errors=True):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        duration_out = x_real_scale[0]
        tether_length_end = x_real_scale[3]
        forces_out = x_real_scale[4:4+self.N_POINTS_PER_PHASE[0]]
        forces_in = x_real_scale[4+self.N_POINTS_PER_PHASE[0]:4+np.sum(self.N_POINTS_PER_PHASE)]
        kappas_out = x_real_scale[4+np.sum(self.N_POINTS_PER_PHASE):
                                  4+np.sum(self.N_POINTS_PER_PHASE)+self.N_POINTS_PER_PHASE[0]]
        kappas_in = x_real_scale[4+np.sum(self.N_POINTS_PER_PHASE)+self.N_POINTS_PER_PHASE[0]:
                                 4+2*np.sum(self.N_POINTS_PER_PHASE)]

        cycle = OptCycle(self.system_properties, self.environment_state)
        cycle.n_points_per_phase = self.N_POINTS_PER_PHASE
        cycle_res = cycle.run_simulation(*x_real_scale[:4], forces_out, forces_in, kappas_out, kappas_in, relax_errors=relax_errors)
        if not relax_errors:
            return cycle_res

        # Prepare the simulation by updating simulation parameters.
        env_state = self.environment_state
        env_state.calculate(100.)
        power_wind_100m = .5*env_state.air_density*env_state.wind_speed**3

        # Determine optimization objective and constraints.
        obj = -cycle_res['mean_cycle_power']/power_wind_100m/self.system_properties.kite_projected_area

        eq_cons = np.hstack([[(cycle_res['tether_length_end']-tether_length_end)*1e-3],
                             cycle_res['out']['lift_to_drag_errors'], cycle_res['in']['lift_to_drag_errors']])

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit

        ineq_cons = []
        # for i in range(self.N_POINTS_PER_PHASE[1]-1):
        #     ineq_cons.append(forces_in[i]-forces_in[i+1])
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_traction / speed_min_limit)
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[1]):
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_retraction / speed_min_limit)
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_retraction / speed_max_limit)
        for v_tau in cycle_res['out']['tangential_speeds']:
            pattern_duration = 300/v_tau
            ineq_cons.append(duration_out/pattern_duration - 1)

        return obj, np.hstack([eq_cons, np.array(ineq_cons),
                               np.array(cycle_res['in']['tangential_speed_factors'])])


class OptimizerCycleCut(Optimizer):
    """Tether force controlled cycle optimizer. Zero reeling speed is used as setpoint for transition phase."""
    N_POINTS_PER_PHASE = [5, 10]
    OPT_VARIABLE_LABELS = [
        "Duration reel-out [s]",
        "Duration reel-in [s]",
        "Reel-out\nelevation\nangle [rad]",
        "Minimum tether length [m]",
    ]
    OPT_VARIABLE_LABELS = OPT_VARIABLE_LABELS + \
                          ["Reel-out\nforce {} [N]".format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Reel-in\nforce {} [N]".format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                          ["Kinematics\nratios out".format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Kinematics\nratios in".format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                          ['Wind speed [m/s]']

    X0_REAL_SCALE_DEFAULT = np.array([30, 20, 30*np.pi/180., 170])
    X0_REAL_SCALE_DEFAULT = np.hstack([X0_REAL_SCALE_DEFAULT, np.ones(N_POINTS_PER_PHASE[0])*2000,
                                       np.ones(N_POINTS_PER_PHASE[1])*1500, np.ones(N_POINTS_PER_PHASE[0])*4,
                                       np.ones(N_POINTS_PER_PHASE[1])*1.5, [8]])
    SCALING_X_DEFAULT = np.array([1e-2, 1e-2, 1, 1e-3])
    SCALING_X_DEFAULT = np.hstack([SCALING_X_DEFAULT, np.ones(np.sum(N_POINTS_PER_PHASE))*1e-4,
                                   np.ones(np.sum(N_POINTS_PER_PHASE)), [1e-1]])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [20, 120],
        [1, 180],
        [25*np.pi/180, 70.*np.pi/180.],
        [150, 250],
    ])
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.empty((np.sum(N_POINTS_PER_PHASE), 2))*np.nan, axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[0, 15]] * np.sum(N_POINTS_PER_PHASE)), axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[3, 25]]), axis=0)
    N_INEQ_CONS = 1 + np.sum(N_POINTS_PER_PHASE)*3 + N_POINTS_PER_PHASE[1]-1
    N_EQ_CONS = 1 + np.sum(N_POINTS_PER_PHASE)

    def __init__(self, system_properties, environment_state, cut='out', cut_power=1500):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[4:4+np.sum(self.N_POINTS_PER_PHASE), :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)
        self.cut = cut
        self.cut_power = cut_power

    def eval_fun(self, x, scale_x=True, relax_errors=True):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        duration_out = x_real_scale[0]
        tether_length_end = x_real_scale[3]
        forces_out = x_real_scale[4:4+self.N_POINTS_PER_PHASE[0]]
        forces_in = x_real_scale[4+self.N_POINTS_PER_PHASE[0]:4+np.sum(self.N_POINTS_PER_PHASE)]
        kappas_out = x_real_scale[4+np.sum(self.N_POINTS_PER_PHASE):
                                  4+np.sum(self.N_POINTS_PER_PHASE)+self.N_POINTS_PER_PHASE[0]]
        kappas_in = x_real_scale[4+np.sum(self.N_POINTS_PER_PHASE)+self.N_POINTS_PER_PHASE[0]:
                                 4+2*np.sum(self.N_POINTS_PER_PHASE)]
        wind_speed = x_real_scale[-1]

        cycle = OptCycle(self.system_properties, self.environment_state)
        cycle.n_points_per_phase = self.N_POINTS_PER_PHASE
        cycle_res = cycle.run_simulation(*x_real_scale[:4], forces_out, forces_in, kappas_out, kappas_in,
                                         relax_errors=True, wind_speed=wind_speed)
        if not relax_errors:
            return cycle_res

        if self.cut == 'in':
            obj = wind_speed
        else:
            obj = -wind_speed

        eq_cons = np.hstack([[cycle_res['tether_length_end']-tether_length_end],
                             cycle_res['out']['lift_to_drag_errors'], cycle_res['in']['lift_to_drag_errors']])

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit

        ineq_cons = [(cycle_res['mean_cycle_power'] - self.cut_power)*1e-4]
        for i in range(self.N_POINTS_PER_PHASE[1]-1):
            ineq_cons.append(forces_in[i]-forces_in[i+1])
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_traction / speed_min_limit)
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[1]):
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_retraction / speed_min_limit)
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_retraction / speed_max_limit)
        for v_tau in cycle_res['out']['tangential_speeds']:
            pattern_duration = 300/v_tau
            ineq_cons.append(duration_out/pattern_duration - 1)

        return obj, np.hstack([eq_cons, np.array(ineq_cons), np.array(cycle_res['in']['tangential_speed_factors'])])


def eval_limit(wind_speeds=[22, 22.5, 22.7, 22.9], cut=None):
    from qsm import LogProfile, NormalisedWindTable1D
    from kitev3_10mm_tether import sys_props_v3

    fig, ax = plt.subplots(5, 1)
    ax_traj = plt.figure().gca()
    plt.axis('equal')
    plt.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    # env_state = LogProfile()
    env_state = NormalisedWindTable1D()
    h = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
    v = [0.31916794, 0.38611869, 0.55029902, 0.65804212, 0.73519261, 0.79609015,
         0.84480401, 0.88139352, 0.89594957, 0.90781971, 0.92918228, 0.94297737,
         0.95193377, 0.9588552, 0.95571666, 0.88390919, 0.84849594]
    env_state.heights = h
    env_state.h_ref = h[np.argmax(v)]
    env_state.normalised_wind_speeds = v/np.amax(v)
    mcps, success = [], []
    for i, v in enumerate(wind_speeds):
        print("Wind speed = {:.1f} m/s".format(v))
        env_state.set_reference_wind_speed(v)
        oc = OptimizerCycleKappa(sys_props_v3, env_state)
        if i > 0:
            oc.x0_real_scale = x_opt
        oc.optimize(maxiter=1000)
        x_opt = oc.x_opt_real_scale
        print("x_opt = ", x_opt)
        cycle_res = oc.eval_point()
        mcps.append(cycle_res['mean_cycle_power'])
        success.append(oc.op_res['success'])

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        if not oc.op_res['success']:
            plt.figure()
            plt.bar(range(oc.N_EQ_CONS), cons[:oc.N_EQ_CONS])
            plt.xticks(range(oc.N_EQ_CONS), oc.EQ_CONS_LABELS, rotation='vertical')

        ax[0].plot(cycle_res['out']['time'], [ss.tether_force_ground for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[0].plot(cycle_res['in']['time'], [ss.tether_force_ground for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[0].set_ylabel('Tether\nforce [N]')
        ax[1].plot(cycle_res['out']['time'], [ss.reeling_speed for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[1].plot(cycle_res['in']['time'], [ss.reeling_speed for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[1].set_ylabel('Reeling\nspeed [m/s]')
        ax[2].plot(cycle_res['out']['time'], [kp.straight_tether_length for kp in cycle_res['out']['kite_positions']], '.-', color='C{}'.format(i))
        ax[2].plot(cycle_res['in']['time'], [kp.straight_tether_length for kp in cycle_res['in']['kite_positions']], '.-', color='C{}'.format(i))
        ax[2].plot(cycle_res['in']['time'][-1], cycle_res['out']['kite_positions'][0].straight_tether_length, 's', ms=6, mfc='None', color='C{}'.format(i))
        ax[2].set_ylabel('Tether\nlength [m]')
        ax[3].plot(cycle_res['out']['time'], [ss.kite_tangential_speed for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[3].plot(cycle_res['in']['time'], [ss.kite_tangential_speed for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[3].set_ylabel('Tangential\nspeed [m/s]')
        ax[4].plot(cycle_res['out']['time'], [ss.kinematic_ratio for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[4].plot(cycle_res['in']['time'], [ss.kinematic_ratio for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[4].set_ylabel('Kinematic\nratio [-]')
        ax[-1].set_xlabel('Time [s]')
    
        ax_traj.plot([kp.x for kp in cycle_res['out']['kite_positions']], [kp.z for kp in cycle_res['out']['kite_positions']], '.-', color='C{}'.format(i))
        ax_traj.plot([kp.x for kp in cycle_res['in']['kite_positions']], [kp.z for kp in cycle_res['in']['kite_positions']], '.-', color='C{}'.format(i))

    print("Mean cycle powers", mcps)
    plt.figure()
    plt.plot(wind_speeds, mcps)
    wind_speeds_x, mcps_x = zip(*[(v, mcp) for v, mcp, s in zip(wind_speeds, mcps, success) if not s])
    plt.plot(wind_speeds_x, mcps_x, 'x')


    if cut is not None:
        i += 1
        if cut == 'in':
            cut_power = 0
        else:
            cut_power = 4000
        oc = OptimizerCycleCut(sys_props_v3, env_state, cut, cut_power)
        oc.x0_real_scale = np.hstack([x_opt, v])
        x_opt = oc.optimize()

        ax_x = plt.subplots(2, 2)[1].reshape(-1)
        for a, x, bnds, lbl in zip(ax_x, x_opt, oc.bounds_real_scale, oc.OPT_VARIABLE_LABELS):
            a.axhline(bnds[0], ls='--', color='C3')
            a.axhline(bnds[1], ls='--', color='C3')
            a.bar(0, x)
            a.xaxis.set_visible(False)
            a.set_ylabel(lbl)

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Equality constraints:", cons[:oc.N_EQ_CONS])
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        print("Inequality constraints:", cons[oc.N_EQ_CONS:])
        print("x_opt = ", x_opt)
        print("Minimum wind speed @ 100 m = {:.2f} m/s".format(x_opt[-1]))
        env_state.calculate(10)
        print("Minimum wind speed @ 10 m = {:.2f} m/s".format(env_state.wind_speed))
        env_state.set_reference_wind_speed(x_opt[-1])
        cycle_res = oc.eval_point()
        print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))

        ax[0].plot(cycle_res['out']['time'], [ss.tether_force_ground for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[0].plot(cycle_res['in']['time'], [ss.tether_force_ground for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[0].set_ylabel('Tether\nforce [N]')
        ax[1].plot(cycle_res['out']['time'], [ss.reeling_speed for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[1].plot(cycle_res['in']['time'], [ss.reeling_speed for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[1].set_ylabel('Reeling\nspeed [m/s]')
        ax[2].plot(cycle_res['out']['time'], [kp.straight_tether_length for kp in cycle_res['out']['kite_positions']], '.-', color='C{}'.format(i))
        ax[2].plot(cycle_res['in']['time'], [kp.straight_tether_length for kp in cycle_res['in']['kite_positions']], '.-', color='C{}'.format(i))
        ax[2].plot(cycle_res['in']['time'][-1], cycle_res['out']['kite_positions'][0].straight_tether_length, 's', ms=6, mfc='None', color='C{}'.format(i))
        ax[2].set_ylabel('Tether\nlength [m]')
        ax[3].plot(cycle_res['out']['time'], [ss.kite_tangential_speed for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[3].plot(cycle_res['in']['time'], [ss.kite_tangential_speed for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[3].set_ylabel('Tangential\nspeed [m/s]')
        ax[4].plot(cycle_res['out']['time'], [ss.kinematic_ratio for ss in cycle_res['out']['steady_states']], '.-', color='C{}'.format(i))
        ax[4].plot(cycle_res['in']['time'], [ss.kinematic_ratio for ss in cycle_res['in']['steady_states']], '.-', color='C{}'.format(i))
        ax[4].set_ylabel('Kinematic\nratio [-]')
        ax[-1].set_xlabel('Time [s]')

        ax_traj.plot([kp.x for kp in cycle_res['out']['kite_positions']], [kp.z for kp in cycle_res['out']['kite_positions']], '.-', color='C{}'.format(i))
        ax_traj.plot([kp.x for kp in cycle_res['in']['kite_positions']], [kp.z for kp in cycle_res['in']['kite_positions']], '.-', color='C{}'.format(i))

    plt.show()


def find_cut_in_speed(x0=None):
    from qsm import LogProfile
    from kitev3_10mm_tether import sys_props_v3

    fig, ax = plt.subplots(5, 1)
    ax_traj = plt.figure().gca()
    plt.axis('equal')
    plt.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    env_state = LogProfile()
    oc = OptimizerCycleCut(sys_props_v3, env_state)
    if x0 is not None:
        oc.x0_real_scale = x0
    x_opt = oc.optimize()

    ax_x = plt.subplots(2, 2)[1].reshape(-1)
    for a, x, bnds, lbl in zip(ax_x, x_opt, oc.bounds_real_scale, oc.OPT_VARIABLE_LABELS):
        a.axhline(bnds[0], ls='--', color='C3')
        a.axhline(bnds[1], ls='--', color='C3')
        a.bar(0, x)
        a.xaxis.set_visible(False)
        a.set_ylabel(lbl)

    cons = oc.eval_point(relax_errors=True)[1]
    print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
    print("Equality constraints:", cons[:oc.N_EQ_CONS])
    print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
    print("Inequality constraints:", cons[oc.N_EQ_CONS:])
    print("x_opt = ", x_opt)
    print("Minimum wind speed @ 100 m = {:.2f} m/s".format(x_opt[-1]))
    env_state.calculate(10)
    print("Minimum wind speed @ 10 m = {:.2f} m/s".format(env_state.wind_speed))
    env_state.set_reference_wind_speed(x_opt[-1])
    cycle_res = oc.eval_point()
    print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))

    ax[0].plot(cycle_res['out']['time'], [ss.tether_force_ground for ss in cycle_res['out']['steady_states']], '.-', color='C0')
    ax[0].plot(cycle_res['in']['time'], [ss.tether_force_ground for ss in cycle_res['in']['steady_states']], '.-', color='C0')
    ax[0].set_ylabel('Tether\nforce [N]')
    ax[1].plot(cycle_res['out']['time'], [ss.reeling_speed for ss in cycle_res['out']['steady_states']], '.-', color='C0')
    ax[1].plot(cycle_res['in']['time'], [ss.reeling_speed for ss in cycle_res['in']['steady_states']], '.-', color='C0')
    ax[1].set_ylabel('Reeling\nspeed [m/s]')
    ax[2].plot(cycle_res['out']['time'], [kp.straight_tether_length for kp in cycle_res['out']['kite_positions']], '.-', color='C0')
    ax[2].plot(cycle_res['in']['time'], [kp.straight_tether_length for kp in cycle_res['in']['kite_positions']], '.-', color='C0')
    ax[2].plot(cycle_res['in']['time'][-1], cycle_res['out']['kite_positions'][0].straight_tether_length, 's', ms=6, mfc='None', color='C0')
    ax[2].set_ylabel('Tether\nlength [m]')
    ax[3].plot(cycle_res['out']['time'], [ss.kite_tangential_speed for ss in cycle_res['out']['steady_states']], '.-', color='C0')
    ax[3].plot(cycle_res['in']['time'], [ss.kite_tangential_speed for ss in cycle_res['in']['steady_states']], '.-', color='C0')
    ax[3].set_ylabel('Tangential\nspeed [m/s]')
    ax[4].plot(cycle_res['out']['time'], [ss.kinematic_ratio for ss in cycle_res['out']['steady_states']], '.-', color='C0')
    ax[4].plot(cycle_res['in']['time'], [ss.kinematic_ratio for ss in cycle_res['in']['steady_states']], '.-', color='C0')
    ax[4].set_ylabel('Kinematic\nratio [-]')
    ax[-1].set_xlabel('Time [s]')

    ax_traj.plot([kp.x for kp in cycle_res['out']['kite_positions']], [kp.z for kp in cycle_res['out']['kite_positions']], '.-', color='C0')
    ax_traj.plot([kp.x for kp in cycle_res['in']['kite_positions']], [kp.z for kp in cycle_res['in']['kite_positions']], '.-', color='C0')

    plt.show()


if __name__ == "__main__":
    # find_cut_in_speed()
    # # Log profile
    # eval_limit(wind_speeds=[22, 22.5, 22.7, 22.9], cut='out')
    # eval_limit(wind_speeds=[7, 6.5, 6.4], cut='in')
    # eval_limit(wind_speeds=np.linspace(7, 22, 6))

    # eval_limit(wind_speeds=[9, 8.5], cut='in')
    eval_limit(wind_speeds=np.arange(9, 30, .5))  #, cut='out')
