import pandas as pd
from pyOpt import Optimization, SLSQP
from scipy import optimize as op
import numpy as np
import matplotlib.pyplot as plt
from utils import add_panel_labels

from qsm import OptCycle, LogProfile, NormalisedWindTable1D, SteadyState, KiteKinematics
from kitev3_10mm_tether import sys_props_v3

class OptimizerError(Exception):
    pass


dowa_heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
height_ceiling = 500
gtol = 1e-4


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
        'exit_mode': op_sol.opt_inform['value'],
        'message': op_sol.opt_inform['text'],
        'fun': op_sol._objectives[0].value,
        'nit': nit,
        'nfev': nfev,
        'njev': njev,
    }
    if print_details:
        line = "-"*60
        print(line)
        print("{}    (Exit mode {})".format(op_res['message'], op_sol.opt_inform['value']))
        print("            Current function value: {}".format(op_res['fun']))
        if iprint > 0:
            print("            Iterations: {}".format(nit))
            print("            Function evaluations: {}".format(nfev))
            print("            Gradient evaluations: {}".format(njev))
        print(line)
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

        # bounds_adhered = (x - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e6).all() and \
        #                  (x - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e6).all()
        # if not bounds_adhered:
        #     raise OptimizerError("Optimization bounds violated.")

        obj, cons = self.eval_fun(x, *args)
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

    def optimize(self, *args, maxiter=30, iprint=-1):
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


class OptimizerReelOutState(Optimizer):
    OPT_VARIABLE_LABELS = [
        'Reel-out\nforce [N]',
        "Reel-out elevation\nangle [rad]",
        'Tether length [m]',
        'Kinematic\nratio [-]',
        'Wind speed [m/s]'
    ]
    X0_REAL_SCALE_DEFAULT = np.array([2000, 30*np.pi/180., 250, 4, 10])

    SCALING_X_DEFAULT = np.array([1e-4, 1, 1e-2, 1, 1e-1])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [np.nan, np.nan],
        [0*np.pi/180, 70.*np.pi/180.],
        [100, np.inf],
        [0, 15],
        [1, 50],
    ])
    N_INEQ_CONS = 4
    N_EQ_CONS = 1

    def __init__(self, system_properties, environment_state, elevation_angle=None, wind_speed=None, tether_length=None, obj_factors=[0, 1, 0]):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[0, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        self.obj_factors = obj_factors

        x0 = self.X0_REAL_SCALE_DEFAULT.copy()
        if elevation_angle is not None:
            x0[1] = elevation_angle
            bounds[1, :] = elevation_angle
        if tether_length is not None:
            x0[2] = tether_length
            bounds[2, :] = tether_length
        if wind_speed is not None:
            x0[4] = wind_speed
            bounds[4, :] = wind_speed
        super().__init__(x0, bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)

    def eval_fun(self, x, scale_x=True, relax_errors=True):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        f = x_real_scale[0]
        beta = x_real_scale[1]
        tether_length = x_real_scale[2]
        kappa = x_real_scale[3]
        wind_speed = x_real_scale[4]

        # Create objects containing the kite position and course angle for reel-out and reel-in.
        kp = {
            'straight_tether_length': tether_length,
            'azimuth_angle': 0.2012354421510459,  # [rad]
            'elevation_angle': beta,
            'course_angle': 1.6249951922452899,  # [rad]
        }
        kp = KiteKinematics(**kp)

        self.system_properties.update(tether_length, True)
        self.environment_state.set_reference_wind_speed(wind_speed)
        self.environment_state.calculate(kp.z)

        ss = SteadyState({'enable_steady_state_errors': not relax_errors})
        ss.control_settings = ('tether_force_ground', f)
        ss.find_state_opt(kappa, self.system_properties, self.environment_state, kp, relax_errors)

        p = ss.power_ground
        vt = ss.reeling_speed

        if not relax_errors:
            return p, vt, kp.z

        obj = self.obj_factors[0] * wind_speed * 1e-2 - self.obj_factors[1] * p * 1e-3 - self.obj_factors[2]*tether_length*1e-2
        eq_cons = [ss.lift_to_drag_error]
        assert len(eq_cons) == self.N_EQ_CONS

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit

        ineq_cons = []

        speed_violation_traction = vt - speed_min_limit
        ineq_cons.append(speed_violation_traction / speed_max_limit)

        speed_violation_traction = vt - speed_max_limit
        ineq_cons.append(-speed_violation_traction / speed_max_limit)

        ineq_cons += [kp.z - 100]
        ineq_cons += [500 - kp.z]
        assert len(ineq_cons) == self.N_INEQ_CONS

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
                          ["Kinematic\nratio out {} [-]".format(i + 1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Kinematic\nratio in {} [-]".format(i + 1) for i in range(N_POINTS_PER_PHASE[1])]

    X0_REAL_SCALE_DEFAULT = np.array([30, 20, 15*np.pi/180., 170])
    X0_REAL_SCALE_DEFAULT = np.hstack([X0_REAL_SCALE_DEFAULT, np.ones(N_POINTS_PER_PHASE[0]) * 2000,
                                       np.ones(N_POINTS_PER_PHASE[1]) * 1500, np.ones(N_POINTS_PER_PHASE[0]) * 4,
                                       np.ones(N_POINTS_PER_PHASE[1]) * 1.5])
    SCALING_X_DEFAULT = np.array([1e-2, 1e-2, 1, 1e-3])
    SCALING_X_DEFAULT = np.hstack([SCALING_X_DEFAULT, np.ones(np.sum(N_POINTS_PER_PHASE)) * 1e-4,
                                   np.ones(np.sum(N_POINTS_PER_PHASE))])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [1, 500],
        [20, 300],
        [0*np.pi/180, 50.*np.pi/180.],
        [100, np.inf],
    ])
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.empty((np.sum(N_POINTS_PER_PHASE), 2))*np.nan, axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[0, 15]] * np.sum(N_POINTS_PER_PHASE)), axis=0)
    N_INEQ_CONS = np.sum(N_POINTS_PER_PHASE)*4 + N_POINTS_PER_PHASE[1]
    INEQ_CONS_LABELS = ['Reel-in force reduction {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[1]-1)] + \
                       ['Min. reel-out speed {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                       ['Max. reel-out speed {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                       ['Max. reel-in speed {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                       ['Min. tangential speed out {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                       ['Min. tangential speed in {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                       ['Min. height out'] + \
                       ['Max. height in {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                       ['Max. reel-out power {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                       ['Max. reel-in power {}'.format(i+1) for i in range(N_POINTS_PER_PHASE[1])]
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
        assert len(eq_cons) == self.N_EQ_CONS

        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit
        power_max_limit = self.system_properties.reeling_power_max_limit

        ineq_cons = []
        for i in range(self.N_POINTS_PER_PHASE[1]-1):
            ineq_cons.append(forces_in[i]-forces_in[i+1])
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[1]):
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_retraction / speed_max_limit)
        for v_tau in cycle_res['out']['tangential_speeds']:
            pattern_duration = 300/v_tau
            ineq_cons.append(duration_out/pattern_duration - 1)
        ineq_cons += cycle_res['in']['tangential_speed_factors']
        ineq_cons += [cycle_res['out']['kite_positions'][0].z - 100]
        ineq_cons += [height_ceiling - kp.z for kp in cycle_res['in']['kite_positions']]
        ineq_cons += [power_max_limit - p for p in cycle_res['out']['power']]
        ineq_cons += [power_max_limit + p for p in cycle_res['in']['power']]
        assert len(ineq_cons) == self.N_INEQ_CONS

        return obj, np.hstack([eq_cons, np.array(ineq_cons)])


class OptimizerCycleCutKappa(Optimizer):
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
                          ["Kinematic\nratio out {} [-]".format(i+1) for i in range(N_POINTS_PER_PHASE[0])] + \
                          ["Kinematic\nratio in {} [-]".format(i+1) for i in range(N_POINTS_PER_PHASE[1])] + \
                          ['Wind speed [m/s]']

    X0_REAL_SCALE_DEFAULT = np.array([30, 20, 15*np.pi/180., 170])
    X0_REAL_SCALE_DEFAULT = np.hstack([X0_REAL_SCALE_DEFAULT, np.ones(N_POINTS_PER_PHASE[0])*2000,
                                       np.ones(N_POINTS_PER_PHASE[1])*1500, np.ones(N_POINTS_PER_PHASE[0])*4,
                                       np.ones(N_POINTS_PER_PHASE[1])*1.5, [11]])
    SCALING_X_DEFAULT = np.array([1e-2, 1e-2, 1, 1e-3])
    SCALING_X_DEFAULT = np.hstack([SCALING_X_DEFAULT, np.ones(np.sum(N_POINTS_PER_PHASE))*1e-4,
                                   np.ones(np.sum(N_POINTS_PER_PHASE)), [1e-1]])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [1, 500],
        [20, 300],
        [0*np.pi/180, 50.*np.pi/180.],
        [100, np.inf],
    ])
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.empty((np.sum(N_POINTS_PER_PHASE), 2))*np.nan, axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[0, 15]] * np.sum(N_POINTS_PER_PHASE)), axis=0)
    BOUNDS_REAL_SCALE_DEFAULT = np.append(BOUNDS_REAL_SCALE_DEFAULT, np.array([[3, 50]]), axis=0)
    N_INEQ_CONS = 1 + np.sum(N_POINTS_PER_PHASE)*4 + N_POINTS_PER_PHASE[1]
    N_EQ_CONS = 1 + np.sum(N_POINTS_PER_PHASE)

    def __init__(self, system_properties, environment_state, obj_factors=[-1e-2, -1e-5], force_wind_speed=False):
        # obj_factors suggested for cut-out: [-1e-2, -1e-5]
        # obj_factors suggested for cut-in: [1e-1, -1e-3]

        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[4:4+np.sum(self.N_POINTS_PER_PHASE), :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(), self.N_INEQ_CONS,
                         self.N_EQ_CONS, system_properties, environment_state)
        self.obj_factors = obj_factors
        self.force_wind_speed = force_wind_speed
        if force_wind_speed:
            self.N_EQ_CONS += 1

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

        obj = self.obj_factors[0] * wind_speed + self.obj_factors[1] * cycle_res['mean_cycle_power']

        eq_cons = np.hstack([[cycle_res['tether_length_end'] - tether_length_end],
                             cycle_res['out']['lift_to_drag_errors'], cycle_res['in']['lift_to_drag_errors']])
        if self.force_wind_speed:
            eq_cons = np.hstack([[wind_speed-self.force_wind_speed], eq_cons])
        assert len(eq_cons) == self.N_EQ_CONS

        speed_min_limit = self.system_properties.reeling_speed_min_limit
        speed_max_limit = self.system_properties.reeling_speed_max_limit
        power_max_limit = self.system_properties.reeling_power_max_limit

        ineq_cons = []
        for i in range(self.N_POINTS_PER_PHASE[1] - 1):
            ineq_cons.append(forces_in[i] - forces_in[i + 1])
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_min_limit
            ineq_cons.append(speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[0]):
            speed_violation_traction = cycle_res['out']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_traction / speed_max_limit)
        for i in range(self.N_POINTS_PER_PHASE[1]):
            speed_violation_retraction = -cycle_res['in']['reeling_speeds'][i] - speed_max_limit
            ineq_cons.append(-speed_violation_retraction / speed_max_limit)
        for v_tau in cycle_res['out']['tangential_speeds']:
            pattern_duration = 300 / v_tau
            ineq_cons.append(duration_out / pattern_duration - 1)
        ineq_cons += cycle_res['in']['tangential_speed_factors']
        ineq_cons += [cycle_res['out']['kite_positions'][0].z - 100]
        ineq_cons += [height_ceiling - kp.z for kp in cycle_res['in']['kite_positions']]
        ineq_cons += [power_max_limit - p for p in cycle_res['out']['power']]
        ineq_cons += [power_max_limit + p for p in cycle_res['in']['power']]
        ineq_cons += [cycle_res['mean_cycle_power']*1e-4]
        assert len(ineq_cons) == self.N_INEQ_CONS

        return obj, np.hstack([eq_cons, np.array(ineq_cons)])


def eval_limit(wind_speeds=[22, 22.5, 22.7, 22.9], cut=None, env_state=LogProfile()):
    fig, ax = plt.subplots(5, 1)
    ax_traj = plt.figure().gca()
    plt.axis('equal')
    plt.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    mcps, success = [], []
    opt_vars = np.empty((0, 4))
    for i, v in enumerate(wind_speeds):
        print("Wind speed = {:.1f} m/s".format(v))
        env_state.set_reference_wind_speed(v)
        oc = OptimizerCycleKappa(sys_props_v3, env_state)
        if i > 0:
            oc.x0_real_scale = x_opt
        oc.optimize(maxiter=100)
        x_opt = oc.x_opt_real_scale
        print("x_opt = ", x_opt)
        cycle_res = oc.eval_point()
        opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
        mcps.append(cycle_res['mean_cycle_power'])
        success.append(oc.op_res['success'])

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        if not oc.op_res['success']:
            plt.figure()
            plt.bar(range(oc.N_EQ_CONS), cons[:oc.N_EQ_CONS])
            plt.xticks(range(oc.N_EQ_CONS), oc.EQ_CONS_LABELS, rotation='vertical')

        plot_sol(cycle_res, i, ax, ax_traj)

    # if cut is not None:
    if cut == 'in':
        factors = [0.]
    elif cut == 'out':
        factors = [.5, .25, 0]
    else:
        factors = []
    for f in factors:
        i += 1
        oc = OptimizerCycleCutKappa(sys_props_v3, env_state, cut, [1, f])
        oc.x0_real_scale = np.hstack([x_opt, v])
        x_opt_cut = oc.optimize(maxiter=100)

        ax_x = plt.subplots(2, 2)[1].reshape(-1)
        for a, x, bnds, lbl in zip(ax_x, x_opt_cut, oc.bounds_real_scale, oc.OPT_VARIABLE_LABELS):
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
        print("x_opt = ", x_opt_cut)
        print("Minimum wind speed @ 100 m = {:.2f} m/s".format(x_opt_cut[-1]))
        env_state.calculate(10)
        print("Minimum wind speed @ 10 m = {:.2f} m/s".format(env_state.wind_speed))
        env_state.set_reference_wind_speed(x_opt_cut[-1])
        cycle_res = oc.eval_point()
        print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))
        print("Optimization successful = ", oc.op_res['success'])

        plot_sol(cycle_res, i, ax, ax_traj, ls='--')

        opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
        wind_speeds = np.append(wind_speeds, x_opt_cut[-1])
        mcps.append(cycle_res['mean_cycle_power'])
        success.append(oc.op_res['success'])

    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(wind_speeds, mcps, '.-')
    ax[1].plot(wind_speeds, opt_vars[:, 0])
    ax[1].set_ylabel('Duration reel-out [s]')
    ax[2].plot(wind_speeds, opt_vars[:, 1])
    ax[2].set_ylabel('Duration reel-in [s]')
    ax[3].plot(wind_speeds, opt_vars[:, 2]*180./np.pi)
    ax[3].set_ylabel('Elevation angle [deg]')
    ax[4].plot(wind_speeds, opt_vars[:, 3])
    ax[4].set_ylabel('Min. tether length [m]')
    try:
        wind_speeds_x, mcps_x = zip(*[(v, mcp) for v, mcp, s in zip(wind_speeds, mcps, success) if not s])
        ax[0].plot(wind_speeds_x, mcps_x, 'x')
    except ValueError:
        pass

    plt.show()


def find_cut_speed(x0=None, cut='out', env_state=LogProfile(), maxiter=100):
    fig, ax = plt.subplots(5, 1)
    ax_traj = plt.figure().gca()
    plt.axis('equal')
    plt.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    oc = OptimizerCycleCutKappa(sys_props_v3, env_state, cut)
    if x0 is not None:
        oc.x0_real_scale = x0
    x_opt = oc.optimize(maxiter=maxiter)

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
    print("Optimization successful = ", oc.op_res['success'])

    plot_sol(cycle_res, 0, ax, ax_traj)


def plot_sol(cycle_res, i, ax, ax_traj, ls='.-'):
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

    ax_traj.plot([kp.x for kp in cycle_res['out']['kite_positions']], [kp.z for kp in cycle_res['out']['kite_positions']], ls, color='C{}'.format(i), mfc='None')
    ax_traj.plot([kp.x for kp in cycle_res['in']['kite_positions']], [kp.z for kp in cycle_res['in']['kite_positions']], ls, color='C{}'.format(i), mfc='None')

    l = cycle_res['out']['kite_positions'][0].straight_tether_length
    beta = np.linspace(cycle_res['out']['kite_positions'][0].elevation_angle,
                       cycle_res['in']['kite_positions'][-1].elevation_angle, 30)
    ax_traj.plot(np.cos(beta)*l, np.sin(beta)*l, color='C{}'.format(i), lw=.5)

    traj = [
        [kp.x for kp in cycle_res['out']['kite_positions']]+[kp.x for kp in cycle_res['in']['kite_positions']],
        [kp.z for kp in cycle_res['out']['kite_positions']] + [kp.z for kp in cycle_res['in']['kite_positions']],
    ]
    return traj


def construct_power_curve_start_middle(starting_wind_speed=9., wind_speed_step=[.2, .5], power_optimization_limits=[6, 22], env_state=LogProfile(), maxiter=300):
    cut = True
    plot_nth_step = 3

    ax_pc = plt.figure().gca()
    ax_vars = plt.subplots(4, 1, sharex=True)[1]

    fig, ax = plt.subplots(5, 1)
    ax_profile, ax_traj = plt.subplots(1, 2, sharey=True)[1]
    env_state.set_reference_wind_speed(1)
    env_state.plot_wind_profile(ax_profile)
    ax_traj.set_aspect('equal')
    ax_traj.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    mcps_low, success_low, wind_speeds_low = [], [], []
    opt_vars = np.empty((0, 4))
    success_counter = 0
    v = starting_wind_speed
    while True:
        print("Wind speed = {:.1f} m/s".format(v))
        env_state.set_reference_wind_speed(v)
        oc = OptimizerCycleKappa(sys_props_v3, env_state)
        if success_counter == 0:
            oc.optimize(maxiter=300)
        else:
            oc.x0_real_scale = x0_next
            try:
                oc.optimize(maxiter=maxiter)
            except FloatingPointError:
                if v < power_optimization_limits[0]:
                    break
                else:
                    opt_vars = np.vstack((opt_vars, [np.nan] * 4))
                    mcps_low.append(np.nan)
                    success_low.append(False)
                    wind_speeds_low.append(v)
                    v -= wind_speed_step[0]
                    continue

        if oc.op_res['success']:  # or oc.op_res.get('exit_mode', -1) == 9:
            x0_next = oc.x_opt_real_scale
            v0_cut_in = v
            if success_counter == 0:
                x0_start = x0_next
            cycle_res = oc.eval_point()
        else:
            try:
                cycle_res = oc.eval_point()
            except:
                cycle_res = {}

        opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
        mcps_low.append(cycle_res['mean_cycle_power'])
        success_low.append(oc.op_res['success'])
        wind_speeds_low.append(v)

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        # plt.figure()
        # plt.bar(range(oc.N_INEQ_CONS), cons[oc.N_EQ_CONS:])
        # plt.xticks(range(oc.N_INEQ_CONS), oc.INEQ_CONS_LABELS, rotation='vertical')
        # if not oc.op_res['success']:
        #     plt.figure()
        #     plt.bar(range(oc.N_EQ_CONS), cons[:oc.N_EQ_CONS])
        #     plt.xticks(range(oc.N_EQ_CONS), oc.EQ_CONS_LABELS, rotation='vertical')

        if oc.op_res['success']:
            if success_counter == 0:
                plot_sol(cycle_res, 0, ax, ax_traj, '-s')
                ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C0', mfc='None')
                for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(v, x, 's', color='C0', mfc='None')
            elif success_counter % plot_nth_step == 0 or v < 6.:
                i_clr = success_counter//plot_nth_step
                plot_sol(cycle_res, i_clr, ax, ax_traj)
                ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
                for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(v, x, 's', color='C{}'.format(i_clr), mfc='None')
            success_counter += 1

        if not oc.op_res['success'] and v < power_optimization_limits[0]:
            break
        v -= wind_speed_step[0]

    if cut:
        # Cut-in
        oc = OptimizerCycleCutKappa(sys_props_v3, env_state, 'in')
        oc.x0_real_scale = np.hstack([x0_next, v0_cut_in])
        x_opt = oc.optimize(maxiter=maxiter)

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Equality constraints:", cons[:oc.N_EQ_CONS])
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        print("Inequality constraints:", cons[oc.N_EQ_CONS:])
        print("Minimum wind speed @ 100 m = {:.2f} m/s".format(x_opt[-1]))
        env_state.calculate(10)
        print("Minimum wind speed @ 10 m = {:.2f} m/s".format(env_state.wind_speed))
        env_state.set_reference_wind_speed(x_opt[-1])
        cycle_res = oc.eval_point()
        print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))
        i_clr = success_counter // plot_nth_step
        plot_sol(cycle_res, i_clr, ax, ax_traj, ls='--.')
        ax_pc.plot(x_opt[-1], cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
        for a, x in zip(ax_vars, x_opt[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(x_opt[-1], x, 's', color='C{}'.format(i_clr), mfc='None')
        success_counter += 1

        opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
        mcps_low.append(cycle_res['mean_cycle_power'])
        success_low.append(oc.op_res['success'])
        wind_speeds_low.append(x_opt[-1])

    opt_vars = opt_vars[::-1, :]

    # High
    mcps_high, success_high, wind_speeds_high = [], [], []
    success_counter += 1
    v = starting_wind_speed + wind_speed_step[1]
    while True:
        print("Wind speed = {:.1f} m/s".format(v))
        env_state.set_reference_wind_speed(v)
        oc = OptimizerCycleKappa(sys_props_v3, env_state)
        if not mcps_high:
            oc.x0_real_scale = x0_start
        else:
            oc.x0_real_scale = x0_next
        oc.optimize(maxiter=maxiter)

        if oc.op_res['success']:  # or oc.op_res.get('exit_mode', -1) == 9:
            x0_next = oc.x_opt_real_scale
            v0_cut_out = v
            print("x_opt = ", x0_next)
            cycle_res = oc.eval_point()
        else:
            try:
                cycle_res = oc.eval_point()
            except:
                cycle_res = {}

        opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
        mcps_high.append(cycle_res.get('mean_cycle_power', np.nan))
        success_high.append(oc.op_res['success'])
        wind_speeds_high.append(v)

        cons = oc.eval_point(relax_errors=True)[1]
        print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
        print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
        # plt.figure()
        # plt.bar(range(oc.N_INEQ_CONS), cons[oc.N_EQ_CONS:])
        # plt.xticks(range(oc.N_INEQ_CONS), oc.INEQ_CONS_LABELS, rotation='vertical')
        # if not oc.op_res['success']:
        #     plt.figure()
        #     plt.bar(range(oc.N_EQ_CONS), cons[:oc.N_EQ_CONS])
        #     plt.xticks(range(oc.N_EQ_CONS), oc.EQ_CONS_LABELS, rotation='vertical')

        if oc.op_res['success']:
            if success_counter % plot_nth_step == 0:
                i_clr = success_counter // plot_nth_step
                plot_sol(cycle_res, i_clr, ax, ax_traj)
                ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
                for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(v, x, 's', color='C{}'.format(i_clr), mfc='None')
            success_counter += 1

        if not oc.op_res['success'] and v > power_optimization_limits[1]:
            break
        v += wind_speed_step[1]

    if cut:
        # Cut-out
        for f in [.5, .1, 0]:
            oc = OptimizerCycleCutKappa(sys_props_v3, env_state, 'out', [1, f])
            oc.x0_real_scale = np.hstack([x0_next, v0_cut_out])
            oc.optimize(maxiter=maxiter)

            cons = oc.eval_point(relax_errors=True)[1]
            print("Max. abs. equality constraints:", np.max(np.abs(cons[:oc.N_EQ_CONS])))
            print("Equality constraints:", cons[:oc.N_EQ_CONS])
            print("Min. inequality constraint:", np.min(cons[oc.N_EQ_CONS:]))
            print("Inequality constraints:", cons[oc.N_EQ_CONS:])
            print("Minimum wind speed @ 100 m = {:.2f} m/s".format(oc.x_opt_real_scale[-1]))
            env_state.calculate(10)
            print("Minimum wind speed @ 10 m = {:.2f} m/s".format(env_state.wind_speed))
            env_state.set_reference_wind_speed(oc.x_opt_real_scale[-1])
            cycle_res = oc.eval_point()
            print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))
            i_clr = success_counter // plot_nth_step
            plot_sol(cycle_res, i_clr, ax, ax_traj, '--.')
            ax_pc.plot(oc.x_opt_real_scale[-1], cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
            for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(oc.x_opt_real_scale[-1], x, 's', color='C{}'.format(i_clr), mfc='None')
            success_counter += 1

            opt_vars = np.vstack((opt_vars, [oc.x_opt_real_scale[:4]]))
            mcps_high.append(cycle_res['mean_cycle_power'])
            success_high.append(oc.op_res['success'])
            wind_speeds_high.append(oc.x_opt_real_scale[-1])

    #Power curve
    mcps = mcps_low[::-1] + mcps_high
    success = success_low[::-1] + success_high
    wind_speeds = wind_speeds_low[::-1] + wind_speeds_high

    wind_speeds_s, mcps_s = zip(*[(v, mcp) for v, mcp, s in zip(wind_speeds, mcps, success) if s])
    ax_pc.plot(wind_speeds_s, mcps_s, '.-')
    ax_pc.plot(wind_speeds, mcps)

    wind_speeds = np.array(wind_speeds)
    success = np.array(success)
    ax_pc.plot(wind_speeds[~success], [0]*np.sum(~success), 'x')
    ax_pc.set_ylim([0, 1e4])

    ax_vars[0].plot(wind_speeds, opt_vars[:, 0], '.-')
    ax_vars[0].plot(wind_speeds[~success], opt_vars[~success, 0], 'x')
    ax_vars[0].set_ylabel('Duration\nreel-out [s]')
    ax_vars[1].plot(wind_speeds, opt_vars[:, 1], '.-')
    ax_vars[1].plot(wind_speeds[~success], opt_vars[~success, 1], 'x')
    ax_vars[1].set_ylabel('Duration\nreel-in [s]')
    ax_vars[2].plot(wind_speeds, opt_vars[:, 2] * 180. / np.pi, '.-')
    ax_vars[2].plot(wind_speeds[~success], opt_vars[~success, 2] * 180. / np.pi, 'x')
    ax_vars[2].set_ylabel('Elevation\nangle [deg]')
    ax_vars[3].plot(wind_speeds, opt_vars[:, 3], '.-')
    ax_vars[3].plot(wind_speeds[~success], opt_vars[~success, 3], 'x')
    ax_vars[3].set_ylabel('Min. tether\nlength [m]')


def test_convergence(starting_wind_speed=9., wind_speed_step=3., env_state=LogProfile(), maxiter=300):
    ax_pc = plt.figure().gca()
    ax_vars = plt.subplots(4, 1, sharex=True)[1]

    fig, ax = plt.subplots(5, 1)
    ax_profile, ax_traj = plt.subplots(1, 2, sharey=True)[1]
    env_state.set_reference_wind_speed(1)
    env_state.plot_wind_profile(ax_profile)
    ax_traj.set_aspect('equal')
    ax_traj.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    v = starting_wind_speed

    print("Wind speed = {:.1f} m/s".format(v))
    env_state.set_reference_wind_speed(v)
    oc = OptimizerCycleKappa(sys_props_v3, env_state)
    oc.optimize(maxiter=300)

    assert oc.op_res['success']
    x0_next = oc.x_opt_real_scale
    cycle_res = oc.eval_point()

    plot_sol(cycle_res, 0, ax, ax_traj, '-s')
    ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C0', mfc='None')
    for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(v, x, 's', color='C0', mfc='None')

    v += wind_speed_step

    print("Wind speed = {:.1f} m/s".format(v))
    env_state.set_reference_wind_speed(v)
    oc = OptimizerCycleKappa(sys_props_v3, env_state)
    oc.x0_real_scale = x0_next
    oc.optimize(maxiter=maxiter)
    cycle_res = oc.eval_point()

    plot_sol(cycle_res, 1, ax, ax_traj, '-s')
    ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C1', mfc='None')
    for a, x in zip(ax_vars, oc.x_opt_real_scale[:4] * np.array([1, 1, 180. / np.pi, 1])): a.plot(v, x, 's', color='C1',
                                                                                                  mfc='None')

    oc = OptimizerCycleCutKappa(sys_props_v3, env_state, obj_factors=[0, 1], force_wind_speed=v)
    oc.x0_real_scale = np.hstack([x0_next, starting_wind_speed])
    x_opt = oc.optimize(maxiter=maxiter)
    print(x_opt[-1])
    cycle_res = oc.eval_point()

    plot_sol(cycle_res, 1, ax, ax_traj, '-s')
    ax_pc.plot(v, cycle_res['mean_cycle_power'], 's', color='C1', mfc='None')
    for a, x in zip(ax_vars, oc.x_opt_real_scale[:4] * np.array([1, 1, 180. / np.pi, 1])): a.plot(v, x, 's', color='C1', mfc='None')


def record_opt_result(env_state, x_opt, cycle_res, eq_cons, ineq_cons):
    assert len(x_opt) == 34
    record = {}
    for h in dowa_heights:
        env_state.calculate(h)
        record['vw{0:03.0f}'.format(h)] = env_state.wind_speed
    for i, xi in enumerate(x_opt):
        record['x{:02d}'.format(i)] = x_opt[i]

    for i, hi in enumerate(eq_cons):
        record['h{:03d}'.format(i)] = hi
    for i, gi in enumerate(ineq_cons):
        record['g{:03d}'.format(i)] = gi

    record['mcp'] = cycle_res['mean_cycle_power']
    record['mean_ro_power'] = np.mean([ss.power_ground for ss in cycle_res['out']['steady_states']])
    record['mean_ri_power'] = np.mean([ss.power_ground for ss in cycle_res['in']['steady_states']])
    record['max_height_ro'] = cycle_res['out']['kite_positions'][-1].z
    record['min_height_ro'] = cycle_res['out']['kite_positions'][0].z
    record['max_height_ri'] = max([kp.z for kp in cycle_res['in']['kite_positions']])

    n_out, n_in = 5, 10
    for i in range(n_out):
        record['ro_speed{:02d}'.format(i)] = cycle_res['out']['reeling_speeds'][i]
        record['ro_power{:02d}'.format(i)] = cycle_res['out']['steady_states'][i].power_ground
        record['tan_speed_ro{:02d}'.format(i)] = cycle_res['out']['tangential_speeds'][i]
        record['x_kite{:02d}'.format(i)] = cycle_res['out']['kite_positions'][i].x
        record['z_kite{:02d}'.format(i)] = cycle_res['out']['kite_positions'][i].z
    for i in range(n_in):
        record['ri_speed{:02d}'.format(i)] = cycle_res['in']['reeling_speeds'][i]
        record['ri_power{:02d}'.format(i)] = cycle_res['in']['steady_states'][i].power_ground
        record['tan_speed_ri{:02d}'.format(i)] = cycle_res['in']['tangential_speeds'][i]
        record['x_kite{:02d}'.format(i+n_out)] = cycle_res['in']['kite_positions'][i].x
        record['z_kite{:02d}'.format(i+n_out)] = cycle_res['in']['kite_positions'][i].z

    record = pd.Series(record).sort_index()

    return record


def construct_power_curve(wind_speed_step=[.01, 1.], power_optimization_limits=20, env_state=LogProfile(), maxiter=300,
                          export_file=None, obj_factors_mcp={'in': -1e-7, 'out': -5e-6}, plot=False):
    plot_nth_step = 3
    if plot:
        ax_pc = plt.figure().gca()
        ax_vars = plt.subplots(4, 1, sharex=True)[1]

        fig, ax = plt.subplots(5, 1)

        fig = plt.figure(figsize=[6.4, 2.5])
        plt.subplots_adjust(left=.1, right=1, bottom=.195, wspace=0)
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 2], height_ratios=[1])
        ax_profile = fig.add_subplot(spec[0, 0])
        ax_traj = fig.add_subplot(spec[0, 1])
        ax_traj.set_xlabel('Downwind position [m]')
        ax_traj.yaxis.set_ticklabels([])

        env_state.set_reference_wind_speed(1)
        env_state.plot_wind_profile(ax_profile)
        ax_traj.set_aspect('equal')
        ax_traj.grid()
        ax_profile.get_shared_y_axes().join(ax_profile, ax_traj)
        # ax_traj.plot(150*np.cos(np.linspace(0, np.pi/2, 15)), 150*np.sin(np.linspace(0, np.pi/2, 15)), ':', color='grey')

    # Run cut-in optimization
    oc_ci = OptimizerCycleCutKappa(sys_props_v3, env_state, [1e-1, obj_factors_mcp['in']])  #-1e-3
    x_opt = oc_ci.optimize(maxiter=maxiter)
    assert oc_ci.op_res['success'], "Cut-in optimization failed"
    x0_next = x_opt[:-1]

    print("Minimum wind speed @ {:.0f} m = {:.2f} m/s".format(env_state.h_ref, x_opt[-1]))
    env_state.set_reference_wind_speed(x_opt[-1])

    # Collect and write cut-in results
    cycle_res = oc_ci.eval_point()
    print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))

    cons = oc_ci.eval_point(relax_errors=True)[1]
    eq_cons = cons[:oc_ci.N_EQ_CONS]
    ineq_cons = cons[oc_ci.N_EQ_CONS:]
    print("Max. abs. equality constraints:", np.max(np.abs(eq_cons)))
    print("Min. inequality constraint:", np.min(ineq_cons))
    if not (np.max(np.abs(eq_cons)) < gtol and np.min(ineq_cons) > -gtol):
        print("CONSTRAINTS VIOLATED")
        print("Equality constraints:", eq_cons)
        print("Inequality constraints:", ineq_cons)
        raise ValueError

    opt_vars = np.empty((0, 4))
    success_counter = 0
    mcps, success, wind_speeds = [], [], []
    succeeded_opts = pd.DataFrame()

    # opt_vars = np.vstack((opt_vars, oc_ci.x_opt_real_scale[:4]))
    # mcps.append(cycle_res['mean_cycle_power'])
    # success.append(oc_ci.op_res['success'])
    # wind_speeds.append(x_opt[-1])
    #
    # record = record_opt_result(env_state, x_opt[:-1], cycle_res, eq_cons, ineq_cons)
    # succeeded_opts = succeeded_opts.append(record, ignore_index=True)
    #
    # # Plot cut-in results
    # i_clr = 0
    # plot_sol(cycle_res, i_clr, ax, ax_traj, ls='.-')
    # ax_pc.plot(x_opt[-1], cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
    # for a, x in zip(ax_vars, x_opt[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(x_opt[-1], x, 's', color='C{}'.format(i_clr), mfc='None')
    # success_counter += 1

    # Run main optimization
    vw = x_opt[-1]
    i = 0
    while True:
        vw += wind_speed_step[min((i, len(wind_speed_step)-1))]
        print("Wind speed = {:.2f} m/s".format(vw))
        env_state.set_reference_wind_speed(vw)
        oc = OptimizerCycleKappa(sys_props_v3, env_state)
        oc.x0_real_scale = x0_next
        try:
            oc.optimize(maxiter=maxiter)
        except (FloatingPointError, OptimizerError):
            if vw > power_optimization_limits:
                print("Aborted due to FloatingPointError")
                break
            else:
                opt_vars = np.vstack((opt_vars, [np.nan]*4))
                mcps.append(np.nan)
                success.append(False)
                wind_speeds.append(vw)
                continue

        # Collect and write results
        if oc.op_res['success']:  # or oc.op_res.get('exit_mode', -1) == 9:
            try:
                cycle_res = oc.eval_point()
                cons = oc.eval_point(relax_errors=True)[1]
                eq_cons = cons[:oc.N_EQ_CONS]
                ineq_cons = cons[oc.N_EQ_CONS:]
                print("Mean cycle power = {:.2f}".format(cycle_res.get('mean_cycle_power', np.nan)))
                print("Max. abs. equality constraints:", np.max(np.abs(eq_cons)))
                print("Min. inequality constraint:", np.min(ineq_cons))
                if not (np.max(np.abs(eq_cons)) < gtol and np.min(ineq_cons) > -gtol):
                    print("CONSTRAINTS VIOLATED")
                    print("Equality constraints:", eq_cons)
                    print("Inequality constraints:", ineq_cons)
                    success_flag = False
                else:
                    record = record_opt_result(env_state, oc.x_opt_real_scale, cycle_res, eq_cons, ineq_cons)
                    succeeded_opts = succeeded_opts.append(record, ignore_index=True)
                    success_flag = True
            except:
                print("Evaluation of objective function failed despite successful optimization.")
                cycle_res = {}
                success_flag = False
        else:
            success_flag = False
            cycle_res = {}

        opt_vars = np.vstack((opt_vars, oc.x_opt_real_scale[:4]))
        mcps.append(cycle_res.get('mean_cycle_power', np.nan))
        success.append(success_flag)
        wind_speeds.append(vw)

        # plt.figure()
        # plt.bar(range(oc.N_INEQ_CONS), cons[oc.N_EQ_CONS:])
        # plt.xticks(range(oc.N_INEQ_CONS), oc.INEQ_CONS_LABELS, rotation='vertical')
        # if not oc.op_res['success']:
        #     plt.figure()
        #     plt.bar(range(oc.N_EQ_CONS), cons[:oc.N_EQ_CONS])
        #     plt.xticks(range(oc.N_EQ_CONS), oc.EQ_CONS_LABELS, rotation='vertical')

        # Plot results
        if success_flag:
            x0_next = oc.x_opt_real_scale
            vw0_next = vw

            if plot and success_counter % plot_nth_step == 0:
                i_clr = success_counter // plot_nth_step
                plot_sol(cycle_res, i_clr, ax, ax_traj)
                ax_pc.plot(vw, cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
                for a, x in zip(ax_vars, oc.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(vw, x, 's', color='C{}'.format(i_clr), mfc='None')
            success_counter += 1

        if not success_flag and vw > power_optimization_limits:
            break
        i += 1

    # Run cut-out optimizations
    while True:
        oc_co = OptimizerCycleCutKappa(sys_props_v3, env_state, [-1e-2, obj_factors_mcp['out']])
        oc_co.bounds_real_scale[-1, 1] = vw
        oc_co.x0_real_scale = np.hstack([x0_next, vw0_next])
        oc_co.optimize(maxiter=maxiter)
        assert oc_co.op_res['success'], "Cut-out optimization failed"
        if oc_co.x_opt_real_scale[-1] < vw0_next:
            print("Wind speed found ({:.2f}) smaller than starting wind speed ({:.2f})"
                  .format(oc_co.x_opt_real_scale[-1], vw0_next))
            break

        print("Maximum wind speed @ {:.0f} m = {:.2f} m/s".format(env_state.h_ref, oc_co.x_opt_real_scale[-1]))
        env_state.set_reference_wind_speed(oc_co.x_opt_real_scale[-1])

        # Collect and write cut-out results
        cycle_res = oc_co.eval_point()
        print("Mean cycle power = {:.2f}".format(cycle_res['mean_cycle_power']))

        cons = oc_co.eval_point(relax_errors=True)[1]
        eq_cons = cons[:oc_co.N_EQ_CONS]
        ineq_cons = cons[oc_co.N_EQ_CONS:]
        print("Max. abs. equality constraints:", np.max(np.abs(eq_cons)))
        print("Min. inequality constraint:", np.min(ineq_cons))
        if not (np.max(np.abs(eq_cons)) < gtol and np.min(ineq_cons) > -gtol):
            print("CONSTRAINTS VIOLATED")
            print("Equality constraints:", eq_cons)
            print("Inequality constraints:", ineq_cons)

        opt_vars = np.vstack((opt_vars, oc_co.x_opt_real_scale[:4]))
        mcps.append(cycle_res['mean_cycle_power'])
        success.append(oc_co.op_res['success'])
        wind_speeds.append(oc_co.x_opt_real_scale[-1])

        if oc_co.op_res['success']:
            record = record_opt_result(env_state, oc_co.x_opt_real_scale[:-1], cycle_res, eq_cons, ineq_cons)
            succeeded_opts = succeeded_opts.append(record, ignore_index=True)

        # Plot cut-out results
        if plot:
            i_clr = success_counter // plot_nth_step
            plot_sol(cycle_res, i_clr, ax, ax_traj, '.-')
            ax_pc.plot(oc_co.x_opt_real_scale[-1], cycle_res['mean_cycle_power'], 's', color='C{}'.format(i_clr), mfc='None')
            for a, x in zip(ax_vars, oc_co.x_opt_real_scale[:4]*np.array([1, 1, 180./np.pi, 1])): a.plot(oc_co.x_opt_real_scale[-1], x, 's', color='C{}'.format(i_clr), mfc='None')
        success_counter += 1

        if oc_co.x_opt_real_scale[-1] < vw-1e-3:
            print("Stopping cut-out optimizations at vw={:.3f}m/s".format(vw))
            break
        i += 1
        vw0_next = vw
        x0_next = oc_co.x_opt_real_scale[:-1]
        vw += wind_speed_step[min((i, len(wind_speed_step)-1))]

    # x0_next = oc_co.x_opt_real_scale[:-1]
    # vw = oc_co.x_opt_real_scale[-1]
    # print("Wind speed = {:.2f} m/s".format(vw))
    # env_state.set_reference_wind_speed(vw)
    # oc = OptimizerCycleKappa(sys_props_v3, env_state)
    # oc.x0_real_scale = x0_next
    # oc.optimize(maxiter=maxiter)
    # cycle_res = oc.eval_point()
    # print("Mean cycle power = {:.2f}".format(cycle_res.get('mean_cycle_power', np.nan)))

    if plot:
        #Power curve
        wind_speeds_s, mcps_s = zip(*[(v, mcp) for v, mcp, s in zip(wind_speeds, mcps, success) if s])
        ax_pc.plot(wind_speeds_s, mcps_s, '.-')
        ax_pc.plot(wind_speeds, mcps)

        ax_traj.set_xlim([0, None])
        ax_traj.set_ylim([0, None])

        wind_speeds = np.array(wind_speeds)
        success = np.array(success)
        ax_pc.plot(wind_speeds[~success], [0]*np.sum(~success), 'x')
        ax_pc.set_ylim([0, None])

        ax_vars[0].plot(wind_speeds, opt_vars[:, 0], '.-')
        ax_vars[0].plot(wind_speeds[~success], opt_vars[~success, 0], 'x')
        ax_vars[0].set_ylabel('Duration\nreel-out [s]')
        ax_vars[1].plot(wind_speeds, opt_vars[:, 1], '.-')
        ax_vars[1].plot(wind_speeds[~success], opt_vars[~success, 1], 'x')
        ax_vars[1].set_ylabel('Duration\nreel-in [s]')
        ax_vars[2].plot(wind_speeds, opt_vars[:, 2] * 180. / np.pi, '.-')
        ax_vars[2].plot(wind_speeds[~success], opt_vars[~success, 2] * 180. / np.pi, 'x')
        ax_vars[2].set_ylabel('Elevation\nangle [$^\circ$]')
        ax_vars[3].plot(wind_speeds, opt_vars[:, 3], '.-')
        ax_vars[3].plot(wind_speeds[~success], opt_vars[~success, 3], 'x')
        ax_vars[3].set_ylabel('Min. tether\nlength [m]')

    if export_file is not None:
        # 'opt_res_llj_profile2.csv'
        succeeded_opts.to_csv(export_file, index=False)

    return succeeded_opts


def mpp_curve(env_state, beta_deg=30, vw_step=.5, plot=False):
    if beta_deg is None:
        beta = None
    else:
        beta = beta_deg*np.pi/180

    oc = OptimizerReelOutState(sys_props_v3, env_state, elevation_angle=beta, obj_factors=[1, 0, 0])
    x_opt = oc.optimize(maxiter=100)
    vw = x_opt[-1]
    p, vt, h = oc.eval_point(relax_errors=False)
    opt_heights = [h]
    vw_range = [vw]
    powers = [p]
    opt_vars = [x_opt]

    sys_limit_reached = False
    while True:
        vw += vw_step

        oc = OptimizerReelOutState(sys_props_v3, env_state, elevation_angle=beta, wind_speed=vw, obj_factors=[0, 1, 0])
        x_opt = oc.optimize(maxiter=100)
        p, vt, h = oc.eval_point(relax_errors=False)
        # if not sys_limit_reached and vt > 9.99:
        if not sys_limit_reached and p > .9*sys_props_v3.reeling_power_max_limit:
            vw -= vw_step
            vw_step = .1
            sys_limit_reached = True
            continue
        elif sys_limit_reached and p > sys_props_v3.reeling_power_max_limit:
            break
        opt_heights.append(h)
        vw_range.append(vw)
        powers.append(p)
        opt_vars.append(x_opt)
    opt_vars = np.array(opt_vars)

    if plot:
        ax = plt.subplots(3, 2)[1]
        ax[0, 0].plot(vw_range, opt_vars[:, 0]*1e-3, '*-', label=beta_deg)
        ax[0, 0].set_ylabel('Tether force [kW]')
        ax[1, 0].plot(vw_range, opt_vars[:, 1]*180./np.pi, '*-', label=beta_deg)
        ax[1, 0].set_ylabel('Elevation angle [deg]')
        ax[2, 0].plot(vw_range, opt_vars[:, 2], '*-', label=beta_deg)
        ax[2, 0].set_ylabel('Tether length [m]')
        ax[0, 1].plot(vw_range, opt_heights, '*-', label=beta_deg)
        ax[0, 1].plot(vw_range, np.sin(opt_vars[:, 1])*opt_vars[:, 2], '--', label=beta_deg)
        ax[0, 1].set_ylabel('Optimal height [m]')
        ax[1, 1].plot(vw_range, np.array(powers)*1e-3, '*-', label=beta_deg)
        ax[1, 1].set_ylabel('Power [kW]')

    return vw_range, opt_vars, opt_heights


def plot_curves_and_trajectories(df, env_state, reference_height=200):
    highlight_ith_opts = np.array(range(0, df.shape[0], 2))
    # highlight_ith_opts = np.array(range(df.shape[0]))
    highlight_ith_opts[-1] = df.index[-1]
    x_col = 'vw{:03d}'.format(reference_height)
    x_label = '$v_{{w,{:03d}m}}$ [m/s]'.format(reference_height)
    highlight_wind_speeds = [df.loc[i, x_col] for i in highlight_ith_opts]
    highlight_labels = ["{:.1f}".format(v) for v in highlight_wind_speeds]

    oc = OptimizerCycleKappa(sys_props_v3, env_state)

    ax_vars = plt.subplots(3, 2, figsize=[8, 4], sharex=True)[1].reshape(-1)
    plt.subplots_adjust(left=.13, bottom=.136, right=.98, top=.98, wspace=.37)
    ax_vars[0].plot(df[x_col], df['mcp']*1e-3, '-')
    ax_vars[0].set_ylim([0, None])
    # f_cubic = np.poly1d(np.polyfit(np.insert(df[x_lbl].values[:2], 0, 0), np.insert(df['mcp'].values[:2]*1e-3, 0, 0), 3))
    # ax_vars[0].plot(np.linspace(0, 10, 100), f_cubic(np.linspace(0, 10, 100)))

    # ax_vars[0].plot(df[x_lbl], df['mean_ro_power']*1e-3, '-')
    # ax_vars[0].plot(df[x_lbl], -df['mean_ri_power']*1e-3, '-')
    ax_vars[0].set_ylabel('Mean cycle\npower [kW]')
    ax_vars[1].plot(df[x_col], df['x00']/(df['x00']+df['x01']+17)*100, '-')
    ax_vars[1].set_ylabel('Duty cycle [%]')
    ax_vars[1].set_ylim([0, 100])
    ax_vars[2].plot(df[x_col], df['x00'], '-')
    ax_vars[2].set_ylabel('Duration trac-\ntion phase [s]')
    ax_vars[3].plot(df[x_col], df['x01'], '-')
    ax_vars[3].axhline(oc.bounds_real_scale[1, 0], color='k', ls=':')
    ax_vars[3].set_ylabel('Duration retrac\n-tion phase [s]')
    ax_vars[4].plot(df[x_col], df['x02'] * 180./np.pi, '-')
    ax_vars[4].axhline(oc.bounds_real_scale[2, 0] * 180./np.pi, color='k', ls=':')
    ax_vars[4].axhline(oc.bounds_real_scale[2, 1] * 180./np.pi, color='k', ls=':')
    ax_vars[4].set_ylabel('Elevation\nangle [$^\circ$]')
    ax_vars[5].plot(df[x_col], df['x03'], '-')
    ax_vars[5].set_ylabel('Min. tether\nlength [m]')
    ax_vars[4].set_xlabel(x_label)
    ax_vars[5].set_xlabel(x_label)

    for i, a in enumerate(ax_vars):
        if i > 0:
            a.set_ylim([0, None])
        a.set_xticks(highlight_wind_speeds)
        a.set_xticklabels(highlight_labels)
        a.grid()
    add_panel_labels(ax_vars, .35)

    fig = plt.figure(figsize=[8, 7])
    plt.subplots_adjust(left=.12, bottom=.07, right=.99, top=.9, wspace=1.3, hspace=.17)
    spec = fig.add_gridspec(ncols=6, nrows=6, height_ratios=[1, 1, 1, 1, .4, 1])

    ax_char = []
    for i in range(8):
        if i % 2:
            ax_char.append(fig.add_subplot(spec[i//2, 3:]))
        else:
            ax_char.append(fig.add_subplot(spec[i//2, :3]))
    ax_char.append(fig.add_subplot(spec[5, 1:4]))
    ax_char = np.array(ax_char)
    ax_char[0].get_shared_x_axes().join(*ax_char)

    ax_char[0].set_title("Traction phase")
    ax_char[1].set_title("Retraction phase")
    ax_char[0].plot(df[x_col], df['x04']*1e-3, '>-', ms=5, label='Start')
    ax_char[0].plot(df[x_col], df['x08']*1e-3, '.-', label='End')
    ax_char[0].plot(df[x_col], np.amin(df.loc[:, 'x04':'x08'], axis=1)*1e-3, '--', label='Min.')
    ax_char[0].plot(df[x_col], np.amax(df.loc[:, 'x04':'x08'], axis=1)*1e-3, '--', label='Max.')
    ax_char[0].plot(df[x_col], [np.nan]*len(df[x_col]), label='Mean', lw=1)
    ax_char[0].axhline(sys_props_v3.tether_force_min_limit*1e-3, color='k', ls=':', label='Limit')
    ax_char[0].axhline(sys_props_v3.tether_force_max_limit*1e-3, color='k', ls=':')
    ax_char[0].set_ylabel('Tether force [kN]')
    ax_char[0].legend(bbox_to_anchor=(0.2, 1.35, 1.85, .5), loc="lower left", mode="expand",
                      borderaxespad=0, ncol=6)

    ax_char[1].plot(df[x_col], df['x09']*1e-3, '>-', ms=5)
    ax_char[1].plot(df[x_col], df['x18']*1e-3, '.-')
    ax_char[1].plot(df[x_col], np.amin(df.loc[:, 'x09':'x18'], axis=1)*1e-3, '--')
    ax_char[1].plot(df[x_col], np.amax(df.loc[:, 'x09':'x18'], axis=1)*1e-3, '--')
    ax_char[1].axhline(oc.bounds_real_scale[9, 0] * 1e-3, color='k', ls=':')
    ax_char[1].axhline(oc.bounds_real_scale[9, 1] * 1e-3, color='k', ls=':')

    ax_char[2].plot(df[x_col], df['ro_speed00'], '>-', ms=5)
    ax_char[2].plot(df[x_col], df['ro_speed04'], '.-')
    ax_char[2].plot(df[x_col], np.amin(df.loc[:, 'ro_speed00':'ro_speed04'], axis=1), '--')
    ax_char[2].plot(df[x_col], np.amax(df.loc[:, 'ro_speed00':'ro_speed04'], axis=1), '--')
    ax_char[2].axhline(sys_props_v3.reeling_speed_min_limit, color='k', ls=':')
    ax_char[2].axhline(sys_props_v3.reeling_speed_max_limit, color='k', ls=':')
    ax_char[2].set_ylabel('Reeling\nspeed [m/s]')
    ax_char[3].plot(df[x_col], df['ri_speed00'], '>-', ms=5)
    ax_char[3].plot(df[x_col], df['ri_speed09'], '.-')
    ax_char[3].plot(df[x_col], np.amin(df.loc[:, 'ri_speed00':'ri_speed09'], axis=1), '--')
    ax_char[3].plot(df[x_col], np.amax(df.loc[:, 'ri_speed00':'ri_speed09'], axis=1), '--')
    ax_char[3].axhline(-sys_props_v3.reeling_speed_max_limit, color='k', ls=':')

    ax_char[4].plot(df[x_col], df['ro_power00']*1e-3, '>-', ms=5)
    ax_char[4].plot(df[x_col], df['ro_power04']*1e-3, '.-')
    ax_char[4].plot(df[x_col], np.amin(df.loc[:, 'ro_power00':'ro_power04']*1e-3, axis=1), '--')
    ax_char[4].plot(df[x_col], np.amax(df.loc[:, 'ro_power00':'ro_power04']*1e-3, axis=1), '--')
    ax_char[4].plot(df[x_col], df['mean_ro_power']*1e-3, lw=1)
    # f_cubic = lambda x, a: a*x**3
    # a = df.loc[0, 'mean_ro_power']*1e-3/df.loc[0, x_lbl]**3
    # ax_char[4].plot(np.linspace(0, 15, 100), f_cubic(np.linspace(0, 15, 100), a))

    power_limit = min([sys_props_v3.reeling_speed_max_limit*sys_props_v3.tether_force_max_limit,
                       sys_props_v3.reeling_power_max_limit])
    ax_char[4].axhline(power_limit*1e-3, color='k', ls=':')
    ax_char[4].set_ylabel('Power [kW]')

    ax_char[5].plot(df[x_col], df['ri_power00']*1e-3, '>-', ms=5)
    ax_char[5].plot(df[x_col], df['ri_power09']*1e-3, '.-')
    ax_char[5].plot(df[x_col], np.amin(df.loc[:, 'ri_power00':'ri_power09']*1e-3, axis=1), '--')
    ax_char[5].plot(df[x_col], np.amax(df.loc[:, 'ri_power00':'ri_power09']*1e-3, axis=1), '--')
    ax_char[5].plot(df[x_col], df['mean_ri_power']*1e-3, lw=1)
    ax_char[5].axhline(-power_limit*1e-3, color='k', ls=':')

    ax_char[6].plot(df[x_col], df['tan_speed_ro00'], '>-', ms=5)
    ax_char[6].plot(df[x_col], df['tan_speed_ro04'], '.-')
    ax_char[6].plot(df[x_col], np.amin(df.loc[:, 'tan_speed_ro00':'tan_speed_ro04'], axis=1), '--')
    ax_char[6].plot(df[x_col], np.amax(df.loc[:, 'tan_speed_ro00':'tan_speed_ro04'], axis=1), '--')
    ax_char[6].plot(df[x_col], 300/df['x00'], ':', color='k')
    ax_char[6].set_ylabel('Tangential\nspeed [m/s]')
    ax_char[7].plot(df[x_col], df['tan_speed_ri00'], '>-', ms=5)
    ax_char[7].plot(df[x_col], df['tan_speed_ri09'], '.-')
    ax_char[7].plot(df[x_col], np.amin(df.loc[:, 'tan_speed_ri00':'tan_speed_ri09'], axis=1), '--')
    ax_char[7].plot(df[x_col], np.amax(df.loc[:, 'tan_speed_ri00':'tan_speed_ri09'], axis=1), '--')
    ax_char[7].axhline(0, color='k', ls=':')
    ax_char[8].plot(df[x_col], df['min_height_ro'], '>-', ms=5, label='Start traction')
    ax_char[8].plot(df[x_col], df['max_height_ro'], '.-', label='End traction')
    ax_char[8].plot(df[x_col], df['max_height_ri'], '--', color='C3', label='Max. retraction')
    ax_char[8].axhline(100, color='k', ls=':')
    ax_char[8].axhline(500, color='k', ls=':')
    ax_char[8].set_ylabel('Kite height [m]')
    ax_char[6].set_xlabel(x_label)
    ax_char[7].set_xlabel(x_label)
    ax_char[8].set_xlabel(x_label)

    mpp_curve_res = mpp_curve(env_state, beta_deg=None, vw_step=.5, plot=True)
    ax_vars[4].plot(mpp_curve_res[0], mpp_curve_res[1][:, 1]*180./np.pi, '-', color='C6', lw=1, label='Max. power point')
    ax_vars[4].legend(loc='upper left')
    # beta_deg = 20
    # mpp_curve_res = mpp_curve(env_state, beta_deg=beta_deg)
    ax_char[8].plot(mpp_curve_res[0], mpp_curve_res[2], '-', color='C6', lw=1, label='Max. power point')

    mpp_heights_elevated = []
    for idx, row in df.iterrows():
        oc = OptimizerReelOutState(sys_props_v3, env_state, elevation_angle=row['x02'], wind_speed=row[x_col])
        # if idx > 0:
        #     oc.x0_real_scale = x_opt
        x_opt = oc.optimize(maxiter=100)
        # try:
        p, vt, h = oc.eval_point(relax_errors=False)
        mpp_heights_elevated.append(h)
        # except FloatingPointError:
        #     mpp_heights_elevated.append(np.nan)
    ax_char[8].plot(df[x_col], mpp_heights_elevated, '*-', color='C5', label=r'Max. power point $\beta^*$')
    ax_char[8].legend(loc='upper left', bbox_to_anchor=(1, 1.15))

    for i, a in enumerate(ax_char):
        if i not in [3, 5]:
            a.set_ylim([0, None])
        else:
            a.set_ylim([None, 0])
        a.set_xticks(highlight_wind_speeds)
        if i >= 6:
            a.set_xticklabels(highlight_labels)
        else:
            a.set_xticklabels([])
        a.grid()
    add_panel_labels(ax_char, [.3, .20]*5)

    fig = plt.figure(figsize=[8, 2.6])
    plt.subplots_adjust(left=.115, right=.96, top=.95, bottom=.195, wspace=-0.1, hspace=.2)
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 3], height_ratios=[1])
    ax_profile = fig.add_subplot(spec[0, 0])
    ax_traj = fig.add_subplot(spec[0, 1])
    ax_profile.get_shared_y_axes().join(ax_profile, ax_traj)
    ax_traj.set_xlabel('Downwind position [m]')
    ax_traj.yaxis.set_ticklabels([])
    add_panel_labels(np.array([ax_profile, ax_traj]), [.5, .1])

    env_state.set_reference_wind_speed(1)
    env_state.plot_wind_profile(ax_profile)
    ax_profile.set_xlabel('Normalised wind speed [-]')
    ax_traj.set_aspect('equal')
    ax_traj.grid()

    for i_clr, i in enumerate(highlight_ith_opts):
        ax_traj.plot(df.loc[i, 'x_kite00':'x_kite04'], df.loc[i, 'z_kite00':'z_kite04'], '.-',
                     color='C{}'.format(i_clr), mfc='None', label='{:.1f}'.format(df.loc[i, x_col]))
        ax_traj.plot(df.loc[i, 'x_kite05':'x_kite14'], df.loc[i, 'z_kite05':'z_kite14'], '.-',
                     color='C{}'.format(i_clr), mfc='None')

        xf = df.iloc[i]['x_kite14']
        zf = df.iloc[i]['z_kite14']

        l = (xf**2 + zf**2)**.5
        beta_f = np.arctan(zf/xf)
        beta = np.linspace(df.iloc[i]['x02'], beta_f, 30)
        ax_traj.plot(np.cos(beta) * l, np.sin(beta) * l, color='C{}'.format(i_clr), lw=.5)
    ax_traj.set_xlim([0, None])
    ax_traj.set_ylim([0, None])
    ax_traj.legend(title=x_label, loc='upper left', bbox_to_anchor=(1, 1.05))


def plot_constraints(df):
    oc = OptimizerCycleKappa(sys_props_v3, env_state)

    n_opts = df.shape[0]
    n_eq_cons = 16
    ax = plt.subplots(3, n_eq_cons // 3 + 1)[1].reshape(-1)
    plt.subplots_adjust(left=.04, right=.99, wspace=.7)
    for i, hs in enumerate(df.loc[:, 'h000':'h{:03d}'.format(n_eq_cons-1)].values.T):
        ax[i].bar(range(n_opts), hs, color=np.where(np.abs(hs) > gtol, 'C3', 'C0'))
        ax[i].set_ylabel(oc.EQ_CONS_LABELS[i])

    n_ineq_cons = 56
    ax = plt.subplots(5, n_ineq_cons // 5 + 1)[1].reshape(-1)
    plt.subplots_adjust(left=.04, right=.99, wspace=.7)
    ineq_cons_labels = oc.INEQ_CONS_LABELS + ['Mean cycle power']
    for i, gs in enumerate(df.loc[:, 'g000':'g{:03d}'.format(n_ineq_cons-1)].values.T):
        ax[i].bar(range(n_opts), gs, color=np.where(gs < -gtol, 'C3', 'C0'))
        ax[i].set_ylabel(ineq_cons_labels[i])

    highlight_keys = [['Min. reel-out speed 1', 'Max. reel-out speed 1', 'Min. tangential speed out 1'],
                      ['Min. reel-out speed 5', 'Max. reel-out speed 5', 'Min. tangential speed out 5'],
                      ['Max. reel-in speed 1', 'Min. tangential speed in 1'],
                      ['Max. reel-in speed 10', 'Min. tangential speed in 10']]

    plot_nth_step = 3
    ax = plt.subplots(4, 4)[1]
    for a, ix in zip(ax[:, 0], [4, 8, 9, 18]):
        a.bar(range(n_opts), df['x{:02d}'.format(ix)], color='grey')
        for i_opt, xi in zip(range(n_opts), df['x{:02d}'.format(ix)]):
            if i_opt % plot_nth_step == 0:
                i_clr = i_opt // plot_nth_step
                a.bar(i_opt, xi, color='C{}'.format(i_clr))
        a.axhline(oc.bounds_real_scale[ix, 0], ls='--', color='k')
        a.axhline(oc.bounds_real_scale[ix, 1], ls='--', color='k')
        a.set_ylabel(oc.OPT_VARIABLE_LABELS[ix])
    for i, row_keys in enumerate(highlight_keys):
        for j, k in enumerate(row_keys):
            ki = ineq_cons_labels.index(k)
            gs = df['g{:03d}'.format(ki)]  # g_sol[:, ki]
            ax[i, j+1].bar(range(n_opts), gs, color=np.where(gs < -gtol, 'C3', 'C0'))
            ax[i, j+1].set_ylabel(k)


def run_opt_for_shapes(loc='mmc'):
    ax_profiles = plt.figure().gca()
    ax_power_curves = plt.figure().gca()

    if loc == 'mmc':
        roughness_length = 0.1
    reference_height = 200
    env_state = LogProfile(reference_height, roughness_length)
    env_state.plot_wind_profile(ax=ax_profiles)
    obj_factors_mcp = {'in': -2e-7, 'out': []}
    wind_speed_step = [0., .4]
    df = construct_power_curve(wind_speed_step, obj_factors_mcp=obj_factors_mcp, env_state=env_state,
                               export_file='opt_res_{}/opt_res_{}1.csv'.format(loc, loc))
    ax_power_curves.plot(df['vw200'], df['mcp'])

    h = [0., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
    hand_picked_shapes = np.load("hand_picked_shapes_{}.npy".format(loc))
    for i, vw_norm in enumerate(hand_picked_shapes):
        vw_norm[0] = 0.

        env_state = NormalisedWindTable1D(h, vw_norm)
        env_state.plot_wind_profile(ax=ax_profiles)
        df = construct_power_curve([0., .4], obj_factors_mcp={'in': 0, 'out': []}, env_state=env_state,
                                   export_file='opt_res_{}/opt_res_{}{}.csv'.format(loc, loc, i+2))
        ax_power_curves.plot(df['vw200'], df['mcp'])
    plt.show()


def compare_power_curves_tether_diameters():
    #TODO: adjust roughness length for mmij
    f_mmc = np.load('density_vw200m_mmc.npy')
    f_mmij = np.load('density_vw200m_mmij.npy')
    n_bins = 50
    w_bin = 30/n_bins
    vw_200m_bin_edges = np.linspace(0, 30, n_bins+1)
    vw_200m_bin_centers = (vw_200m_bin_edges[1:] + vw_200m_bin_edges[:-1])/2
    
    roughness_length = .1
    reference_height = 200
    env_state = LogProfile(reference_height, roughness_length)

    diameters = [0.006]  #0.006, 0.007]  #[0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]  #
    tether_force_max_limit = {
        0.004: 19.6/5*1e3,
        0.005: 28.4/5*1e3,
        0.006: 40.8/5*1e3,
        # 0.0065: (40.8+54.)/10*1e3,
        0.007: 54./5*1e3,
        0.008: 65.5/5*1e3,
        0.009: 87.3/5*1e3,
        0.01: 105/5*1e3,
    }

    ax = plt.subplots(2, 1, sharex=True)[1]
    p_avg = np.zeros((2, len(diameters)))
    for i, d in enumerate(diameters):
        sys_props_v3.tether_diameter = d
        sys_props_v3.tether_force_max_limit = tether_force_max_limit[d]
        if sys_props_v3.tether_diameter == 0.008:
            obj_factors_mcp = {'in': 0, 'out': -5e-6}
            wind_speed_step = [.01, 1.]
        elif sys_props_v3.tether_diameter == 0.006:
            obj_factors_mcp = {'in': -2e-7, 'out': -1e-6}
            wind_speed_step = [0.01] + [1.]
        else:
            obj_factors_mcp = {'in': -1e-7, 'out': -5e-6}
            wind_speed_step = [.01, 1.]

        # elif sys_props_v3.tether_diameter == 0.01:  # starting wind speed 11 m/s
        #     obj_factors_mcp = {'in': -1e-7, 'out': -5e-6}
        #     wind_speed_step = [.07] + [1.]
        # else:
        #     obj_factors_mcp = {'in': 0., 'out': -5e-6}
        #     wind_speed_step = [.01, 1.]

        df = construct_power_curve(wind_speed_step, obj_factors_mcp=obj_factors_mcp, env_state=env_state)
        plot_curves_and_trajectories(df, env_state, reference_height)
        ax[0].plot(df['vw200'], df['mcp']*1e-3, label="{:.0f}mm".format(d*1e3))

        p_bin_centers = np.interp(vw_200m_bin_centers, df['vw200'], df['mcp'], left=0, right=0)
        p_avg_mmc = np.sum(f_mmc*p_bin_centers*w_bin)
        p_avg_mmij = np.sum(f_mmij*p_bin_centers*w_bin)
        print("Cabauw/{:.0f}mm {:.1f}kW".format(d*1e3, p_avg_mmc*1e-3))
        print("IJmuiden/{:.0f}mm {:.1f}kW".format(d*1e3, p_avg_mmij*1e-3))
        p_avg[0, i] = p_avg_mmc
        p_avg[1, i] = p_avg_mmij

    ax[1].plot(vw_200m_bin_centers, f_mmc, label='Cabauw')
    ax[1].plot(vw_200m_bin_centers, f_mmij, label='IJmuiden')

    ax[1].set_xlabel("Wind speed @ 200m [m/s]")
    ax[0].set_ylabel("Mean cycle power [kW]")
    ax[1].set_ylabel("Probability [-]")
    ax[0].set_xlim([0, None])
    for a in ax:
        a.set_ylim([0, None])
        a.grid()
        a.legend()

    plt.figure()
    plt.plot([d*1e3 for d in diameters], p_avg[0, :]*1e-3, label='Cabauw')
    plt.plot([d*1e3 for d in diameters], p_avg[1, :]*1e-3, label='IJmuiden')
    plt.xlabel('Tether diameter [mm]')
    plt.ylabel('Annual average power [kW]')
    plt.grid()
    plt.legend()
    plt.show()


def compare_power_curves_power_limits():
    f_mmc = np.load('density_vw200m_mmc.npy')
    f_mmij = np.load('density_vw200m_mmij.npy')
    n_bins = 50
    w_bin = 30 / n_bins
    vw_200m_bin_edges = np.linspace(0, 30, n_bins + 1)
    vw_200m_bin_centers = (vw_200m_bin_edges[1:] + vw_200m_bin_edges[:-1]) / 2

    roughness_length = .1
    reference_height = 200
    env_state = LogProfile(reference_height, roughness_length)

    power_limits = [40e3, 60e3, 80e3]

    ax = plt.subplots(2, 1, sharex=True)[1]
    p_avg = np.zeros((2, len(power_limits)))
    for i, p_max in enumerate(power_limits):
        sys_props_v3.reeling_power_max_limit = p_max
        obj_factors_mcp = {'in': -2e-7, 'out': -1e-6}
        wind_speed_step = [0.] + [1.5]

        df = construct_power_curve(wind_speed_step, obj_factors_mcp=obj_factors_mcp, env_state=env_state)
        ax[0].plot(df['vw200'], df['mcp'] * 1e-3, label="{:.0f}kW".format(p_max*1e-3))

        p_bin_centers = np.interp(vw_200m_bin_centers, df['vw200'], df['mcp'], left=0, right=0)
        p_avg_mmc = np.sum(f_mmc * p_bin_centers * w_bin)
        p_avg_mmij = np.sum(f_mmij * p_bin_centers * w_bin)
        print("Cabauw/{:.0f}kW {:.1f}kW".format(p_max*1e-3, p_avg_mmc * 1e-3))
        print("IJmuiden/{:.0f}kW {:.1f}kW".format(p_max*1e-3, p_avg_mmij * 1e-3))
        p_avg[0, i] = p_avg_mmc
        p_avg[1, i] = p_avg_mmij

    ax[1].plot(vw_200m_bin_centers, f_mmc, label='Cabauw')
    ax[1].plot(vw_200m_bin_centers, f_mmij, label='IJmuiden')

    ax[1].set_xlabel("Wind speed @ 200m [m/s]")
    ax[0].set_ylabel("Mean cycle power [kW]")
    ax[1].set_ylabel("Probability [-]")
    ax[0].set_xlim([0, None])
    for a in ax:
        a.set_ylim([0, None])
        a.grid()
        a.legend()

    plt.figure()
    plt.plot([p_max*1e-3 for p_max in power_limits], p_avg[0, :]/p_avg[0, -1]*100, 's-', label='Cabauw')
    plt.plot([p_max*1e-3 for p_max in power_limits], p_avg[1, :]/p_avg[1, -1]*100, 's-', label='IJmuiden')
    plt.xlabel('Reel-out power limit [kW]')
    plt.ylabel('Annual average power [%]')
    plt.grid()
    plt.legend()
    plt.show()


# def redo_last_optimization():





if __name__ == "__main__":
    compare_power_curves_tether_diameters()
    # compare_power_curves_power_limits()
    exit()
    for roughness_length in [.1]:  #[.1]:  #, .0002]:
        reference_height = 200
        env_state = LogProfile(reference_height, roughness_length)
        # env_state = NormalisedWindTable1D()
        # h = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
        # v = [0.31916794, 0.38611869, 0.55029902, 0.65804212, 0.73519261, 0.79609015,
        #      0.84480401, 0.88139352, 0.89594957, 0.90781971, 0.92918228, 0.94297737,
        #      0.95193377, 0.9588552, 0.95571666, 0.88390919, 0.84849594]
        # env_state.heights = h
        # env_state.h_ref = h[np.argmax(v)]
        # env_state.normalised_wind_speeds = v / np.amax(v)
        # env_state.plot_wind_profile()
        # plt.show()

        # find_cut_speed(x0=[1.13764272e+02, 6.08751315e+01, 9.48468048e-01, 2.49842316e+02,
        #                    5.00000000e+03, 5.00000000e+03, 5.00000000e+03, 2.90299356e+03,
        #                    5.00000000e+03, 5.00000000e+03, 5.00000000e+03, 5.00000000e+03,
        #                    5.00000000e+03, 2.81732604e+03, 2.57137223e+03, 2.57137223e+03,
        #                    2.57137223e+03, 2.57137223e+03, 2.57137223e+03, 2.95622546e+00,
        #                    2.70502689e+00, 2.53066712e+00, 2.15106589e+00, 2.29456606e+00,
        #                    1.62580871e+00, 1.70031843e+00, 1.79227046e+00, 1.90501750e+00,
        #                    1.99054171e+00, 2.08459706e+00, 2.18167218e+00, 2.28132096e+00,
        #                    2.38136934e+00, 2.47913410e+00, 2.28599887e+01], env_state=env_state)
        # # Log profile
        # eval_limit(wind_speeds=[22, 22.5, 22.7, 22.9], cut='out')
        # eval_limit(wind_speeds=[7, 6.5, 6.4], cut='in')
        # eval_limit(wind_speeds=np.linspace(7, 22, 6))

        # eval_limit(wind_speeds=[9, 8.5], cut='in')
        # eval_limit(env_state=env_state, wind_speeds=np.arange(25, 26.5, .5), cut='out')
        # eval_limit(env_state=LogProfile(), wind_speeds=np.arange(21, 23., .5), cut='out')

        # mpp_curve_res = mpp_curve(env_state, beta_deg=None, vw_step=.5, plot=True)

        if sys_props_v3.tether_diameter == 0.004:
            if roughness_length == .1:
                obj_factors_mcp = {'in': -1e-7, 'out': -5e-6}
                wind_speed_step = [.06, 1.]
            else:
                obj_factors_mcp = {'in': -2e-7, 'out': -5e-6}
                wind_speed_step = [.01, 1.]
        elif sys_props_v3.tether_diameter == 0.01:  # starting wind speed 11 m/s
            if roughness_length == .1:
                obj_factors_mcp = {'in': -1e-7, 'out': -5e-6}
                wind_speed_step = [.07] + [1.]
        else:
            if roughness_length == .1:
                obj_factors_mcp = {'in': -2e-7, 'out': -5e-6}
                wind_speed_step = [0.] + [1.]
        df = construct_power_curve(wind_speed_step, obj_factors_mcp=obj_factors_mcp, env_state=env_state)  #, export_file='opt_res_log_profile.csv')  #, power_optimization_limits=26)  #, export_file='opt_res_llj_profile2.csv')
        # df = pd.read_csv('opt_res_log_profile.csv')
        plot_curves_and_trajectories(df, env_state, reference_height)
        print('Max. power = {:.3f} kW'.format(df['mcp'].max()*1e-3))
    plt.show()
