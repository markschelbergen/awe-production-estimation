from qsm import SystemProperties

# from numpy import pi
# rho = 1.225
# g = 9.81
# d_t = .01
# rho_t = 724.
# cd_t = 1.1
# tether_modulus = 614600/(pi*.002**2)  # From Uwe's thesis: 48.9 GPa
# tether_stiffness = tether_modulus*pi*(d_t/2)**2  # 3,841,250
# # tether_stiffness = 490000  # From Williams

m_canopy = 11 + .7 + .7 + 1.8
m_kcu = 25
m_kite = m_canopy + m_kcu
l_bridle = 11.5

cd_kcu = 1.
frontal_area_kcu = .25

# Kitepower's V3.
kite_projected_area = 19.75
sys_props_v3 = {
    'kite_projected_area': kite_projected_area,  # [m^2] - 25 m^2 total flat area
    'kite_mass': m_kite,  # [kg]
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.006,  #0.01,  # [m]
    'tether_force_min_limit': 750,  # ~ min_wing_loading * projected_area [N]
    'kite_lift_coefficient_powered': 0.70,  # [-] - in the range of .9 - 1.0
    'kite_drag_coefficient_powered': 0.16 + cd_kcu * frontal_area_kcu/kite_projected_area,  # [-]
    'kite_lift_coefficient_depowered': .22,
    'kite_drag_coefficient_depowered': .11 + cd_kcu * frontal_area_kcu/kite_projected_area,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 0,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
    'reeling_power_max_limit': 40e3,  # [W]
}
tether_force_max_limit = {
    0.004: 19.6/5*1e3,
    0.005: 28.4/5*1e3,
    0.006: 40.8/5*1e3,
    0.007: 54./5*1e3,
    0.008: 65.5/5*1e3,
    0.009: 87.3/5*1e3,
    0.01: 105/5*1e3,
}
sys_props_v3['tether_force_max_limit'] = tether_force_max_limit[sys_props_v3['tether_diameter']]
sys_props_v3 = SystemProperties(sys_props_v3)
