
import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../Exp1_Lotka_Volterra')
sys.path.insert(1, '../Exp2_Modified_Lotka_Volterra')
sys.path.insert(1, '../Exp3_Brusselator')
sys.path.insert(1, '../Exp4_Van_der_Pol')
sys.path.insert(1, '../Exp5_Lorenz')
sys.path.insert(1, '../Exp6_Pendulum')

import numpy as np
import matplotlib.pyplot as plt
from GSINDy import SLS
from utils import ode_solver, get_deriv, get_theta
from utils import func_Lotka_Voltera, func_M_Lotka_Voltera, func_Brusselator, func_Van_der_Pol, func_Lorenz, func_Pendulum
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5

from M_Lotka_constants import get_basis_functions
import M_Lotka_constants as constants
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


########## hyper parameters ###########
ensemble = constants.ensemble
precision = constants.precision
deriv_spline = constants.deriv_spline
alpha = constants.alpha
threshold_sindy_list = constants.threshold_sindy_list

########## function variable ###########
t = constants.t
x0_list = constants.x0_list
a_list = constants.a_list

func = constants.func
real_list = constants.real_list

########## basis functions and optimizer ###########
basis_type = constants.basis_type
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=True)

basis_functions_list = basis['functions']

threshold_sindy = .01


######################################################
################## get data ##########################
######################################################
x0 = x0_list[0]
a = a_list[0]

sol_, t_ = ode_solver(func, x0, t, a)
_, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)


#######################################################
###################### SLS ############################
#######################################################
Theta0 = get_theta(sol_, basis_functions_list[0])
Theta1 = get_theta(sol_, basis_functions_list[1])
DXdt0 = sol_deriv_[:,[0]]
DXdt1 = sol_deriv_[:,[1]]

num_feature, num_basis = sol_deriv_.shape[1], Theta0.shape[1]
Xi = np.zeros([num_feature, num_basis])

Xi[0,:] = SLS(Theta0, DXdt0, threshold_sindy)[...,0]
Xi[1,:] = SLS(Theta1, DXdt1, threshold_sindy)[...,0]



###########################################################
###################### pysindy ############################
###########################################################
import pysindy_ as ps
from pysindy_.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary

basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=False)
basis_functions_list = basis['functions']
basis_functions_name_list = basis['names']

assert (basis_functions_list[0]==basis_functions_list[1]).all(), 'pysindy does not support different features with different basis functions'

basis_functions = basis_functions_list[0]
basis_functions_name = basis_functions_name_list[0]
    
lib_custom = CustomLibrary(library_functions=basis_functions, function_names=basis_functions_name)
lib_generalized = GeneralizedLibrary([lib_custom])

if opt=='SQTL':
    optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=alpha)
elif opt=='SR3':
    optimizer = ps.SR3(threshold=threshold_sindy, nu=.1)

model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
    
### sindy
model.fit(sol_, t=t_, x_dot=sol_deriv_, ensemble=ensemble, quiet=True)


###########################################################
################ compare SLS with pysindy #################
###########################################################
if (model.coefficients()==Xi).all():
    print('SLS is the same as SQTL')
else:
    print('SLS is different from SQTL')
    