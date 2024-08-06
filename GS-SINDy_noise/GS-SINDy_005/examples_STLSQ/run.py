#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:39:15 2024

@author: dliu
"""

import subprocess
import os

os.chdir('Exp1_Lotka_Volterra')
subprocess.run(['python', "./Lotka_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(['python', "./Lotka_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')

os.chdir('Exp2_Modified_Lotka_Volterra')
subprocess.run(['python', "./M_Lotka_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(['python', "./M_Lotka_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')

os.chdir('Exp3_Brusselator')
subprocess.run(['python', "./Brusselator_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(['python', "./Brusselator_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')

os.chdir('Exp4_Van_der_Pol')
subprocess.run(["python", "./Van_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(["python", "./Van_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')

os.chdir('Exp5_Lorenz')
subprocess.run(["python", "./Lorenz_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(["python", "./Lorenz_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')

os.chdir('Exp7_FitzHugh')
subprocess.run(["python", "./FitzHugh_gsindy.py"], capture_output=True, text=True, check=True)
subprocess.run(["python", "./FitzHugh_sindy.py"], capture_output=True, text=True, check=True)
os.chdir('..')
