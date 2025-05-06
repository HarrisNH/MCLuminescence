# MCLuminescence
Luminescence Monte Carlo simulations

- It is suggested using the class folder, when running the code.
- Simulate.py is used to run simulation, where parameters are set, and saves a .csv file with results at results/simultations/
- Parameters are set through conf/config_fp.yaml and the underlying folders exp_type_fp and physics_fp.
In these .yaml files you can change the specific configuration of the simulation such as dose, trap depth, alpha values etc. 

- Optimizer.py is used to optimize the parameters of the simulation, but that requires actually having lab data to use for MSE caculations.

- src/est_params/ is the original folder with parameter estimation,
before the code was refactored into object oriented code. 

- Feel free to reach out if you have any questions or suggestions.