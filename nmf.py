# TODO: method function need SAVE flag
# TODO: maybe add a check for METHOD_PARAMS, that only valid params are included?

import os
from importlib import import_module

from utils import save_results


class NMF:
    """ Docstring """

    def __init__(self, data=None, factors=None, saving=True, param_file=None):
        self.data = data
        self.factors = factors
        self.saving = saving
        if param_file is not None:
            try:
                parameters = import_module(param_file)
                self.method_params = parameters.method_params
            except ImportError:
                print('No parameter file found.')
                return


    def factorize(self, method='mur', saving=False, **method_params):
        """ DOCSTRING """

        if method == 'mur':
            import mur
            self.results = mur.mur(self.data, self.factors, **method_params)
        
        elif method == 'anls':
            import anls
            self.results = anls.anls(self.data, self.factors, **method_params)

        elif method == 'admm':
            import admm
            self.results = admm.admm(self.data, self.factors, **method_params)

        elif method == 'ao_admm':
            import ao_admm
            self.results = ao_admm.ao_admm(self.data, self.factors, **method_params)

        else:
            raise Exception('Method not known. Choose one from: mur anls admm ao_admm')
    
        print('Factorization done.')
        if saving:
            self.save_factorization() 


    def save_factorization(self, save_dir='./results', save_name=None):
        """ DOCSTRING """
        # TODO: save_name needs to include more parameters and NNDSVD
    
        # create results folder, if not existing
        os.makedirs(save_dir, exist_ok=True)

        # generate standard save_name from parameters
        if save_name is None:
            save_name = f'nmf_{self.results.experiment.method}'
            save_name += f'_{self.factors}'
            save_name += f'_{self.results.experiment.distance_type}'
            if self.results.experiment.method == 'admm':
                save_name += f'_{self.results.experiment.rho}'
            
            save_name += f'_{self.results.experiment.lambda_w}'
            if self.results.experiment.method in {'admm', 'ao_admm'}:
                save_name += f':{self.results.experiment.prox_w}'
            
            save_name += f'_{self.results.experiment.lambda_h}'
            if self.results.experiment.method in {'admm', 'ao_admm'}:
                save_name += f':{self.results.experiment.prox_h}'
        
            if self.results.experiment.nndsvd_init[0]:
                save_name += f'_nndsvd{self.results.experiment.nndsvd_init[1][0]}'
            else:
                save_name += '_random'

            if self.results.experiment.method == 'anls' and self.results.experiment.fcnnls:
                save_name += '_fcnnls'

        # create save string including folder and save
        save_str = os.path.join(save_dir, save_name)
        save_results(save_str,
                     w = self.results.w,
                     h = self.results.h,
                     i = self.results.i,
                     obj_history = self.results.obj_history,
                     experiment = self.results.experiment)