from sys import getdefaultencoding
from NERDA.models import NERDA
from NERDA.datasets import get_dane_data
from hyperopt import fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
import numpy as np

def objective(params):
    
    print(params)

    model = NERDA(dataset_training = get_dane_data('train', 20),
                  dataset_validation = get_dane_data('dev', 20),
                  hyperparameters = params)

    model.train()

    return model.valid_loss

def run_parameter_optimization(objective, number_of_evals = 3):
   
    hpspace = {
            'learning_rate': hp.loguniform('lr', np.log(0.00005), np.log(0.01)),
            'train_batch_size': scope.int(hp.uniform('batch_size', 8, 16)),
            'epochs': scope.int(hp.uniform('epochs', 1, 3)),
            'warmup_steps': hp.choice('warmup_steps', [0, 250, 500]),
        }

    print('Running hyperparameter optimization...')

    best_params = fmin(objective, space = hpspace, algo = tpe.suggest, max_evals= number_of_evals)

    return best_params

# best_params = run_parameter_optimization(objective = objective, number_of_evals=3)
