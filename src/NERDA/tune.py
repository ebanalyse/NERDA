from .train import train_model

from hyperopt import fmin, hp, tpe, space_eval
from hyperopt.pyll import scope

def objective(params):
    print(params)

    loss, epoch = train_model(learning_rate= params['learning_rate'],
                              train_batch_size= params['train_batch_size'],
                              epochs= params['epochs'],
                              warmup_steps= params['warmup_steps'], 
                              test_run_size = 100,
                              return_valid_loss= True)

    return loss

def run_parameter_optimization(number_of_evals = 20):
    """
    Runs a hyperparameter optimization search, for an optimal combination of hyperparameters. 
    Current version only contains a selection of the model parameters, more can be added if needed or desired. 

    Args:
        number_of_evals ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    hpspace = {
            'learning_rate': hp.loguniform('lr', np.log(0.00005), np.log(0.01)),
            'train_batch_size': scope.int(hp.uniform('batch_size', 8, 16)),
            'epochs': scope.int(hp.uniform('epochs', 5, 10)),
            'warmup_steps': hp.choice('warmup_steps', [0, 100, 250, 500, 1000, 2000]),
        }

    print('Running hyperparameter optimization...')

    best = fmin(objective, space = hpspace, algo = tpe.suggest, max_evals= number_of_evals)

    return best

if __name__ == '__main__':
    print(run_parameter_optimization(number_of_evals=5))
