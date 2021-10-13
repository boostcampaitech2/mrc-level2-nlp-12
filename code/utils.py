from ray import tune

def ray_hp_space(trial):
    '''A function for returning ranges of hyperparameters for ray tune search

        Returns:
            parameter (Dict): Ranges of parameters to search best value
    '''
    return {
        "learning_rate": tune.loguniform(5e-5, 5e-4),
        "num_train_epochs": tune.choice(range(5, 7)),
        "per_device_train_batch_size": tune.choice([32, 50, 64]),
        # "seed": tune.choice(range(10, 43)),
    }
