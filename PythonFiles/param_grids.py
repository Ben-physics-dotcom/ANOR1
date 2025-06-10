# nn sklearn:
# param grid scikit
param_grid = {
    'hidden_layer_sizes': [(h,) for h in [5, 10, 20, 50, 100, 200, 300, 406]],
    'activation': ['relu', 'tanh', 'logistic'],                   # activation functions
    'solver': ['adam', 'sgd'],                                    # optimization algorithms
    'alpha': [0.0001, 0.001, 0.01, 0.1],                                # L2 regularization
    'learning_rate': ['constant', 'adaptive'],                    # learning rate strategy
    'learning_rate_init': [0.001, 0.005, 0.01, 0.05],                           # initial learning rate
    'max_iter': [100, 200, 500, 750, 1000]                                        # training epochs
}

# param grid keras:
param_grid = {
    "hidden_layer_size": [16, 32, 64, 128, 200, 256, 300, 350, 400, 406],
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "sgd"],
    "learning_rate": [0.001, 0.005, 0.01, 0.05],
    "batch_size": [16, 32, 64],
    "epochs": [10, 20, 30]
}

# ebm
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_leaves': [3, 5, 10, 15, 20, 32],
    'min_samples_leaf': [2, 5, 10, 15, 20],
    'interactions': [5, 10, 15, 20]
}
