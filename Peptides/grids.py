merged_grids = {
    'hidden_dim': [30, 20],
    'num_layers': [1, 2, 3, 4, 5],
    'K': [3,5,10], 
    'step_size': [0.1, 0.01, 0.1, 0.2, 0.3],
    'dissipation_force': [0., 0.01, 0.5, 1.],
    'activ_fun': ['tanh', 'relu'], 

    'lr': 0.003,
    'weight_decay': 1e-6
}

best_grid = {
    'hidden_dim': [30], 
    'num_layers': [2, 5],
    'K': [5, 10], 
    'step_size': [0.1, 0.2, 0.3],
    'dissipation_force': [0., 0.01, 1., 0.5],
    'activ_fun': ['relu'], 
}

winning_config_ecc = {
    'hidden_dim': 30,
    'num_layers': 5, 
    'K': 10,  
    'step_size': 0.3,  
    'dissipation_force': 0.0,  
    'activ_fun': relu  
}

winning_config_sssp = {
    'hidden_dim': 30,
    'num_layers': 5, 
    'K': 10,  
    'step_size': 0.3,  
    'dissipation_force': 0.0,  
    'activ_fun': relu  
}

winning_config_diam = {
    'hidden_dim': 30,
    'num_layers': 5, 
    'K': 10,  
    'step_size': 0.3,  
    'dissipation_force': 1.0,  
    'activ_fun': relu  
}