
from models import *


def config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=False, 
                                       train_weights=True, weight_sharing=True):
    for h in [30, 20, 10]:
        for l in [20, 10, 5, 1]:
            for e in [1., 1e-1, 1e-2, 1e-3]:
                    for g in [1., 1e-1, 1e-2, 1e-3]:
                        yield {
                            'model': {
                                'input_dim': num_features,
                                'output_dim': num_classes,
                                'hidden_dim': h,
                                'num_layers': l,
                                'epsilon': e,
                                'gamma': g,
                                'activ_fun': 'tanh',
                                'gcn_norm': gcn_norm,
                                'train_weights': train_weights, 
                                'weight_sharing': weight_sharing
                            },
                            'optim': {
                                'lr': 0.003,
                                'weight_decay': 1e-6
                            }
                        }


# def config_dgn_GraphProp(num_features, num_classes, conv):
#     for h in [30, 20, 10]:
#         for num_layers in [20, 10, 5, 1]:
#                 yield {
#                     'model': {
#                         'input_dim': num_features,
#                         'output_dim': num_classes,
#                         'hidden_dim': h,
#                         'num_layers': num_layers,
#                         'conv_layer': conv
#                     },
#                     'optim': {
#                         'lr': 0.003,
#                         'weight_decay': 1e-6
#                     }
#                 }


def config_dgn_GraphProp(num_features, num_classes, conv):
    for h in [40]:
        for num_layers in [4,5,6]:
          for Kop in [4]: 
            for epsil in [0.7,0.8]: 
                for sigm in [0.01]: 
                    for lr in [0.003, 0.01]:
                        yield {
                            'model': {
                                'input_dim': num_features,
                                'K':Kop,
                                'epsilon':epsil,
                                'dissipation_force':sigm,
                                'output_dim': num_classes,
                                'hidden_dim': h,
                                'num_layers': num_layers,
                                'conv_layer': conv,
                                
                            },
                            'optim': {
                                'lr': lr,
                                'weight_decay': 1e-6
                            }
                }


def config_gcn2_GraphProp(num_features, num_classes, conv):
    for h in [30, 20, 10]:
        for num_layers in [20, 10, 5, 1]:
            for alpha in [1., 1e-1, 1e-2]:
                yield {
                    'model': {
                        'input_dim': num_features,
                        'output_dim': num_classes,
                        'hidden_dim': h,
                        'num_layers': num_layers,
                        'conv_layer': conv,
                        'alpha': alpha
                    },
                    'optim': {
                        'lr': 0.003,
                        'weight_decay': 1e-6
                    }
                }


def config_ODE_GraphProp(num_features, num_classes):
    for h in [30, 20, 10]:
        for num_layers in [20, 10, 5, 1]:
            for e in [1., 1e-1, 1e-2, 1e-3]:
                conf = {
                    'model': {
                        'input_dim': num_features,
                        'output_dim': num_classes,
                        'hidden_dim': h,
                        'epsilon': e,
                        'iterations':num_layers,
                        'cached': False
                    },
                    'optim': {
                        'lr': 0.003,
                        'weight_decay': 1e-6
                    }
                }
                yield conf


c0 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GINConv')
c1 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GCNConv')
c2 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'SAGEConv')
c3 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GATConv')
c12 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'ChebConv')
c13 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'ChebConvDis')
c15 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'NonDisCheb')
c14 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'Euler')
c16 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'Euler2')
c4 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=True)
c5 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes)
c6 = lambda num_features, num_classes: config_ODE_GraphProp(num_features, num_classes)
c7 = lambda num_features, num_classes: config_gcn2_GraphProp(num_features, num_classes, 'GCN2Conv')
c8 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=True, train_weights=False)
c9 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, train_weights=False)
c10 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=True, weight_sharing=False)
c11 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, weight_sharing=False)
CONFIGS = {
    'GIN_GraphProp': (c0, DGN_GraphProp),
    'GCN_GraphProp': (c1, DGN_GraphProp),
    'SAGE_GraphProp': (c2, DGN_GraphProp),
    'GAT_GraphProp': (c3, DGN_GraphProp),
    'Cheb_GraphProp': (c12, DGN_GraphProp),
    'ChebDis_GraphProp': (c13, DGN_GraphProp),
    'Euler_GraphProp': (c14, DGN_GraphProp),
    'Euler2_GraphProp': (c16, DGN_GraphProp),
    'NonDisv2_GraphProp': (c15, DGN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_gcnnorm_GraphProp': (c4, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_GraphProp': (c5, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_gcnnorm_randomized_GraphProp': (c8, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_randomized_GraphProp': (c9, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_layer_dependent_weights_gcnnorm_GraphProp': (c10, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_layer_dependent_weights_GraphProp': (c11, GraphAntiSymmetricNN_GraphProp),
    'DGC_GraphProp': (c6, DGC_GraphProp),
    'GRAND_GraphProp': (c6, GRAND_GraphProp),
    'GCN2_GraphProp': (c7, DGN_GraphProp)
}

