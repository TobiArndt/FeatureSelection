import pandas as pd
from sklearn.model_selection import train_test_split
from construction import ExperimentTrie
from Experiments.pte_experiment import ExperimentExecuter
#from Models.definitions_simple import model_generator, cv_rmse
from Models.definitions_internal_optimisation import get_lgbm
from Objectives.definitions_kf import cv_rmse
from construction import A
from SearchStrategies.pte import Search
from PreprocessingAlgorithms.Utils.optimizer_utils import SuggestFloat, SuggestLogUniform

import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':
    data = pd.read_csv('df_all_merged.csv', index_col=0)
    data = data.drop(['Stücknummer', 'FELT_NUMBER'], axis=1)
    data = data[data['FELT_LIFE_NET'].notna()]
    print(data.columns)
    print(data.shape)


    categorical_cols_labrep = ['ID',
                           'FELT_NUMBER']

    numerical_cols_labrep = ['CONTAMINATION',
                         'CONTAMINATION_TOTAL',
                         'CONTAMINATION_RESIN',
                         'CONTAMINATION_PAPER_STOCK',
                         'CONTAMINATION_ASH',
                         'TENSILE_STRENGTH_NEW',
                         'TENSILE_STRENGTH_RUNNING',
                         'ELONGATION_NEW',
                         'ELONGATION_RUNNING',
                         'AVERAGE_AIR_PERM',
                         'AVERAGE_WEIGHT',
                         'AVERAGE_THICKNESS',
                         'TENSILE_STRENGTH_CROSS_NEW',
                         'TENSILE_STRENGTH_CROSS_RUNNING',
                         'WASHING',
                         'WEAR',
                         'AVERAGE_WIDTH',
                         'FELT_LIFE_NET',
                         'PRODCT_EFFECTIVE_AIR_PERM',
                         'PRODCT_EFFECTIVE_THICKNESS',
                         'PRODCT_EFFECTIVE_M2_WEIGHT',
                         'ASH_SOLUBLE',
                         'ASH_INSOLUBLE',
                         'SEAM_STRENGTH_NEW',
                         'SEAM_STRENGTH_RUNNING']


    categorical_cols_s1 = [
                    'Produktgruppe Vertrieb',
                    'Träger', 
                    'Auflage', 
                    'Benadelungstechnik',
                    'Oberflächenbehandlung',
                    'Zweckbehandlung',
                    'EA Fixiervorschrift',
                    'SD Maschinenbez.',
                    'SD Einsatzst.-bez.']

    numerical_cols_s1 = ['NA Einstiche pro cm2',
                  'Auflage Gewicht sum.',
                  'SD Soll-Länge',
                  'SD Soll-Breite',
                  'MDS Flächengewicht',
                  'PF Flächengewicht',
                  'PF Flächengew. Type',
                  'NA LD',
                  'NA LD Typenstamm',
                  'NA Dicke',
                  'NA Dicke Soll',
                  'EA LD Textest',
                  'EA LD Text.Typenst.',
                  'NA Kerbentiefe',
                  'Aktive Oberfläche',
                  'EA Abw Länge Soll',
                  'EA Fixierspannung',
                  'EA Breite n. Schm.',
                  'Nadelvorschrift']
    

    train, test, y_train, y_test = train_test_split(data, data['FELT_LIFE_NET'], test_size=0.3, random_state=1)
    print(train.shape)
    
    train_num = train[numerical_cols_s1 + numerical_cols_labrep]
    print(train_num.shape)
    print(train_num.columns)
    '''
    config = [
                [
                    (A.Feature_ColumnDropper,{'columns':['ELONGATION_RUNNING', 'SEAM_STRENGTH_RUNNING', 'TENSILE_STRENGTH_CROSS_RUNNING']}), 
                    (A.Optional,)
                ],
                [
                    #(A.Feature_Variance,{'thresh': SuggestLogUniform(0.03, 0.2)}), 
                    (A.Feature_Variance, {'thresh': 0.1}),
                    (A.Optional,)
                ],
                [
                    (A.Distribution_BoxCox,{'skewing_threshold': 0.3}), 
                    (A.Optional,)
                ],
                [
                    #(A.Correlation_VIF,{'vif_thresh': SuggestFloat(4.0, 10.0)}),
                    (A.Correlation_VIF,{'vif_thresh': 5.0}),
                ],
                [
                    (A.Outlier_IQR,), 
                    (A.Outlier_LOF,), 
                    (A.Optional,)
                ],
                [   
                    (A.Imputer_Iterative,{'estimator': 'BayesianRidge', 'tolerance': 1e-3, 'max_iter': 25}), 
                    (A.Optional,)
                ]
            ]
    '''

    config = [
                [
                    (A.Feature_ColumnDropper,{'columns':['ELONGATION_RUNNING', 'SEAM_STRENGTH_RUNNING', 'TENSILE_STRENGTH_CROSS_RUNNING']}), 
                    (A.Optional,)
                ],
                [
                    (A.Feature_Variance,{'thresh': 0.3}), 
                    (A.Optional,)
                ],
                [
                    (A.Distribution_BoxCox,{'skewing_threshold': 0.3}), 
                    (A.Optional,)
                ],
                [
                    (A.Correlation_VIF,{'vif_thresh': 5}),
                ],
                [
                    (A.Outlier_IQR,), 
                    (A.Outlier_LOF,), 
                    (A.Optional,)
                ],
                [   
                    (A.Imputer_Iterative,{'estimator': 'BayesianRidge', 'tolerance': 1e-3, 'max_iter': 25}), 
                    (A.Optional,)
                ]
            ]

    experiment_handler = ExperimentExecuter(
        target='FELT_LIFE_NET',
        objective_fn=cv_rmse)
    trie = ExperimentTrie(name = 'Numeric', building_blocks= config)
    search = Search(trie= trie, experiment_handler= experiment_handler, model_fn=get_lgbm)
    search.start(train_num, n_trials=10)


