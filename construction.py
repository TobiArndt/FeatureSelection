from PreprocessingAlgorithms.CorrelationHandling.algorithms import VifColumns
from PreprocessingAlgorithms.Distributions.algorithms import BoxCoxTransform
from PreprocessingAlgorithms.OutlierDetection.algorithms import OutlierIQR, OutlierLOF
from PreprocessingAlgorithms.UnsupervisedFeatureSelection.algorithms import ColumnDropper, VarianceThresholdHandler
from PreprocessingAlgorithms.Imputing.algorithms import ImputerStupid, ImputerIterative
from PreprocessingAlgorithms.building_parts import Identity, build
from Trie.node import PreprocessNode
from Utils.visualizer import visualize_tree

from collections import OrderedDict


from enum import Enum
class A(Enum):
    Correlation_VIF = 0
    Distribution_BoxCox = 1
    Outlier_IQR = 2
    Outlier_LOF = 3
    Optional = 4
    Imputer_Stupid = 5
    Imputer_Iterative = 6
    Feature_ColumnDropper = 7
    Feature_Variance = 8



building_map = {
    A.Correlation_VIF: VifColumns,
    A.Distribution_BoxCox: BoxCoxTransform,
    A.Outlier_IQR: OutlierIQR,
    A.Outlier_LOF: OutlierLOF,
    A.Optional: Identity,
    A.Imputer_Stupid: ImputerStupid,
    A.Imputer_Iterative: ImputerIterative,
    A.Feature_ColumnDropper: ColumnDropper,
    A.Feature_Variance: VarianceThresholdHandler
}


class KeyFnPairGeneratorAll:
  def __init__(self, all_stages):
    self.all_stages = all_stages
    self.all_key_fn_pairs = self.get_all_key_fn_pairs()

  def get_all_key_fn_pairs(self):
    def rek_call(all_key_fn_pairs, tmp_key_fn_pairs, level):
      node = self.all_stages[level]

      for l in node:
        key_fn_pairs = list(tmp_key_fn_pairs)
        key = l[0]
        obj_fn = l[1]
        if key != 'Identity':
          key_fn_pairs.append((key, obj_fn))

        if len(key_fn_pairs) > 0:
          all_key_fn_pairs.append(tuple(key_fn_pairs))

        if level < len(self.all_stages) - 1:
          rek_call(all_key_fn_pairs, key_fn_pairs, level + 1)

    all_key_fn_pairs = []
    tmp_key_fn_pairs = []

    rek_call(all_key_fn_pairs, tmp_key_fn_pairs, 0)
    all_key_fn_pairs = list(OrderedDict.fromkeys(all_key_fn_pairs))
    all_key_fn_pairs = list(map(lambda x: list(
      filter(lambda y: y[0] != 'Identity', x)
    ), all_key_fn_pairs))

    for p in all_key_fn_pairs:
      print(p)

    print(len(all_key_fn_pairs))
    return all_key_fn_pairs






class ExperimentTrie:
    def __init__(self, name = 'Numeric',
                 building_blocks=
                            [
                                [
                                    (A.Feature_ColumnDropper,{'columns':['ELONGATION_RUNNING', 'SEAM_STRENGTH_RUNNING', 'TENSILE_STRENGTH_CROSS_RUNNING']}), 
                                    (A.Optional,)
                                ],
                                [
                                    (A.Feature_Variance,{'thresh': 0.1}), 
                                    (A.Optional,)
                                ],
                                [
                                    (A.Distribution_BoxCox,{'skewing_threshold': 0.3}), 
                                    (A.Optional,)
                                ],
                                [
                                    (A.Correlation_VIF,{'vif_thresh': 5}),
                                    (A.Optional,)
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
                 ):
        self.building_blocks = building_blocks
        self.name = name
        self.trie = self.construct_experiment()

    def construct_experiment(self):
        building_blocks_out = []
        identity = ('Identity', build(Identity))

        for block in self.building_blocks:
            block_out = []
            for p in block:
                if p[0] == A.Optional:
                    block_out.append(identity)
                else:
                    if len(p) == 2:
                        block_out.append(
                            (A(p[0]).name, build(building_map[p[0]], **p[1]))
                        )
                    else:
                        block_out.append(
                            (A(p[0]).name, build(building_map[p[0]]))
                        )

            building_blocks_out.append(block_out)

        pg = KeyFnPairGeneratorAll(building_blocks_out)
        root = PreprocessNode(self.name)
        for p in pg.all_key_fn_pairs:
            root.add_path(p)

        return root
    
    def preprocess(self, path, dataset):
       new_df = self.trie.process_path(path, dataset)
       return new_df

if __name__ == '__main__':
    experiment = ExperimentTrie()
    visualize_tree(experiment.trie)
