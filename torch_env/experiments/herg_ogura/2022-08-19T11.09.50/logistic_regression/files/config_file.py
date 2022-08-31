from riseqsar.dataset.resampling import SubsamplingConfig
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.experiment.hyperparameter_optimization import HyperParameterOptimizationConfig
from riseqsar.experiment.experiment_config import ExperimentSpecification, ExperimentSpecificationCollection
from riseqsar.evaluation.performance import metric_roc_auc


resample_config = SubsamplingConfig(subsampling_ratios={TRAIN: .8, DEV: .1, TEST: .1},
                                    n_subsamples=25,            
                                    random_seed=1234,
                                    mol_sample_strategy='stratified')

# Note that the resamples of the hyper parameter tuning will use the 
# dev set from the upper resample loop as its test set, so doesn't need a TEST split
hp_tune_resample_config = SubsamplingConfig(subsampling_ratios={TRAIN: .9, DEV: .1},
                                            n_subsamples=10,            
                                            random_seed=1234,
                                            mol_sample_strategy='stratified')

hp_config = HyperParameterOptimizationConfig(hp_iterations=5, 
                                             hp_direction='maximize', 
                                             hp_evaluation_metric=metric_roc_auc,
                                             hp_resample_config=hp_tune_resample_config)

model_rng_seed = 1234
dataset_rng_seed = 1234

dataset_spec_path = 'dataset/herg_ogura_filtered/dataset_spec.py'

model_common_kwargs = dict(dataset_spec_path=dataset_spec_path,
                            evaluation_metrics=[metric_roc_auc],
                            hp_config=hp_config,
                            resample_config=resample_config,
                            model_rng_seed=model_rng_seed,
                            dataset_rng_seed=dataset_rng_seed)

logistic_regression_experiment_specification = ExperimentSpecification(name='logistic_regression', 
                                                                       experiment_environment='rise-qsar-torch', 
                                                                       model_spec_path='configs/model_configs/herg_ogura/logistic_regression_hpsearch.py',
                                                                       **model_common_kwargs
                                                                       )

random_forest_experiment_specification = ExperimentSpecification(name='random_forest', 
                                                                experiment_environment='rise-qsar-torch', 
                                                                model_spec_path='configs/model_configs/herg_ogura/random_forest_hpsearch.py',
                                                                **model_common_kwargs)

xgboost_experiment_specification = ExperimentSpecification(name='xgboost', 
                                                                       experiment_environment='rise-qsar-xgboost', 
                                                                       model_spec_path='configs/model_configs/herg_ogura/xgboost_hpsearch.py',
                                                                       **model_common_kwargs)

svm_experiment_specification = ExperimentSpecification(name='thundersvm', 
                                                        experiment_environment='rise-qsar-thundersvm', 
                                                        model_spec_path='configs/model_configs/herg_ogura/svm_hpsearch.py',
                                                        **model_common_kwargs)

gnn_experiment_specification = ExperimentSpecification(name='graph_neural_network', 
                                                        experiment_environment='rise-qsar-torch', 
                                                        model_spec_path='configs/model_configs/herg_ogura/gnn_hpsearch.py',
                                                        **model_common_kwargs)

ffn_experiment_specification = ExperimentSpecification(name='feedforward_neural_network', 
                                                        experiment_environment='rise-qsar-torch', 
                                                        model_spec_path='configs/model_configs/herg_ogura/ffn_hpsearch.py',
                                                        **model_common_kwargs)

rnn_experiment_specification = ExperimentSpecification(name='recurrent_neural_network', 
                                                        experiment_environment='rise-qsar-torch', 
                                                        model_spec_path='configs/model_configs/herg_ogura/rnn_hpsearch.py',
                                                        **model_common_kwargs)

transformer_experiment_specification = ExperimentSpecification(name='transformer_neural_network', 
                                                        experiment_environment='rise-qsar-torch', 
                                                        model_spec_path='configs/model_configs/herg_ogura/transformer_hpsearch.py',
                                                        **model_common_kwargs)

experiment_config = ExperimentSpecificationCollection(name='herg_ogura', output_dir='experiments', 
                                                      experiments=[
                                                                   #svm_experiment_specification,
                                                                   #gnn_experiment_specification,
                                                                   #xgboost_experiment_specification,
                                                                   #random_forest_experiment_specification,
                                                                   logistic_regression_experiment_specification,
                                                                   #ffn_experiment_specification
                                                      ])


