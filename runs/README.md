# TensorBoard Run Archives

This directory contains TensorBoard event logs for the experiments described in the main [README](../README.md). Each archive can be extracted and opened with TensorBoard. The corresponding result logs record the experiment outcomes; the event archives provide training and evaluation traces.

## Experiment Archives

- `SBUKId.sVO.Checkpoints.zip` contains 300 event files for six single-view orientations across the five SBUKId cross-validation folds. The corresponding [results log](../results/SBUKId.sVO.Checkpoints.log) records the reported accuracies.
- `sVO.mVO.Checkpoints.zip` contains 1,035 event files from the single-stream and two-stream orientation experiments. See the corresponding [results log](../results/sVO.mVO.Checkpoints.log).
- `mVO.e2eEnsembleTuning.zip` contains 285 event files from multi-stream ensemble-tuner experiments. See the corresponding [results log](../results/mVO.e2eEnsembleTuning.log).
- `allDatasets-e2eEnsembleTuning-Evaluation.zip` contains 29 event files from final evaluation across the benchmark datasets. See the corresponding [evaluation log](../results/allDatasets-e2eEnsembleTuning-Evaluation.log) and [summary](../results/allDatasets-e2eEnsembleTuning-Summary.log).
