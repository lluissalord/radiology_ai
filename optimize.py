import os
import pandas as pd
import torch
import gc
import joblib

import optuna
import mlflow

from train.params import default_params
from train_script import train


def define_run_name(trial):
    run_name_list = []
    for p, val in trial.params.items():
        if type(val) is float:
            if val < 1e-4:
                run_name_list.append(f"{p[4:]}={val:.2e}")
            elif val >= 0.1:
                run_name_list.append(f"{p[4:]}={val:.2f}")
            elif val < 0.01:
                run_name_list.append(f"{p[4:]}={val:.3f}")
            else:
                run_name_list.append(f"{p[4:]}={val:.4f}")
        else:
            run_name_list.append(f"{p[4:]}={val}")
    run_name = "_".join(run_name_list)

    return run_name


def suggest_hyperparameters(trial, run_params):
    # Pending hyp on lr_mult (also div?) of fine tune

    run_params["OPT_LR"] = trial.suggest_float("hyp_LR", 2e-3, 2e-2, log=True)
    run_params["OPT_WD"] = trial.suggest_float("hyp_WD", 0.01, 1, log=True)
    run_params["OPT_MOM"] = trial.suggest_float("hyp_mom", 0.85, 0.95)
    run_params["OPT_SQR_MOM"] = trial.suggest_float(
        "hyp_sqr_mom", 0.99, 0.999, log=True
    )

    # adabelief_params = {
    #     'betas': (run_params['BETA_1'], 0.999),
    #     'weight_decay': run_params['OPT_WD'],
    # }

    # run_params['OPTIMIZER'] = partial(OptimWrapper, opt=AdaBelief, print_change_log=False, **adabelief_params)
    run_params["OPTIMIZER"] = "Adam"

    # run_params['MODEL'] = trial.suggest_categorical("hyp_model", ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'])
    # run_params['MODEL'] = trial.suggest_categorical("hyp_model", ['resnet18', 'resnet34', 'densenet121'])

    # is_efficientNet = run_params['MODEL'].startswith('efficientnet')
    # is_densenet = run_params['MODEL'].startswith('densenet')
    # if not is_efficientNet:
    #     run_params['MODEL'] = getattr(models, run_params['MODEL'])

    # if (is_efficientNet or is_densenet)and run_params['BATCH_SIZE'] > 32:
    #     run_params['BATCH_SIZE'] = 32

    print(f"Suggested hyperparameters: \n{trial.params}")
    # Log the obtained trial parameters using mlflow
    mlflow.log_params(trial.params)

    # Update run name depending on trial
    mlflow.set_tag("mlflow.runName", define_run_name(trial))

    return run_params


def optimize(trial, experiment_id, seed=42, data_seed=None):

    run_params, cb_params, loss_params = default_params(in_colab=False)

    run_params["SEED"] = seed
    run_params["DATA_SEED"] = data_seed if data_seed is not None else seed

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Start mlflow run
    with mlflow.start_run(experiment_id=experiment_id):

        run_params = suggest_hyperparameters(trial, run_params)

        mlflow.log_params(run_params)

        learn = train(run_params, cb_params, loss_params, debug=False)

        dls_idx = 2
        preds, targs, decoded, all_losses = learn.get_preds(
            dls_idx, with_decoded=True, with_loss=True
        )
        max_preds, outs = torch.max(preds, axis=1)

        result = None
        for metric in learn.metrics:
            try:
                metric_name = metric.name
            except AttributeError:
                metric_name = metric.__name__

            if isinstance(metric, AvgMetric):
                metric = metric.func

            try:
                metric_value = metric(preds, targs)
            except AssertionError:
                metric_value = metric(outs, targs)

            print(metric_name, ":", metric_value)

            if metric_name == "fbeta_score":
                result = deepcopy(metric_value)

        return result


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


# %%
if __name__ == "__main__":
    seed = 42
    data_seed = 42
    # for seed in [42,3]:
    study_name = "adam-hyper-balanced" + f"_seed={seed}"
    study_file = study_name + ".pkl"
    study_filepath = os.path.join(
        run_params["PATH_PREFIX"], "optuna_studies", study_file
    )

    run_optimize = True

    # Create the optuna study which shares the experiment name
    if os.path.exists(study_filepath):
        study = joblib.load(study_filepath)
    else:
        study = optuna.create_study(study_name=study_name, direction="maximize")

    if run_optimize:
        experiment_id = mlflow.set_experiment(study_name)
        mlflow.fastai.autolog()

        # Propagate logs to the root logger.
        optuna.logging.set_verbosity(verbosity=optuna.logging.INFO)

        try:
            study.optimize(
                lambda trial: optimize(
                    trial, experiment_id, seed=seed, data_seed=data_seed
                ),
                n_trials=10,
                callbacks=[lambda study, trial: clean_memory()],
            )
        except (RuntimeError, KeyboardInterrupt) as e:
            print(e)

    # Print optuna study statistics
    # Filter optuna trials by state
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\n++++++++++++++++++++++++++++++++++\n")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Best trial value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    if run_optimize:
        joblib.dump(study, study_filepath)

# # %%
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_slice
# import plotly.io as pio
# # %%
# fig = plot_optimization_history(study)
# pio.show(fig)

# fig = plot_parallel_coordinate(study)
# pio.show(fig)
# # plot_parallel_coordinate(study, params=["hyp_LR", "hyp_batch_size_power"])

# fig = plot_contour(study)
# pio.show(fig)

# fig = plot_slice(study)
# pio.show(fig)

# fig = plot_param_importances(study)
# pio.show(fig)

# fig = plot_edf(study)
# pio.show(fig)
# # %%
# results_belief = {
#     'Adam': [
#         0.655737704918033,
#         0.5063291139240507,
#         0.5586592178770949,
#         0.4821428571428572
#     ],
#     'ranger': [
#         0.572289156626506,
#         0.4575163398692811,
#         0.5479452054794521,
#         0.37037037037037035
#     ],
#     'QHAdam': [
#         0.5828220858895706,
#         0.5414012738853503,
#         0.5144032921810701,
#         0.5144032921810701
#     ],
#     'RAdam': [
#         0.6547619047619048,
#         0.5089820359281437,
#         0.5248618784530387,
#         0.3754940711462451
#     ]
# }
# # %%
# results = []
# seed = 42
# # for seed in [42,1,2,3]:
# study_name = 'test'
# study_file = study_name + '.pkl'

# experiment_id = mlflow.set_experiment(study_name)
# mlflow.fastai.autolog()

# run_params, cb_params, loss_params = default_params(in_colab=False)

# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
# mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# # Start mlflow run
# with mlflow.start_run(experiment_id=experiment_id, run_name=f"seed={seed}"):

#     # run_params = suggest_hyperparameters(trial, run_params)
#     run_params['SEED'] = seed

#     mlflow.log_params(run_params)

#     result = main(run_params, cb_params, loss_params, debug = False)
#     print('Results: ', result)
#     print('-'*40)

#     if len(results) == 0:
#         results = [result]
#     else:
#         results.append(result)
# # %%

# %%
