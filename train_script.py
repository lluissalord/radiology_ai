# %%
import numpy as np

from fastai.basics import (
    RocAuc,
    RocAucBinary,
    error_rate,
    BalancedAccuracy,
    F1Score,
    FBeta,
    Precision,
    Recall,
    ValueMetric,
    Learner,
)
from fastai.callback.all import *
from fastai.callback.tensorboard import TensorBoardCallback

from train.data.dataframe import generate_dfs
from train.data.dataloader import (
    AllLabelsInBatchDL,
    define_ds_params,
    define_dls_params,
    get_label_dl,
    get_unlabel_dls,
)
from train.losses.losses import define_losses
from train.models import get_training_model
from train.optimizer import get_optimizer
from train.params import default_params
from train.self_supervised import get_self_supervised_model
from train.tfms import get_item_tfms, get_batch_tfms
from train.utils import seed_everything


def train(run_params, cb_params={}, loss_params={}, debug=True):

    # Seed
    seed_everything(run_params["SEED"])

    # Data
    label_df, unlabel_df, final_df = generate_dfs(run_params, debug=debug)
    train_df = label_df[label_df["Dataset"] == "train"]

    # Item (CPU) Transformations
    item_tfms = get_item_tfms(run_params)

    # Batch (GPU) transformation
    label_tfms, unlabel_tfms = get_batch_tfms(run_params)

    # Self supervised
    model_self_sup = None
    if run_params["SELF_SUPERVISED"]:
        # if run_params['IN_COLAB']:
        #     # Load the TensorBoard notebook extension
        #     %load_ext tensorboard
        #     %tensorboard --logdir {'"' + os.path.join(run_params['PATH_PREFIX'] , 'tb_logs', 'simCLR') + '"'}

        model_self_sup = get_self_supervised_model(run_params)

    ## Define Dataset parameters
    label_ds_params, unlabel_ds_params = define_ds_params(
        run_params, item_tfms, label_tfms
    )

    ## Define DataLoaders parameters
    dls_params, unlabel_dls_params = define_dls_params(run_params)

    # DataLoaders
    if debug:
        print(f"==> Preparing label dataloaders")
    label_dl = get_label_dl(run_params, dls_params, label_ds_params, label_df)

    unlabel_dls = [None]
    if run_params["SSL"] and run_params["LAMBDA_U"] != 0:
        if debug:
            print(f"==> Preparing unlabel dataloaders")
        unlabel_dls = get_unlabel_dls(
            run_params, unlabel_tfms, unlabel_dls_params, unlabel_ds_params, unlabel_df
        )

    # Callbacks
    if debug:
        print(f"==> Preparing callbacks")
    if run_params["SSL"] == run_params["SSL_FIX_MATCH"]:
        from semisupervised.fixmatch.callback import FixMatchCallback as SSLCallback
    elif run_params["SSL"] == run_params["SSL_MIX_MATCH"]:
        from semisupervised.mixmatch.callback import MixMatchCallback as SSLCallback

    cbs = []
    # cbs = [
    #     TensorBoardCallback(
    #             log_dir=os.path.join(
    #                 run_params['PATH_PREFIX'] ,
    #                 'tb_logs',
    #                 'main',
    #                 run_params['MODEL_SAVE_NAME']
    #             ),
    #             projector=True,
    #     )
    # ]
    if not run_params["SSL"]:
        cbs.insert(0, MixUp())
    elif run_params["LAMBDA_U"] != 0:
        ssl_cb = SSLCallback(*unlabel_dls, **cb_params)
        cbs.append(ssl_cb)

        if run_params["SSL"] == run_params["SSL_MIX_MATCH"]:
            cbs.append(MixUp(alpha=run_params["ALPHA"]))

    if run_params["GRAD_MAX_NORM"] is not None:
        cbs.append(GradientClip(max_norm=run_params["GRAD_MAX_NORM"]))

    # Debugging Callback to print out some values of the data being proceess on each batch
    # cbs.append(DebuggingCallback(start_epoch=0))

    cbs.append(
        SaveModelCallback(
            # monitor="valid_fbeta_score",
            # comp=np.greater,
            # min_delta=0.01,
            reset_on_fit=False,
        )
    )
    cbs.append(
        EarlyStoppingCallback(
            # monitor="valid_fbeta_score",
            # comp=np.greater,
            min_delta=0,
            reset_on_fit=True,
            patience=10,
            # min_lr=run_params["OPT_LR"] * 1E-4
        )
    )

    # Optimizer
    opt_func, opt_params = get_optimizer(run_params)

    # Model
    if debug:
        print("==> Creating model")
    model, splitter = get_training_model(
        run_params,
        loss_params,
        train_df,
        n_in=label_dl.n_inp,
        model_self_sup=model_self_sup,
    )

    # Loss
    if debug:
        print("==> Defining loss")
    loss_func = define_losses(run_params, loss_params, train_df, unlabel_dls)

    # Learner
    if debug:
        print("==> Defining learner")
    # Adapt metrics depending on the number of labels
    if label_dl.c == 2:
        average = "binary"
        roc_auc = RocAucBinary()
    else:
        average = "macro"
        roc_auc = RocAuc()

    metrics = [
        error_rate,
        BalancedAccuracy(),
        roc_auc,
        # FBeta(0.5, average=average),
        F1Score(average=average),
        FBeta(2, average=average),
        Precision(average=average),
        Recall(average=average),
    ]

    if run_params["SSL"] and run_params["LAMBDA_U"] != 0:
        # metrics.insert(0, ValueMetric(loss_func.total_loss))
        metrics.insert(0, ValueMetric(loss_func.Lu))
        metrics.insert(0, ValueMetric(loss_func.w))
        metrics.insert(0, ValueMetric(loss_func.Lx))

    learn = Learner(
        label_dl,
        model,
        splitter=splitter,
        loss_func=loss_func,
        opt_func=opt_func,
        **opt_params,
        lr=run_params["OPT_LR"],
        metrics=metrics,
        cbs=cbs,
    )
    learn.recorder.train_metrics = True

    learn.to_fp16()

    if learn.opt is None:
        learn.create_opt()
    learn.opt.set_hyper(
        "lr", learn.lr if run_params["OPT_LR"] is None else run_params["OPT_LR"]
    )
    learn.freeze()
    learn.fit(run_params["TRAIN_FREEZE_EPOCHS"], lr=slice(run_params["OPT_LR"]))

    learn.unfreeze()
    learn.fit_flat_cos(
        run_params["TRAIN_EPOCHS"],
        lr=slice(run_params["OPT_LR"] / 5 / 100, run_params["OPT_LR"] / 5),
    )

    # Review use of EarlyStopping with fine_tune
    # learn.fit(1)
    # Not able to use MLFlow autolog with repeat cycles
    # i = 0
    # while i < run_params["REPEAT_ONE_CYCLE"]:
    #     learn.fine_tune(
    #         run_params["TRAIN_EPOCHS"],
    #         run_params["OPT_LR"],
    #         freeze_epochs=run_params["TRAIN_FREEZE_EPOCHS"] if not i else 0,
    #     )
    #     i += 1

    # This should be done on ParamSched way
    # run_params['OPT_LR'] /= 10
    # run_params['TRAIN_EPOCHS'] /= 2

    learn.to_fp32()

    return learn


#%%
if __name__ == "__main__":
    import os
    import pandas as pd
    import torch
    import mlflow
    import random
    import gc

    from fastai.basics import AvgMetric
    from fastai.vision import models

    exp_name = "kfolds-3"

    run_params, cb_params, loss_params = default_params(in_colab=False)
    run_params["TRAIN_EPOCHS"] = 70
    run_params["TRAIN_FREEZE_EPOCHS"] = 1
    run_params["DATA_SEED"] = 42

    experiment_id = mlflow.set_experiment(exp_name)
    mlflow.fastai.autolog()

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    run_params["MODEL"] = models.resnet152
    run_params["BATCH_SIZE"] = 32

    run_params["K_FOLDS"] = 3

    if not run_params["K_FOLDS"]:
        run_params["K_FOLDS"] = 1

    results = pd.DataFrame([])
    for k in range(run_params["K_FOLDS"]):
        run_params["SEED"] = run_params["DATA_SEED"] * (k + 1)
        run_params["K"] = k

        # Start mlflow run
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=f'{run_params["K"]}_fold'
        ):

            mlflow.log_params(run_params)

            learn = train(run_params, cb_params, loss_params, debug=False)

            for dls_idx in range(len(learn.dls.loaders)):
                # dls_idx = 2
                # preds, targs, losses = learn.get_preds(dls_idx, with_loss=True)
                preds, targs = learn.tta(dls_idx)
                max_preds, outs = torch.max(preds, axis=1)

                res_col = f"dls{dls_idx}_k{k}"
                # results.loc["loss", res_col] = losses.mean().numpy()
                for metric in learn.metrics:
                    try:
                        metric_name = metric.name
                    except AttributeError:
                        metric_name = metric.__name__

                    if isinstance(metric, AvgMetric):
                        metric = metric.func
                        metric_value = metric(preds, targs).numpy()
                    else:
                        try:
                            metric_value = metric(preds, targs)
                        except AssertionError:
                            try:
                                metric_value = metric(preds[:, 1], targs)
                            except ValueError:
                                metric_value = metric(outs, targs)

                    results.loc[metric_name, res_col] = metric_value
                del metric_value, outs, max_preds, targs, preds
            print(results)

            del learn
            gc.collect()
            torch.cuda.empty_cache()
# %%
