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
    if run_params["SSL"]:
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
    else:
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
            monitor="valid_fbeta_score",
            comp=np.greater,
            min_delta=0.01,
            reset_on_fit=False,
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

    if run_params["SSL"]:
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

    # Review use of EarlyStopping with fine_tune
    # learn.fit(1)
    # Not able to use MLFlow autolog with repeat cycles
    i = 0
    while i < run_params["REPEAT_ONE_CYCLE"]:
        learn.fine_tune(
            run_params["TRAIN_EPOCHS"],
            run_params["OPT_LR"],
            freeze_epochs=run_params["TRAIN_FREEZE_EPOCHS"] if not i else 0,
        )
        plt.plot(
            np.array(learn.recorder.values)[
                :, learn.recorder.metric_names.index("valid_fbeta_score") - 1
            ]
        )
        plt.ylabel("valid_fbeta_score")
        plt.show()
        i += 1

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

    exp_name = "usual-run"

    run_params, cb_params, loss_params = default_params(in_colab=False)
    run_params["TRAIN_EPOCHS"] = 30
    run_params["SEED"] = 3
    run_params["DATA_SEED"] = 3

    experiment_id = mlflow.set_experiment(exp_name)
    mlflow.fastai.autolog()

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Start mlflow run
    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_params(run_params)

        learn = train(run_params, cb_params, loss_params, debug=False)

        results = pd.DataFrame([])
        for dls_idx in range(len(learn.dls.loaders)):
            # dls_idx = 2
            preds, targs, losses = learn.get_preds(dls_idx, with_loss=True)
            max_preds, outs = torch.max(preds, axis=1)

            results.loc["loss", dls_idx] = losses.mean().numpy()
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
                        metric_value = metric(outs, targs)

                results.loc[metric_name, dls_idx] = metric_value
        display(results)
# %%
