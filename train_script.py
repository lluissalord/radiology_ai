# %%
import numpy as np
from copy import deepcopy

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


def prepare_learn(
    run_params, cb_params={}, loss_params={}, filter_centers=False, debug=True
):
    # Seed
    seed_everything(run_params["SEED"])

    # Data
    label_df, unlabel_df = generate_dfs(
        run_params, filter_centers=filter_centers, debug=debug
    )
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

    return learn


def train(run_params, cb_params={}, loss_params={}, filter_centers=False, debug=True):
    learn = prepare_learn(
        run_params,
        cb_params=cb_params,
        loss_params=loss_params,
        filter_centers=filter_centers,
        debug=debug,
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


def add_preds_to_df(df, preds, dl, unique_col_name="ID"):
    no_duplicated_img_ids_match = ~dl.items[unique_col_name].duplicated()
    preds_df = pd.DataFrame(preds, index=dl.items[unique_col_name].values)[
        no_duplicated_img_ids_match.astype(bool).values
    ]
    if len(df) == 0:
        df = pd.concat([df, preds_df])
    else:
        try:
            no_duplicated_img_ids = dl.items[no_duplicated_img_ids_match][
                unique_col_name
            ]
            df[no_duplicated_img_ids] = (df[no_duplicated_img_ids] * k + preds_df) / (
                k + 1
            )
        except KeyError:
            df = pd.concat([df, preds_df])

    return df


# %%
if __name__ == "__main__":
    import os
    import pandas as pd
    import torch
    import mlflow
    import random
    import datetime
    import gc

    from fastai.basics import AvgMetric
    from fastai.vision import models

    for filter_centers in [True, False]:
        load_previous_run = False
        train_model = not load_previous_run
        if load_previous_run:
            current_date = "20211018"
        else:
            current_date = datetime.date.today().strftime("%Y%m%d")

        run_params, cb_params, loss_params = default_params(in_colab=False)

        run_params["K_FOLDS"] = 5

        exp_name = f"kfolds-{run_params['K_FOLDS']}_{current_date}{'_filtered' if filter_centers else ''}"

        if load_previous_run:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        else:
            experiment_id = mlflow.set_experiment(exp_name).experiment_id
        run_params["TRAIN_EPOCHS"] = 70
        run_params["TRAIN_FREEZE_EPOCHS"] = 1
        run_params["DATA_SEED"] = 42

        mlflow.fastai.autolog()

        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        run_params["MODEL"] = models.resnet152
        run_params["BATCH_SIZE"] = 32

        if not run_params["K_FOLDS"]:
            run_params["K_FOLDS"] = 1

        label_df, unlabel_df = generate_dfs(
            run_params, filter_centers=filter_centers, debug=False
        )

        results = pd.DataFrame([])
        all_labeled_preds = pd.DataFrame([])
        all_unlabeled_preds = pd.DataFrame([])
        for k in range(run_params["K_FOLDS"]):
            run_params["SEED"] = run_params["DATA_SEED"] * (k + 1)
            run_params["K"] = k
            if load_previous_run:
                run_infos = sorted(
                    mlflow.list_run_infos(experiment_id=experiment_id),
                    key=lambda x: x.end_time,
                )
                if len(run_infos) > k:
                    run_id = run_infos[k].run_id
                    train_model = False
                else:
                    print(
                        f"Training new model on experiment `{exp_name}` for kFold = {k}"
                    )
                    run_id = None
                    train_model = True
            else:
                print(f"Training new model on experiment `{exp_name}` for kFold = {k}")
                run_id = None

            run_name = f'{run_params["K"]}_fold'

            # Start mlflow run
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=run_name, run_id=run_id
            ) as run:
                try:
                    if train_model:
                        mlflow.log_params(run_params)

                        learn = train(
                            run_params,
                            cb_params,
                            loss_params,
                            filter_centers=filter_centers,
                            debug=False,
                        )
                        learn_template = None
                    else:
                        model_uri = "runs:/{}/model".format(run.info.run_id)
                        learn = mlflow.fastai.load_model(model_uri)

                        learn_template = prepare_learn(
                            run_params,
                            cb_params=cb_params,
                            loss_params=loss_params,
                            filter_centers=filter_centers,
                            debug=False,
                        )
                        learn.dls = learn_template.dls.cuda()
                        learn.model = learn.model.cuda()
                        learn.loss_func.func = learn_template.loss_func.func.to(
                            "cuda:0"
                        )

                    for dls_idx in range(len(learn.dls.loaders)):
                        # dls_idx = 2
                        # preds, targs, losses = learn.get_preds(dls_idx, with_loss=True)
                        preds, targs = learn.tta(dls_idx)
                        max_preds, outs = torch.max(preds, axis=1)

                        if dls_idx in [1, 2]:
                            all_labeled_preds = add_preds_to_df(
                                all_labeled_preds, preds, dl=learn.dls.loaders[dls_idx]
                            )

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
                    print(f'\n\n{"-"*40}RESULTS{"-"*40}\n', results)

                    test_dl = learn.dls.test_dl(
                        unlabel_df,
                        with_labels=False,
                        bs=run_params["BATCH_SIZE"],
                        num_workers=0,
                    )
                    preds, targs = learn.tta(dl=test_dl)
                    max_preds, outs = torch.max(preds, axis=1)
                    all_unlabeled_preds = add_preds_to_df(
                        all_unlabeled_preds,
                        preds,
                        dl=test_dl,
                        unique_col_name="Raw_preprocess",
                    )

                    del outs, max_preds, targs, preds

                    del learn, learn_template
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

                    results.to_csv(f"{exp_name}_results.csv")
                    mlflow.log_artifact(local_path=f"{exp_name}_results.csv")

                    all_labeled_preds.to_csv(f"{exp_name}_labeled_preds.csv")
                    mlflow.log_artifact(local_path=f"{exp_name}_labeled_preds.csv")

                    all_unlabeled_preds.to_csv(f"{exp_name}_unlabeled_preds.csv")
                    mlflow.log_artifact(local_path=f"{exp_name}_unlabeled_preds.csv")
# %%
import pandas as pd

current_date = "20211018"
current_date = "20220106"
current_date = "20220212"
current_date = "20220325"
current_date = "20220326"
exp_name = f"kfolds-{run_params['K_FOLDS']}_{current_date}"
all_unlabeled_preds = pd.read_csv(f"{exp_name}_unlabeled_preds.csv")
# all_unlabeled_preds = pd.read_csv(f"unlabeled_preds.csv", sep=';', decimal=',')
all_unlabeled_preds = all_unlabeled_preds.set_index("Unnamed: 0").astype(float)
all_unlabeled_preds["1"].rename("Unlabeled").hist(bins=100, legend=True)

all_labeled_preds = pd.read_csv(f"{exp_name}_labeled_preds.csv")
# all_labeled_preds = pd.read_csv(f"labeled_preds.csv", sep=';', decimal=',')
all_labeled_preds = all_labeled_preds.set_index("Unnamed: 0").astype(float)
all_labeled_preds["1"].rename("Labeled").hist(bins=100, legend=True)

initial_labeled_preds = pd.read_csv(
    f"kfolds-{run_params['K_FOLDS']}_20211018_labeled_preds.csv"
)
initial_labeled_preds = initial_labeled_preds.set_index("Unnamed: 0").astype(float)


# %%
def weight_preds(reference_df, weights, preds_type="labeled"):
    weighted_preds = reference_df[[]].copy()
    weighted_preds = weighted_preds.drop_duplicates(keep="first")
    for date, weight in weights.items():
        exp_name = f"kfolds-{run_params['K_FOLDS']}_{date}"
        preds = pd.read_csv(f"{exp_name}_{preds_type}_preds.csv")
        preds = preds.set_index("Unnamed: 0").astype(float)
        preds = preds.groupby(preds.index).mean()
        weighted_preds = weighted_preds.add(preds.mul(weight), fill_value=0)
    return weighted_preds


weights = {
    "20211018": 0.0,
    "20220106": 0.0,
    "20220212": 0.0,
    "20220325": 0.25,
    "20220326": 0.25,
    "20220423": 0.5,
}

all_unlabeled_preds = weight_preds(all_unlabeled_preds, weights, preds_type="unlabeled")
all_unlabeled_preds["1"].rename("Unlabeled").hist(bins=100, legend=True)
all_labeled_preds = weight_preds(all_labeled_preds, weights, preds_type="labeled")
all_labeled_preds["1"].rename("Labeled").hist(bins=100, legend=True)
# %%

from train.data.dataframe import generate_dfs
from train.params import default_params
from pathlib import Path
import pandas as pd

filter_centers = False

run_params, cb_params, loss_params = default_params(in_colab=False)
label_df, unlabel_df = generate_dfs(
    run_params, filter_centers=filter_centers, debug=False
)
# %%
try:
    all_labeled_preds = pd.merge(
        all_labeled_preds,
        label_df[["ID", "Target", "Original_Filename"]].drop_duplicates("ID"),
        on="ID",
        suffixes=("", "_labeled"),
    )
except KeyError:
    all_labeled_preds = pd.merge(
        all_labeled_preds,
        label_df[["ID", "Target", "Original_Filename"]].drop_duplicates("ID"),
        left_index=True,
        right_on="ID",
        suffixes=("", "_labeled"),
    )

all_labeled_preds

metadata_save_path = run_params["PATH_PREFIX"] + "metadata_raw.csv"
metadata_df = pd.read_csv(metadata_save_path)
# %%
import pandas as pd
from pathlib import Path
from organize.relation import open_name_relation_file

threshold = 0.85
N = 1050

sent_unlabeled_preds_threshold = all_unlabeled_preds[
    all_unlabeled_preds["1"] > threshold
].copy()
sent_unlabeled_preds_threshold.index = pd.Series(
    all_unlabeled_preds[all_unlabeled_preds["1"] > threshold].index,
    name="Original_Filename",
).apply(lambda x: Path(x).name[:-4])
sum((sent_unlabeled_preds_threshold["1"] > threshold))

sent_unlabeled_preds_threshold = sent_unlabeled_preds_threshold.sample(n=N)
sent_unlabeled_preds_threshold.to_csv("sent_unlabeled_preds_threshold.csv", sep=";")
sum((sent_unlabeled_preds_threshold["1"] > threshold))

# sent_unlabeled_preds_threshold = pd.read_csv('sent_unlabeled_preds_threshold.csv', sep=';')
metadata_labels = pd.read_csv(
    run_params["PATH_PREFIX"] + "metadata_labels.csv", sep=","
)
relation_df = pd.read_csv(run_params["PATH_PREFIX"] + "relation.csv", sep=",")
relation_df = open_name_relation_file(
    run_params["PATH_PREFIX"] + "relation.csv", sep=","
)
all_df = pd.read_excel(run_params["PATH_PREFIX"] + "all.xlsx", engine="openpyxl")

df = all_df.merge(relation_df, how="left", left_on="ID", right_on="Filename").merge(
    sent_unlabeled_preds_threshold, how="left", on="Original_Filename"
)
(sent_unlabeled_preds_threshold["1"] > threshold).sum() - (df["1"].notnull()).sum()

# %%
from train.params import default_params
import pandas as pd

run_params, cb_params, loss_params = default_params(in_colab=False)
all_df = pd.read_excel(run_params["PATH_PREFIX"] + "all.xlsx", engine="openpyxl")
# %%
last_block = 488
last_block = 493

after_last_block_mask = all_df.Blocks.apply(
    lambda blocks: any([int(block) > last_block for block in eval(blocks)])
)
all_df[after_last_block_mask].Target.value_counts(dropna=False)
# %%
model_uri = "runs:/{}/model".format("ac010df970ac48c4bcf7d93153876b5d")
learn = mlflow.fastai.load_model(model_uri)

filter_centers = False

learn_template = prepare_learn(
    run_params,
    cb_params=cb_params,
    loss_params=loss_params,
    filter_centers=filter_centers,
    debug=False,
)
learn.dls = learn_template.dls.cuda()
learn.model = learn.model.cuda()
learn.loss_func.func = learn_template.loss_func.func.to("cuda:0")
# %%
mask = learn.dls[-1].items.ID == "IMG_8336"

preds, targs = learn.tta(-1)
max_preds, outs = torch.max(preds, axis=1)
# %%
preds.numpy()[mask]
# %%
