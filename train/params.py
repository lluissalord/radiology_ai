import os

from fastai.vision import models


def default_param_setup(run_params):
    if run_params["IN_COLAB"]:
        run_params["PATH_DRIVER"] = "/content/gdrive/My Drive/"
        run_params["DATA_FOLDER"] = "Dataset/"
        run_params["RAW_PATH"] = ""
        run_params["TMP_PREFIX"] = "/tmp/"

    else:
        run_params["PATH_DRIVER"] = ""
        run_params["DATA_FOLDER"] = "sources/"
        run_params["RAW_PATH"] = "E:/Google Drive/"
        run_params["TMP_PREFIX"] = "sources/tmp/"

    filename_prefix = "IMG_"

    run_params["PATH_PREFIX"] = os.path.join(
        run_params["PATH_DRIVER"], run_params["DATA_FOLDER"], ""
    )

    run_params["RAW_PREFIX"] = os.path.join(
        run_params["PATH_DRIVER"], run_params["RAW_PATH"], ""
    )
    run_params["RAW_FOLDER"] = run_params["RAW_PREFIX"] + "DICOMS"

    run_params["ORGANIZE_FOLDER"] = run_params["PATH_PREFIX"] + "pending_classification"

    run_params["PREPROCESS_FOLDER"] = run_params["PATH_PREFIX"] + "preprocess"
    run_params["DRIVE_RAW_PREPROCESS_FOLDER"] = (
        run_params["PATH_PREFIX"] + "raw_preprocess"
    )
    if run_params["IN_COLAB"]:
        run_params["RAW_PREPROCESS_FOLDER"] = (
            run_params["TMP_PREFIX"] + "raw_preprocess"
        )
    else:
        run_params["RAW_PREPROCESS_FOLDER"] = (
            run_params["PATH_PREFIX"] + "raw_preprocess"
        )

    run_params["DOUBT_FOLDER"] = run_params["TMP_PREFIX"] + "doubt"
    run_params["TRAIN_FOLDER"] = run_params["TMP_PREFIX"] + "train"

    run_params["MODELS_FOLDER"] = os.path.join(run_params["PATH_PREFIX"], "models")

    run_params["KNEE_SVM_MODEL_PATH"] = (
        run_params["PATH_PREFIX"] + "extras/svm_model.npy"
    )

    return run_params


def SSL_params(run_params):
    cb_params, loss_params = {}, {}
    if run_params["SSL"]:
        if run_params["SSL"] == run_params["SSL_FIX_MATCH"]:
            run_params["BATCH_SIZE"] = 8
            run_params["MOMENTUM"] = 0.9
            run_params["OPT_WD"] = 0.0005
            run_params["LAMBDA_U"] = 1
            run_params["MU"] = 5
            run_params["LABEL_THRESHOLD"] = 0.95

            cb_params = {}

            loss_params = {
                "bs": run_params["BATCH_SIZE"],
                "mu": run_params["MU"],
                "lambda_u": run_params["LAMBDA_U"],
                "label_threshold": run_params["LABEL_THRESHOLD"],
            }
        elif run_params["SSL"] == run_params["SSL_MIX_MATCH"]:
            run_params["BATCH_SIZE"] = 16
            run_params["LAMBDA_U"] = 75
            run_params["T"] = 0.5
            run_params["ALPHA"] = 0.75

            cb_params = {"T": run_params["T"]}

            loss_params = {
                "bs": run_params["BATCH_SIZE"],
                "lambda_u": run_params["LAMBDA_U"],
            }

        loss_params["use_SCL"] = False
        loss_params["beta"] = 0.5

        # Huge impact as with 0.999 it last a lot to converge when low number of batches
        # TODO: It could be adapted to the number of batches per epoch
        # TODO: Besides EMAModel generates some conflicts and evaluation is not performed as it should
        run_params["EMA_DECAY"] = 0.99

    else:
        run_params["BATCH_SIZE"] = 64

    return run_params, cb_params, loss_params


def default_params(in_colab):
    run_params, cb_params, loss_params = {}, {}, {}

    run_params["IN_COLAB"] = in_colab
    run_params = default_param_setup(run_params)

    # Data Split

    run_params["TEST_SIZE"] = 0
    run_params["VALID_SIZE"] = 0.3

    run_params["POSITIVES_ON_TRAIN"] = 0.3
    # run_params['POSITIVES_ON_TRAIN'] = None

    # Transformations

    # run_params['HIST_CLIPPING'] = True
    # run_params['HIST_CLIPPING_CUT_MIN'] = 5.
    # run_params['HIST_CLIPPING_CUT_MAX'] = 99.
    run_params["BACKGROUND_PREPROCESS"] = True

    run_params["KNEE_LOCALIZER"] = True
    run_params["CLAHE_SCALED"] = True
    run_params["HIST_SCALED"] = False
    run_params["HIST_SCALED_SELF"] = True

    run_params["RESIZE"] = 384
    run_params["RANDOM_RESIZE_CROP"] = 224
    run_params["RANDOM_MIN_SCALE"] = 0.5

    # Hyperparameters

    run_params["BINARY_CLASSIFICATION"] = True

    run_params["CLASS_WEIGHT"] = True
    run_params["WEIGTHED_SAMPLER"] = False
    run_params["ALL_LABELS_IN_BATCH"] = True
    run_params["MIN_SAMPLES_PER_LABEL"] = 1

    run_params["TRAIN_FREEZE_EPOCHS"] = 1
    run_params["TRAIN_EPOCHS"] = 50
    run_params["REPEAT_ONE_CYCLE"] = 1

    run_params["OPTIMIZER"] = "AdaBelief"  # AdaBelief, Adam
    if run_params["OPTIMIZER"] == "AdaBelief":
        run_params["OPT_LR"] = 0.005
        run_params["OPT_WD"] = 0.25
        run_params["OPT_MOM"] = 0.85
        run_params["OPT_SQR_MOM"] = 0.999
    else:
        run_params["OPT_LR"] = 0.015
        run_params["OPT_WD"] = 0.15
        run_params["OPT_MOM"] = 0.95
        run_params["OPT_SQR_MOM"] = 0.995

    opt_params = {
        "wd": run_params["OPT_WD"],
        "mom": run_params["OPT_MOM"],
        "sqr_mom": run_params["OPT_SQR_MOM"],
    }
    run_params["GRAD_MAX_NORM"] = 1.0

    run_params["HEAD_DROPOUT_P"] = 0.75

    # Self-Supervised Learning

    run_params["SELF_SUPERVISED"] = False
    run_params["SELF_SUPERVISED_TRAIN"] = True

    run_params["SELF_SUPERVISED_EPOCHS"] = 40
    run_params["SELF_SUPERVISED_WARMUP_EPOCHS"] = (
        run_params["SELF_SUPERVISED_EPOCHS"] // 10
    )

    run_params["SELF_SUPERVISED_BATCH_SIZE"] = 64

    # Semi-Supervised Learning

    run_params["SSL_MIX_MATCH"] = "MixMatch"
    run_params["SSL_FIX_MATCH"] = "FixMatch"

    run_params["SSL"] = run_params["SSL_FIX_MATCH"]
    run_params["SSL"] = None
    run_params, cb_params, loss_params = SSL_params(run_params)

    # Model

    run_params["USE_SAVED_MODEL"] = False
    run_params["SAVE_MODEL"] = False

    run_params["MODEL"] = models.resnet18
    # run_params['MODEL'] = models.densenet121
    # MODEL = 'efficientnet-b0'
    run_params["MODEL_VERSION"] = 0
    run_params[
        "MODEL_DESCRIPTION"
    ] = f'{"SSL_" if run_params["SSL"] else ""}{"SELFSUP_" if run_params["SELF_SUPERVISED"] else ""}sz{min(run_params["RESIZE"], run_params["RANDOM_RESIZE_CROP"])}'
    run_params[
        "MODEL_SAVE_NAME"
    ] = f'{run_params["MODEL"].__name__}_{run_params["MODEL_DESCRIPTION"]}_v{run_params["MODEL_VERSION"]}.pkl'
    run_params["MODEL_SAVE_PATH"] = os.path.join(
        run_params["MODELS_FOLDER"], run_params["MODEL_SAVE_NAME"]
    )

    run_params["PRETRAINED_MODEL_SAVE_NAME"] = "resnet18_v0.pkl"
    run_params["PRETRAINED_MODEL_SAVE_NAME"] = os.path.join(
        run_params["MODELS_FOLDER"], run_params["PRETRAINED_MODEL_SAVE_NAME"]
    )
    if run_params["USE_SAVED_MODEL"] and not os.path.exists(
        run_params["PRETRAINED_MODEL_SAVE_NAME"]
    ):
        print(
            f'Not using pretrained model as there is no model on: {run_params["PRETRAINED_MODEL_SAVE_NAME"]}'
        )
        run_params["USE_SAVED_MODEL"] = False

    # Seed
    run_params["SEED"] = 42
    run_params["DATA_SEED"] = 42

    return run_params, cb_params, loss_params
