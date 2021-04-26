from sklearn.model_selection import train_test_split
import PIL
from tqdm import tqdm

from torch.utils.data import Dataset


class SelfSupervisedDataset(Dataset):
    """Dataset to be used on Self Supervised Learning with Python Lightining"""

    def __init__(
        self,
        df,
        validation=False,
        transform=None,
        path_col="Original_Filename",
        prefix="",
        suffix=".png",
    ):

        self.transform = transform

        # use sklearn's module to return training data and test data
        if validation:
            _, self.df = train_test_split(df, test_size=0.20, random_state=42)

        else:
            self.df, _ = train_test_split(df, test_size=0.20, random_state=42)

        self.image_pairs = []

        for idx, d in tqdm(enumerate(self.df[path_col]), total=len(self.df.index)):

            im = PIL.Image.open(prefix + d + suffix).convert("RGB")

            if self.transform:
                sample = self.transform(
                    im
                )  # applies the SIMCLR transform required, including new rotation
            else:
                sample = im

            self.image_pairs.append(sample)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # doing the PIL.image.open and transform stuff here is quite slow
        return (self.image_pairs[idx], 0)


def get_self_supervised_model(run_params):
    import pl_bolts
    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.models.self_supervised.simclr import (
        SimCLRTrainDataTransform,
        SimCLREvalDataTransform,
    )
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    checkpoints_dir = os.path.join(run_params["PATH_PREFIX"], "checkpoints")
    checkpoint_resume = os.path.join(
        checkpoints_dir, run_params["MODEL_SAVE_NAME"] + ".ckpt"
    )

    dataset = SelfSupervisedDataset(
        final_df,
        validation=False,
        transform=SimCLRTrainDataTransform(
            min(run_params["RESIZE"], run_params["RANDOM_RESIZE_CROP"])
        ),
        prefix=run_params["RAW_PREPROCESS_FOLDER"] + "/",
    )
    val_dataset = SelfSupervisedDataset(
        final_df,
        validation=True,
        transform=SimCLREvalDataTransform(
            min(run_params["RESIZE"], run_params["RANDOM_RESIZE_CROP"])
        ),
        prefix=run_params["RAW_PREPROCESS_FOLDER"] + "/",
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=run_params["SELF_SUPERVISED_BATCH_SIZE"], num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=run_params["SELF_SUPERVISED_BATCH_SIZE"], num_workers=0
    )
    num_samples = len(dataset)

    # #init model with batch size, num_samples (len of data), epochs to train, and autofinds learning rate
    model_self_sup = SimCLR(
        gpus=1,
        arch="resnet50",
        dataset="",
        max_epochs=run_params["SELF_SUPERVISED_EPOCHS"],
        warmup_epochs=run_params["SELF_SUPERVISED_WARMUP_EPOCHS"],
        batch_size=run_params["SELF_SUPERVISED_BATCH_SIZE"],
        num_samples=num_samples,
    )

    if run_params["SELF_SUPERVISED_TRAIN"]:
        logger = TensorBoardLogger(
            os.path.join(run_params["PATH_PREFIX"], "tb_logs", "simCLR"),
            name=run_params["MODEL_SAVE_NAME"],
        )
        early_stopping = EarlyStopping("val_loss", patience=5)

        if os.path.exists(checkpoint_resume):
            trainer = Trainer(
                gpus=1,
                max_epochs=run_params["SELF_SUPERVISED_EPOCHS"],
                logger=logger,
                auto_scale_batch_size=True,
                resume_from_checkpoint=checkpoint_resume,
                callbacks=[early_stopping],
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=checkpoints_dir,
                filename=run_params["MODEL_SAVE_NAME"],
                save_top_k=1,
                mode="min",
            )

            trainer = Trainer(
                gpus=1,
                max_epochs=run_params["SELF_SUPERVISED_EPOCHS"],
                logger=logger,
                auto_scale_batch_size=True,
                callbacks=[checkpoint_callback, early_stopping],
            )

        trainer.fit(model_self_sup, data_loader, val_loader)
        model_self_sup = model_self_sup.load_from_checkpoint(checkpoint_resume)
    elif os.path.exists(checkpoint_resume):
        model_self_sup.load_from_checkpoint(checkpoint_resume)
    else:
        print(
            f"Not checkpoint found, so it could not load model from it\n{checkpoint_resume}"
        )

    return model_self_sup
