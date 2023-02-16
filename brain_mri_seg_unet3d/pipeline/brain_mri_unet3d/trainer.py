from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_dataloader: Dataset,
        validation_dataloader: Optional[Dataset] = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        epochs: int = 100,
        epoch: int = 0,
        writer: torch.utils.tensorboard = None,
        notebook: bool = False,
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.writer = writer
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_dataloader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if (
                    self.validation_dataloader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    self.lr_scheduler.step(
                        self.validation_loss[i]
                    )  # learning rate scheduler step with validation loss
                else:
                    # self.lr_scheduler.batch()  # learning rate scheduler step
                    self.lr_scheduler.step()

            torch.cuda.empty_cache()

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input_x, target_y = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            # print("input, target", input_x.shape, target_y.shape, target_y.dtype, input_x.dtype)
            out = self.model(input_x)  # one forward pass
            out = out.to(self.device)
            # print("out", out.shape, out.dtype)
            if self.criterion.__class__.__name__ != "CrossEntropyLoss":
                target_y = torch.unsqueeze(target_y, 1)
            loss = self.criterion(out, target_y)  # calculate loss
            if self.criterion.__class__.__name__ != "CrossEntropyLoss":
                target_y = target_y.squeeze(1)
            # print('loss', loss)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )  # update progressbar
            if self.writer is not None:
                self.writer.add_images(
                    "Train/masks/true", target_y.unsqueeze(1), len(self.training_loss)
                )
                self.writer.add_images(
                    "Train/masks/pred",
                    torch.argmax(out, dim=1).unsqueeze(1),
                    len(self.training_loss),
                )

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])
        if self.writer is not None:
            self.writer.add_scalar(
                "Loss/train", np.mean(train_losses), len(self.training_loss)
            )
            self.writer.add_scalar(
                "LearningRate/Plateau",
                self.optimizer.param_groups[0]["lr"],
                len(self.training_loss),
            )

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)
            with torch.no_grad():
                out = self.model(input)
                out = out.to(self.device)
                # print('valid shape', target.shape, out.shape)
                if self.criterion.__class__.__name__ != "CrossEntropyLoss":
                    target = torch.unsqueeze(target, 1)
                loss = self.criterion(out, target)
                if self.criterion.__class__.__name__ != "CrossEntropyLoss":
                    target = target.squeeze(1)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")
                if self.writer is not None:
                    self.writer.add_images(
                        "Test/masks/true",
                        target.unsqueeze(1),
                        len(self.validation_loss),
                    )
                    self.writer.add_images(
                        "Test/masks/pred",
                        torch.argmax(out, dim=1).unsqueeze(1),
                        len(self.validation_loss),
                    )

        self.validation_loss.append(np.mean(valid_losses))
        if self.writer is not None:
            self.writer.add_scalar(
                "Loss/validation", np.mean(valid_losses), len(self.validation_loss)
            )

        batch_iter.close()
