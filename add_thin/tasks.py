import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from add_thin.data import Batch
from add_thin.metrics import (
    MMD,
    forecast_wasserstein,
    lengths_distribution_wasserstein_distance,
)


class Tasks(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay: float,
        weight_decay: float = 0.0,
        lr_schedule=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        self.model = model
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        Compute BCE loss for the classification task.
        """
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask.flatten()]
        target = x_n.kept.flatten()[x_n.mask.flatten()]
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        Compute the average (over batch) negative log-likelihood of the event sequences.
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        Compute the loss for the classification and intensity.
        """
        intensity = self.intensity_loss(log_prob_x_0) / self.model.n_max

        classification = (
            self.classification_loss(x_n_int_x_0, x_n) / self.model.n_max
        )
        loss = classification + intensity
        return loss, classification, intensity
    
    def L11_loss(self, result_array, lambda_1, K, sample_array, M = 100, T = 24):    
        # First term
        print(
            "result array min/max/mean/std:",
            result_array.min().item(),
            result_array.max().item(),
            result_array.mean().item(),
            result_array.std().item(),
        )
        
        term1 = result_array.mean() / K

        # Second term
        #TODO add one time sample array
        print(
            "sample_array min/max/mean/std:",
            sample_array.min().item(),
            sample_array.max().item(),
            sample_array.mean().item(),
            sample_array.std().item(),
        )
        inner = sample_array.mean(dim=1).squeeze(-1) / K
        print(
            "inner raw min/max/mean:",
            inner.min().item(),
            inner.max().item(),
            inner.mean().item(),
        )
        inner = torch.clamp(inner, min=-20, max=20)
        exp_term = torch.exp(inner)
        term2 = (lambda_1 * T / M) * exp_term.mean()
        return (term1 - term2).to(result_array.device)

    def mse_loss(self, e_r, e_r1):
        loss = F.mse_loss(e_r, e_r1)
        return loss

    def step(self, batch, name):
        """
        Apply model to batch and compute loss.
        """
        # Forward pass
        result_array, lambda_1, steps, sample_array, T = self.model.forward(batch)

        l11_loss = self.L11_loss(result_array, lambda_1, steps, sample_array, T=T)
        mse_losses = [self.mse_loss(result_array[i], result_array[i + 1])
                  for i in range(len(result_array) - 1)]
        mse_loss = torch.stack(mse_losses).mean()
        
        final_loss = mse_loss - l11_loss

        # final_loss = l11_loss - mse_loss
        # alpha = 1.0
        # beta = 0.1
        # final_loss = alpha * mse_loss - beta * l11_loss
        print(
            f"mse={mse_loss.detach().item():.6f}, "
            f"l11={l11_loss.detach().item():.6f}, "
            f"final={final_loss.detach().item():.6f}"
        )
    
        # Add validation checks
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"Warning: Invalid loss detected: {final_loss}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Ensure loss requires gradient
        if not final_loss.requires_grad:
            print("Warning: Loss doesn't require gradients")
            final_loss = final_loss.detach().requires_grad_(True)

        # Log loss
        self.log(
            f"{name}/l11_loss",
            l11_loss.detach().item(),
            # batch_size=batch[0].batch_size,
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/mse_loss",
            mse_loss.detach().item(),
            # batch_size=batch[0].batch_size,
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/loss",  # This will create "train/loss"
            final_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        return final_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=10, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train/loss",
            },
        }


class Forecasting(Tasks):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay,
        weight_decay,
        lr_schedule,
    ):
        super().__init__(
            model, learning_rate, lr_decay, weight_decay, lr_schedule
        )

    def set_history(self, batch):
        times = (
            torch.rand((len(batch),), device=batch.tmax.device)
            * (batch.tmax - 2 * self.model.forecast_window)
            + self.model.forecast_window
        )

        history, future, forecast_end, forecast_start = batch.split_time(
            times, times + self.model.forecast_window
        )

        if history.batch_size == 0 or future.batch_size == 0:
            return future, forecast_end, forecast_start

        self.model.set_history(history)
        return future, forecast_end, forecast_start

    def training_step(self, batch, batch_idx):
        future, _, forecast_start = self.set_history(batch)

        if future.batch_size == 0:
            self.log("train/empty_future_batch", 1, batch_size=1)
            return None

        future.time = (
            (future.time - forecast_start[:, None]) / self.model.forecast_window
        ) * future.tmax
        future.tau = (future.tau / self.model.forecast_window) * future.tmax

        loss = self.step(future, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            future, tmax, tmin = self.set_history(batch)

            if future.batch_size == 0:
                self.log("val/empty_future_batch", 1, batch_size=1)
                return

            sample = self.model.sample(len(future), tmax=future.tmax)

            sample.time = (sample.time / future.tmax) * (tmax - tmin)[:, None] + tmin[:, None]

            maes = torch.abs(future.mask.sum(-1) - sample.mask.sum(-1)) / (future.mask.sum(-1) + 1)

            wasserstein = forecast_wasserstein(
                sample.to_time_list(),
                future.to_time_list(),
                batch.tmax.detach().cpu().item(),
            )

            self.log("val/MAE_counts", maes.mean(), batch_size=batch.batch_size)
            self.log("val/forecast_wasserstein_distance", wasserstein, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):
        pass


class DensityEstimation(Tasks):
    def __init__(
        self, model, learning_rate, lr_decay, weight_decay, lr_schedule
    ):
        super().__init__(
            model, learning_rate, lr_decay, weight_decay, lr_schedule
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        step = 2
        with torch.no_grad():
            # if self.global_step >= 1:
            sample = self.model.sample(20, tmax=batch.tmax).to_time_list()

            mmd = MMD(
                sample,
                batch.to_time_list(),
                batch.tmax.detach().cpu().numpy(),
            )[0]
            wasserstein = lengths_distribution_wasserstein_distance(
                sample,
                batch.to_time_list(),
                batch.tmax.detach().cpu().numpy(),
                self.model.n_max,
            )
            self.log("val/sample_mmd", mmd, batch_size=batch.batch_size)
            self.log(
                "val/sample_count_wasserstein",
                wasserstein,
                batch_size=batch.batch_size,
            )

    def test_step(self, batch, batch_idx):
        pass
