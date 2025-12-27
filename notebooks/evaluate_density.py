import torch
import sys
import numpy as np
sys.path.append('/home/cyang314/ucr_work/add-thin')

from add_thin.metrics import MMD, lengths_distribution_wasserstein_distance
from add_thin.evaluate_utils import get_task, get_run_data

# Set run id and paths
RUN_ID = "naflz4ti"

WANDB_DIR = "/home/cyang314/ucr_work/add-thin/wandb"
PROJECT_ROOT = "/home/cyang314/ucr_work/add-thin/"  # should include data folder

def sample_model(task, tmax, n=4000):
    """
    Unconditionally draw n event sequences from Add Thin.
    """
    with torch.no_grad():
        samples = task.model.sample(n, tmax=tmax.to(task.device)).to_time_list()

    assert len(samples) == n, "not enough samples"
    x_N = task.model.get_x_N()
    assert x_N != None
    # assert len(x_N) == n, "not enough samples"
    return samples, x_N


# Get run data
data_name, seed, run_path = get_run_data(RUN_ID, WANDB_DIR)

# Get task and datamodule
task, datamodule = get_task(run_path, density=True, data_root=PROJECT_ROOT)

# Get test sequences
test_sequences = []
for (
    batch
) in (
    datamodule.test_dataloader()
):  # batch is set to full test set, but better be safe
    test_sequences = test_sequences + batch.to_time_list()

# Sample event sequences from trained model
store_t_max = []
store_t_max.append(datamodule.tmax)
np.save('/home/cyang314/ucr_work/add-thin/teacher_model/stored_t_max.npy', store_t_max)
samples, x_N = sample_model(task, datamodule.tmax, n=4000)
stored_x_0_list = []
stored_x_N_list = []
pair_size = 2
stored_pair = []

# Create pair size pair data 
# npy formate [x_0, x_N]
for k in range(pair_size):
    store_samples, store_x_N = sample_model(task, datamodule.tmax, n=1000)
    stored_pair.append([store_samples, store_x_N])

stored_np_pair = np.array(stored_pair, dtype=object)
np.save('./stored_pair.npy', stored_np_pair)

# Evaluate metrics against test dataset
mmd = MMD(
    samples,
    test_sequences,
    datamodule.tmax.detach().cpu().item(),
)[0]
wasserstein = lengths_distribution_wasserstein_distance(
    samples,
    test_sequences,
    datamodule.tmax.detach().cpu().item(),
    datamodule.n_max,
)

# Print rounded results for data and seed
print("ADD and Thin density evaluation:")
print("================================")
print(
    f"{data_name} (Seed: {seed}): MMD: {mmd:.3f}, Wasserstein: {wasserstein:.3f}"
)