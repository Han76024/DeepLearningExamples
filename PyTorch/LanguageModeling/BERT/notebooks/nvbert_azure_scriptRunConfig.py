import azureml.core
from azureml.core import Workspace, Datastore, Dataset, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import MpiConfiguration
from azureml.telemetry import set_diagnostics_collection
from azureml.data import OutputFileDatasetConfig
from azureml.exceptions import ComputeTargetException

print("SDK version:", azureml.core.VERSION)
set_diagnostics_collection(send_diagnostics=True)

subscription_id = <<REPLACE ME>>
resource_group = <<REPLACE ME>>
workspace_name = <<REPLACE ME>>
gpu_cluster_name = <<REPLACE ME>>

ws = Workspace(subscription_id, resource_group, workspace_name)
ws_details = ws.get_details()
print('Name:\t\t{}\nLocation:\t{}'.format(ws_details['name'], ws_details['location']))

try:
    gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    if gpu_compute_target.provisioning_state == 'Failed':
        gpu_compute_target.delete()
        gpu_compute_target.wait_for_completion(show_output=True)
        raise ComputeTargetException('failed cluster')
    print('Found existing compute target.')
except ComputeTargetException:
    print('cannot find compute target...')
    gpu_compute_target = None

# Use the 'status' property to get a detailed status for the current cluster.
import pprint as pp

if gpu_compute_target is not None:
    pp.pprint(gpu_compute_target.status.serialize())
else:
    raise ValueError("gpu_compute_target not found")

cog_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                 datastore_name=<<REPLACE ME>>
                                                 container_name=<<REPLACE ME>>,
                                                 account_name=<<REPLACE ME>>,
                                                 account_key=<<REPLACE ME>>,
                                                 create_if_not_exists=True
                                                 )

from math import ceil

num_nodes = 4
gpus_per_node = 4
# total training records = 226637477

total_tokens_40epochs = 65598914560
warmup_proportion = 0.2843
num_steps_per_checkpoint = 10000
random_seed = 42

# --------------------------------------- #
# profile
# --------------------------------------- #
profile = "deepspeed"
# --------------------------------------- #

profiles_dict = {
    "deepspeed": {
        "outputSuffix": "_deepspeed_7",
        "bsz_per_gpu": 64,  # need to update train_batch_size in deepspeedConfig.json as well
    },
    "non_deepspeed": {
        "outputSuffix": "_non_deepspeed_0",
        "bsz_per_gpu": 32,
    }

}

# ====================
# Phase 1 per-training
# ====================
outputSuffix = profiles_dict[profile]["outputSuffix"]

lr = 1.875e-4
num_accumulation_steps = 1
bsz_per_gpu = profiles_dict[profile]["bsz_per_gpu"]
train_batch_size = bsz_per_gpu * num_accumulation_steps
seq_len = 128
max_predictions_per_seq = 20
# warmup_steps = 2000
gbs_phase1 = num_accumulation_steps * bsz_per_gpu * num_nodes * gpus_per_node  # 65536
print(f"Phase 1 global batch size: {gbs_phase1}")

# phase_1: 90% of train steps
train_steps = ceil(0.9 * total_tokens_40epochs / gbs_phase1 / seq_len)  # 900864
print(f"Phase 1 number of train steps: {train_steps}")

pytorch_env = Environment(name='myenv')
pytorch_env.docker.enabled = True
pytorch_env.docker.base_image = <<REPLACE ME>>
pytorch_env.python.user_managed_dependencies = True
pytorch_env.docker.base_image_registry.address = <<REPLACE ME>>
pytorch_env.docker.base_image_registry.username = <<REPLACE ME>>
pytorch_env.docker.base_image_registry.password = <<REPLACE ME>>
pytorch_env.python.interpreter_path =  '/opt/miniconda/bin/python'


def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_mount()


def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()


# Training hparams
script_params = [
    '--train_batch_size', train_batch_size,
    '--learning_rate', lr,
    '--warmup_proportion', warmup_proportion,
    '--input_dir', get_input_dataset(cog_ds, 'inputs/', 'train_file'),
    '--max_seq_length', seq_len,
    '--max_predictions_per_seq', max_predictions_per_seq,
    '--max_steps', train_steps,
    '--num_steps_per_checkpoint', num_steps_per_checkpoint,
    '--seed', random_seed,
    '--do_train',
    '--config_file', "bert_config.json",
    '--output_dir', get_output_dataset(cog_ds, 'output' + outputSuffix, 'output_dir_' + outputSuffix),
    '--json-summary', get_output_dataset(cog_ds, 'output' + outputSuffix + '/dllogger', 'output_dir_' + outputSuffix + "_dllogger"),
    '--fp16',
    '--allreduce_post_accumulation',
    # '--allreduce_post_accumulation_fp16',
    '--gradient_accumulation_steps', num_accumulation_steps,
    '--log_freq', 100,
    '--local_rank', '$AZ_BATCHAI_TASK_INDEX',

    '--deepspeed_config', get_input_dataset(cog_ds, 'deepspeedConfig.json', 'deepspeed_config')
]

if profile == "deepspeed":
    script_params.append('--deepspeed')

print("creating ScriptRunConfig...")
src = ScriptRunConfig(
    source_directory='../',
    script='run_pretraining.py',
    arguments=script_params,
    compute_target=gpu_compute_target,
    distributed_job_config=MpiConfiguration(process_count_per_node=gpus_per_node, node_count=num_nodes),
    environment=pytorch_env,
)

print("submitting experiment...")
experiment = Experiment(ws, 'test-nvidia-bert')
run = experiment.submit(src)

print(f"\n{run.get_portal_url()}")
