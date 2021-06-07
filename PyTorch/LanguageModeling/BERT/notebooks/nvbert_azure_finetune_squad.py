import azureml.core
from azureml.core import Workspace, Datastore, Dataset, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import MpiConfiguration
from azureml.telemetry import set_diagnostics_collection
from azureml.data import OutputFileDatasetConfig
from azureml.exceptions import ComputeTargetException

print("SDK version:", azureml.core.VERSION)
set_diagnostics_collection(send_diagnostics=True)

subscription_id = '<<REPLACE ME>>'
resource_group = '<<REPLACE ME>>'
workspace_name = '<<REPLACE ME>>'
gpu_cluster_name = '<<REPLACE ME>>'

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


# --------------------------------------------------------------------- #
# squad data is prepared using script data/squad/squad_download.sh
# or can be directly downloaded from its official site https://rajpurkar.github.io/SQuAD-explorer/
# download bert-base-uncased vocabulary file from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
# download bert-base-uncased checkpoint from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz unzip to get pytorch_model.bin file
# --------------------------------------------------------------------- #

ds = Datastore.register_azure_blob_container(workspace=ws,
                                             datastore_name='<<REPLACE ME>>',
                                             container_name='<<REPLACE ME>>',
                                             account_name='<<REPLACE ME>>',
                                             account_key='<<REPLACE ME>>',
                                             create_if_not_exists=True
                                             )

num_nodes = 4
gpus_per_node = 4
output_suffix = "squad_output_1"

pytorch_env = Environment(name='nvbert_env')
pytorch_env.docker.enabled = True
pytorch_env.docker.base_image = '<<REPLACE ME>>'
pytorch_env.python.user_managed_dependencies = True
pytorch_env.docker.base_image_registry.address = '<<REPLACE ME>>'
pytorch_env.docker.base_image_registry.username = '<<REPLACE ME>>'
pytorch_env.docker.base_image_registry.password = '<<REPLACE ME>>'
pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'


def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_mount()


def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()


# Training hparams
script_params = [
    "--do_train",
    "--train_file", get_input_dataset(ds, '<<REPLACE ME>>' + "/squad_dir/train-v1.1.json", "train_file"),
    "--train_batch_size", 8,
    "--gradient_accumulation_steps", 4,
    "--do_predict",
    "--predict_file", get_input_dataset(ds, '<<REPLACE ME>>' + "/squad_dir/dev-v1.1.json", "predict_file"),
    "--predict_batch_size", 2,
    "--eval_script", get_input_dataset(ds, '<<REPLACE ME>>' + "/squad_dir/evaluate-v1.1.py", "eval_script"),
    "--do_eval",
    "--do_lower_case",
    "--bert_model", "bert-base-uncased",
    "--learning_rate", 3e-5,
    "--seed", 42,
    "--num_train_epochs", 8,
    "--max_seq_length", 384,
    "--doc_stride", 128,
    # "--version_2_with_negative",
    "--output_dir", get_output_dataset(ds, '<<REPLACE ME>>' + output_suffix + "/output_dir", "output_dir"),
    "--json_summary", get_output_dataset(ds, '<<REPLACE ME>>' + output_suffix + "/dllogger_json_summary", "json_summary"),
    "--vocab_file", get_input_dataset(ds, '<<REPLACE ME>>' + "squad_dir/bert-base-uncased-vocab.txt", "vocab_file"),
    "--config_file", "data/squad/bert_config.json",
    "--init_checkpoint", get_input_dataset(ds, '<<REPLACE ME>>' + "/squad_dir/pytorch_model.bin", "init_checkpoint"),
    # "--max_steps", 1000,
    "--fp16",
    "--skip_cache",
]

print("creating ScriptRunConfig...")
src = ScriptRunConfig(
    source_directory='../',
    script='run_squad.py',
    arguments=script_params,
    compute_target=gpu_compute_target,
    distributed_job_config=MpiConfiguration(process_count_per_node=gpus_per_node, node_count=num_nodes),
    environment=pytorch_env,
)

print("submitting experiment...")
experiment = Experiment(ws, 'nvidia-bert-finetune-0')
run = experiment.submit(src)

print(f"\n{run.get_portal_url()}")
