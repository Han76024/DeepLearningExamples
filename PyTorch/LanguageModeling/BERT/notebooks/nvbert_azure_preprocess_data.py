from azureml.core import Workspace, Datastore, Dataset, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import MpiConfiguration
from azureml.data import OutputFileDatasetConfig

subscription_id = <<REPLACE ME>>
resource_group = <<REPLACE ME>>
workspace_name = <<REPLACE ME>>
gpu_cluster_name = <<REPLACE ME>>

nnodes = 1
gpus_per_node = 1

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
                                                 datastore_name=<<REPLACE ME>>,
                                                 container_name=<<REPLACE ME>>,
                                                 account_name=<<REPLACE ME>>
                                                 account_key=<<REPLACE ME>>
                                                 create_if_not_exists=True
                                                 )

pytorch_env = Environment(name=<<REPLACE ME>>)
pytorch_env.docker.enabled = True
pytorch_env.docker.base_image = <<REPLACE ME>>
pytorch_env.python.user_managed_dependencies = True
pytorch_env.docker.base_image_registry.address = <<REPLACE ME>>
pytorch_env.docker.base_image_registry.username = <<REPLACE ME>>
pytorch_env.docker.base_image_registry.password = <<REPLACE ME>>
pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'


def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_mount()


def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()


# ----------------------------------------- #

# action = 'download'
# action = 'text_formatting'
action = 'sharding'
# action = 'create_hdf5_files'
# action = 'sharding'

dataset = 'bookscorpus'
# dataset = 'wikicorpus_en'


# ----------------------------------------- #
qadMapping = {"bookscorpus": "bookcorpus/bookcorpus.txt", "wikicorpus_en": "wikipedia"}


script_params = [
    '--action', action,
    '--dataset', dataset,
    '--input_files', get_input_dataset(cog_ds, qadMapping[dataset], 'train_file'),
    '--working_dir', get_output_dataset(cog_ds, <<REPLACE ME>>, 'BERT_data_working_dir'),
    '--max_seq_length', 128,
    '--vocab_file', '',
    '--interactive_json_config_generator', ''
]

print("creating ScriptRunConfig...")
src = ScriptRunConfig(
    source_directory='../',
    script='data/bertPrep.py',
    arguments=script_params,
    compute_target=gpu_compute_target,
    distributed_job_config=MpiConfiguration(process_count_per_node=gpus_per_node, node_count=nnodes),
    environment=pytorch_env,
)

print("submitting experiment...")
experiment = Experiment(ws, <<REPLACE ME>>)
run = experiment.submit(src)

print(f"{run.get_portal_url()}")
