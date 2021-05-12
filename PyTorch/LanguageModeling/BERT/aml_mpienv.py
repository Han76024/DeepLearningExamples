import os

def set_environment_variables_for_nccl_backend(master_port=6105, verbose=True):
    # setting env variables for multi node runs in AML
    # will hit KeyError Exception when running with single node in AML
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["WORLD_SIZE"]
    )
    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"
    print(
        "NCCL_SOCKET_IFNAME original value = {}".format(
            os.environ["NCCL_SOCKET_IFNAME"]
        )
    )
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    if verbose:
        print("RANK = {}".format(os.environ["RANK"]))
        print("WORLD_SIZE = {}".format(os.environ["WORLD_SIZE"]))
        print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
        print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
        print(
            "NCCL_SOCKET_IFNAME new value = {}".format(os.environ["NCCL_SOCKET_IFNAME"])
        )
        print(f"NCCL_IB_DISABLE = {os.environ['NCCL_IB_DISABLE']}")


def get_rank():
    return int(os.environ["OMPI_COMM_WORLD_RANK"])


def get_local_rank():
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])


def get_global_size():
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])


def get_local_size():
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])


def get_world_size():
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])


def is_main_process():
    return get_rank() == 0
