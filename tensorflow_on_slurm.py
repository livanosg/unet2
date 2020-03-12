import re
import os


def tf_config_from_slurm(ps_number, port_number=2222):
    """
    Creates configuration for a distributed tensorflow session
    from environment variables  provided by the Slurm cluster
    management system.

    @param: ps_number number of parameter servers to run
    @param: port_number port number to be used for communication
    @return: a tuple containing cluster with fields cluster_spec,
             task_name and task_id
    """
    # Based on https://github.com/deepsense-ai/tensorflow_on_slurm/blob/master/tensorflow_on_slurm/tensorflow_on_slurm.py
    # TODO make it work with evaluator. Not tested yet

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    nodename = os.environ["SLURMD_NODENAME"]
    nodelist = _expand_nodelist(nodelist)
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))

    if len(nodelist) != num_nodes:
        raise ValueError("Number of slurm nodes {} not equal to {}".format(len(nodelist), num_nodes))

    if nodename not in nodelist:
        raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(nodename, nodelist))

    eval_node = [node for index, node in enumerate(nodelist) if index < 1]
    worker_nodes = [node for index, node in enumerate(nodelist) if index >= ps_number]

    if nodename in eval_node:
        my_job_name = "evaluator"  # ps to evaluator
        my_task_index = eval_node.index(nodename)
    else:
        my_job_name = "worker"
        my_task_index = worker_nodes.index(nodename)

    worker_sockets = [":".join([node, str(port_number)]) for node in worker_nodes]
    eval_sockets = [":".join([node, str(port_number)]) for node in eval_node]
    # noinspection PyShadowingNames
    cluster = {"worker": worker_sockets, "evaluator": eval_sockets}

    return cluster, my_job_name, my_task_index


def _pad_zeros(iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)


# noinspection PyShadowingBuiltins
def _expand_ids(ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            token = [int(token) for token in id.split('-')]
            begin, end = token
            result.extend(_pad_zeros(range(begin, end + 1), len(token)))
        else:
            result.append(id)
    return result


def _expand_nodelist(nodelist):
    prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
    ids = _expand_ids(ids)
    # noinspection PyShadowingBuiltins
    result = [prefix + str(id) for id in ids]
    return result


def _worker_task_id(nodelist, nodename):
    return nodelist.index(nodename)
