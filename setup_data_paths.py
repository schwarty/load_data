import glob
import os.path
from pandas import DataFrame
import pandas
import copy


def get_all_paths(data_set=None, root_dir="/storage/data"):
    # TODO
    # if data_set ... collections.Sequence
    # iterate over list
    if data_set is None:
        data_set = ["hcp", "henson2010faces", "ds105", "ds107"]
        root_dir = [os.path.join(root_dir, "data"),
                    os.path.join(root_dir,
                                 "storage/workspace/brainpedia/preproc"),
                    os.path.join(root_dir,
                                 "storage/workspace/brainpedia/preproc"),
                    os.path.join(root_dir,
                                 "storage/workspace/brainpedia/preproc")]
    if hasattr(data_set, "__iter__"):
        df_ = list()
        for (ds, rd) in zip(data_set, root_dir):
            df_.append(get_all_paths(data_set=ds, root_dir=rd))
        df = pandas.concat(df_, keys=data_set)
    elif data_set.startswith("ds") or data_set == "henson2010faces":
        df = collect_data_openfmri(os.path.normpath(os.path.join(
            root_dir, data_set)))
    elif data_set == "hcp":
        df = collect_data_hcp(os.path.normpath(os.path.join(
            root_dir, "HCP/Q2/")))
    return df


def collect_data_hcp(base_path):
    list_ = list()
    for fun_path in sorted(glob.glob(os.path.join(
            base_path, "*/MNINonLinear/Results/", "*/*.nii.gz"))):

        head, tail_ = os.path.split(fun_path)
        if head[-2:] not in ["LR", "RL"]:
            continue
        tail = list()
        while tail_:
            tail.append(tail_)
            head, tail_ = os.path.split(head)
        if tail[0][:-7] != tail[1]:
            continue
        subj_id = tail[4]
        task = tail[1][6:-3]
        if tail[1].startswith("rfMRI"):
            run = task[-1]
            task = task[:-1]
        mode = tail[1][-2:]

        anat = os.path.join(base_path, subj_id, "MNINonLinear/T1w.nii.gz")

        confds = os.path.join(os.path.split(fun_path)[0],
                              "Movement_Regressors.txt")
        list_.append({"subj_id": subj_id,
                      "task": task,
                      "mode": mode,
                      "func": fun_path,
                      "anat": anat,
                      "confds": confds,
                      "TR": 0.72})
        if tail[1].startswith("rfMRI"):
            list_[-1]["run"] = run
        else:
            onsets = [onset for onset in glob.glob(os.path.join(
                os.path.split(fun_path)[0], "EVs/*.txt"))
                if os.path.split(onset)[1][0] != "S"]
            list_[-1]["onsets"] = onsets
    return DataFrame(list_)


def collect_data_openfmri(base_path):
    """
    """
    list_ = list()
    # read conditions from file
    with open(os.path.join(base_path,
                           "models",
                           "model001",
                           "condition_key.txt")) as f:
        conditions = list()
        while True:
            try:
                line = f.readline()
                if not line:
                    raise StopIteration
                if len(line.split()) > 3:
                    conditions.append(" ".join(line.split()[2:]))
                else:
                    conditions.append(line.split()[2])
            except StopIteration:
                break
    # read scan repeat time from file
    with open(os.path.join(base_path, "scan_key.txt")) as file_:
        TR = file_.readline()[3:-1]  # last char is linefeed
    cnt = 0
    for fun_path in sorted(glob.glob(
        os.path.join(base_path,
                     "sub*/model/model*/BOLD/task*/bold.nii.gz"))):
        head, tail_ = os.path.split(fun_path)
        tail = list()
        while tail_:
            tail.append(tail_)
            head, tail_ = os.path.split(head)
        subj_id = tail[5][-3:]
        model = tail[3][-3:]
        task, run = tail[1].split("_")

        tmp_base = os.path.split(os.path.split(os.path.split(
            fun_path)[0])[0])[0]

        anat = os.path.join(tmp_base,
                            "anatomy",
                            "highres{}.nii.gz".format(model[-3:]))

        onsets = glob.glob(os.path.join(tmp_base, "onsets",
                                        "{}_{}".format(task, run),
                                        "cond*.txt"))

        confds = os.path.join(os.path.split(fun_path)[0], "motion.txt")
        tmp_dict = ({"subj_id": subj_id,
                     "model": model,
                     "task": task[-3:],
                     "run": run[-3:],
                     "func": fun_path,
                     "anat": anat,
                     "confds": confds,
                     "TR": TR,
                     "region_ix": cnt})
        if onsets:
            for onset in onsets:
                tmp_dict_ = tmp_dict
                tmp_dict_["cond_onsets"] = onset
                ix = int(onset[-7:-4]) - 1
                tmp_dict_["condition"] = conditions[ix]
                list_.append(copy.copy(tmp_dict_))
            cnt += len(onsets)
        else:
            list_.append(copy.copy(tmp_dict))
            cnt += 1
    return DataFrame(list_)
