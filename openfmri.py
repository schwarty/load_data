import os
import re
import csv
import glob
from StringIO import StringIO
from functools import partial
from os.path import exists

import numpy as np
import pandas as pd
import nibabel as nb

from joblib import Memory, Parallel, delayed
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm


def _check_csv_file(f, delimiter):
    possible_delimiters = ['\t', ' ']
    content = f.read()
    new = []
    for line in content.split('\n'):
        for sep in possible_delimiters:
            line = re.sub('%s+' % sep, delimiter, line).strip()
        new.append(line)
    return StringIO('\n'.join(new))


def _csv_to_dict(path, key_end_pos=1, join_values=False, delimiter=' ', quotechar='"'):
    if join_values and isinstance(join_values, bool):
        join_values = '_'

    data = []
    with open(path, 'rb') as f:
        # look for stupid separators
        f = _check_csv_file(f, delimiter=delimiter)
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            if len(row) == 2:
                data.append(row)
            else:
                key = '_'.join(row[:key_end_pos])
                value = row[key_end_pos:]
                if len(value) == 1:
                    value = value[0]
                if join_values:
                    value = join_values.join(value)
                data.append((key, value))
    return dict(data)
    

def _collect_openfmri_dataset(study_dir, img_ext='nii.gz'):
    # parse study-level metadata files
    dataset = {}
    join_study = partial(os.path.join, study_dir)
    study_id = os.path.basename(study_dir.strip())
    dataset['study_id'] = study_id

    if exists(join_study('scan_key.txt')):
        dataset.update(_csv_to_dict(join_study('scan_key.txt')))

    if exists(join_study('study_key.txt')):
        with open(join_study('study_key.txt'), 'rb') as f:
            dataset['name'] = f.read().strip()

    if exists(join_study('task_key.txt')):
        dataset.update(_csv_to_dict(join_study('task_key.txt'), key_end_pos=1, join_values=' '))

    # parse model-level metadata files
    model = {}
    for model_dir in sorted(glob.glob(join_study('models', '*'))):
        join_model = partial(os.path.join, model_dir)
        model_id = os.path.basename(model_dir.strip())

        model[model_id] = {}
        if exists(join_model('condition_key.txt')):
            model[model_id]['conditions'] = _csv_to_dict(
                join_model('condition_key.txt'), key_end_pos=2)

        if exists(join_model('task_contrasts.txt')):
            model[model_id]['contrasts'] = _csv_to_dict(
                join_model('task_contrasts.txt'), key_end_pos=2)

        model[model_id]['orthogonalize'] = None
        if exists(join_model('orthogonalize.txt')):
            model[model_id]['orthogonalize'] = pd.DataFrame.from_csv(
                join_model('orthogonalize.txt'), header=-1, sep=' ')
            
    # parse subject-level files and align it with study and model
    structural = []
    functional = []
    conditions = {}
    contrasts = {}

    for subject_dir in sorted(glob.glob(join_study('sub???'))):
        join_subject = partial(os.path.join, subject_dir)
        subject_id = os.path.basename(subject_dir.strip())

        # Anatomy data
        preproc_anat = {}
        anat_files = join_subject('model', '*', 'anatomy', 'highres001.%s' % img_ext)
        for anat_file in sorted(glob.glob(anat_files)):
            model_id, _, _ = anat_file.split(os.path.sep)[-3:]
            preproc_anat.setdefault('study', []).append(study_id)
            preproc_anat.setdefault('subject', []).append(subject_id)
            preproc_anat.setdefault('model', []).append(model_id)
            preproc_anat.setdefault('preproc_anatomy', []).append(anat_file)
        preproc_anat = pd.DataFrame(preproc_anat)

        raw_anat = {}
        anat_file = join_subject('anatomy', 'highres001.%s' % img_ext)
        anat_file = anat_file if os.path.exists(anat_file) else np.nan
        raw_anat.setdefault('study', []).append(study_id)
        raw_anat.setdefault('subject', []).append(subject_id)
        raw_anat.setdefault('raw_anatomy', []).append(anat_file)
        raw_anat = pd.DataFrame(raw_anat)

        if raw_anat.shape[0] == 0:
            anat = preproc_anat
        elif preproc_anat.shape[0] == 0:
            anat = raw_anat
        else:  
            anat = preproc_anat.merge(raw_anat)
        structural.append(anat)

        # BOLD data
        preproc_func = {}
        bold_files = join_subject('model', '*', 'BOLD', '*', 'bold.%s' % img_ext)
        for bold_file in sorted(glob.glob(bold_files)):
            model_id, _, session_id, _ = bold_file.split(os.path.sep)[-4:]
            task_id, run_id = session_id.split('_')
            movement_file = os.path.join(os.path.split(bold_file)[0], 'motion.txt')
            movement_file = movement_file if exists(movement_file) else np.nan
            preproc_func.setdefault('study', []).append(study_id)
            preproc_func.setdefault('subject', []).append(subject_id)
            preproc_func.setdefault('model', []).append(model_id)
            preproc_func.setdefault('task', []).append(task_id)
            preproc_func.setdefault('task_name', []).append(dataset.get(task_id, task_id))            
            preproc_func.setdefault('run', []).append(run_id)
            preproc_func.setdefault('preproc_bold', []).append(bold_file)
            preproc_func.setdefault('movement', []).append(movement_file)
            preproc_func.setdefault('TR', []).append(dataset.get('TR', np.nan))
        preproc_func = pd.DataFrame(preproc_func)

        raw_func = {}
        bold_files = join_subject('BOLD', '*', 'bold.%s' % img_ext)
        for bold_file in sorted(glob.glob(bold_files)):
            session_id, _ = bold_file.split(os.path.sep)[-2:]
            task_id, run_id = session_id.split('_')
            raw_func.setdefault('study', []).append(study_id)
            raw_func.setdefault('subject', []).append(subject_id)
            raw_func.setdefault('task', []).append(task_id)
            raw_func.setdefault('task_name', []).append(dataset.get(task_id, task_id))            
            raw_func.setdefault('run', []).append(run_id)
            raw_func.setdefault('raw_bold', []).append(bold_file)
            raw_func.setdefault('TR', []).append(dataset.get('TR', np.nan))
        raw_func = pd.DataFrame(raw_func)

        if raw_func.shape[0] == 0:
            func = preproc_func
        elif preproc_func.shape[0] == 0:
            func = raw_func
        else:  
            func = preproc_func.merge(raw_func)
        functional.append(func)

        # condition files
        cond_files = join_subject('model', '*', 'onsets', '*', '*.txt')
        for cond_file in sorted(glob.glob(cond_files)):
            model_id, _, session_id, cond_id = cond_file.split(os.path.sep)[-4:]
            task_id, run_id = session_id.split('_')
            cond_id = cond_id[:-4]
 
            cond_name = model.get(model_id, model['model001'])['conditions'].get('%s_%s' % (task_id, cond_id), cond_id)
            conditions.setdefault('study', []).append(study_id)
            conditions.setdefault('subject', []).append(subject_id)
            conditions.setdefault('model', []).append(model_id)
            conditions.setdefault('task', []).append(task_id)
            conditions.setdefault('task_name', []).append(dataset.get(task_id, task_id))            
            conditions.setdefault('run', []).append(run_id)
            conditions.setdefault('condition', []).append(cond_id)
            conditions.setdefault('condition_name', []).append(cond_name)
            conditions.setdefault('condition_file', []).append(cond_file)

        # contrast files
        contrast_files = join_subject('model', '*', '*_maps', '*.%s' % img_ext)
        for contrast_file in sorted(glob.glob(contrast_files)):
            model_id, dtype, contrast_id = contrast_file.split(os.path.sep)[-3:]
            contrast_id = contrast_id.split('.%s' % img_ext)[0]
            contrasts.setdefault('study', []).append(study_id)
            contrasts.setdefault('subject', []).append(subject_id)
            contrasts.setdefault('model', []).append(model_id)
            contrasts.setdefault('dtype', []).append(dtype)
            contrasts.setdefault('contrast', []).append(contrast_id)
            contrasts.setdefault('contrast_file', []).append(contrast_file)

    return (dataset, model,
            pd.concat(structural, ignore_index=True),
            pd.concat(functional, ignore_index=True),
            pd.DataFrame(conditions), pd.DataFrame(contrasts))


def collect_openfmri(study_dirs, img_ext='nii.gz', memory=Memory(None), n_jobs=1):
    """Collect paths and identifiers of OpenfMRI datasets.

       Parameters
       ----------
       study_dirs: list
           The list of the datasets paths.
       n_jobs: int
           Number of jobs.

       Returns
       -------
       structual: DataFrame with anat images.
       functional: DataFrame with func images.
       conditions: DataFrame with conditions files.
       contrasts: DataFrame with subject-level contrasts

       Warning
       -------
       All the files from the openfmri structure are not yet collected.
       Among those: motion.txt, orthogonalize.txt
    """
    if isinstance(study_dirs, basestring):
        study_dirs = glob.glob(study_dirs)

    results = Parallel(n_jobs=n_jobs, pre_dispatch='n_jobs')(
        delayed(memory.cache(_collect_openfmri_dataset))(study_dir, img_ext=img_ext)
        for study_dir in study_dirs)
    datasets = [r[0] for r in results]
    models = [r[1] for r in results]
    structural = pd.concat([r[2] for r in results], ignore_index=True)
    functional = pd.concat([r[3] for r in results], ignore_index=True)
    conditions = pd.concat([r[4] for r in results], ignore_index=True)
    contrasts = pd.concat([r[5] for r in results], ignore_index=True)

    # merge datasets and models
    datasets_ = {}
    for dataset, model in zip(datasets, models):
        dataset['models'] = model
        datasets_[dataset['study_id']] = dataset

    return datasets_, structural, functional, conditions, contrasts


def load_glm_inputs(study_dirs, hrf_model='canonical', drift_model='cosine',
                    img_ext='nii.gz', memory=Memory(None), n_jobs=1):
    """Returns data (almost) ready to be used for a GLM.
    """
    datasets, structural, functional, conditions, contrasts = \
        collect_openfmri(study_dirs, img_ext=img_ext, memory=memory, n_jobs=n_jobs)

    main = functional.merge(conditions)

    # computing design matrices
    print 'Computing models...'
    results = Parallel(n_jobs=n_jobs, pre_dispatch='n_jobs')(
        delayed(memory.cache(_make_design_matrix))(
            run_df, hrf_model, drift_model, orthogonalize=datasets[group_id[0]]['models'][group_id[2]]['orthogonalize'])
        for group_id, group_df in main.groupby(['study', 'subject', 'model'])
        for run_id, run_df in group_df.groupby(['task', 'run'])
        )

    # collect results
    print 'Collecting...'
    glm_inputs = {}
    for group_id, group_df in main.groupby(['study', 'subject', 'model']):
        study_id, subject_id, model_id = group_id
        for session_id, run_df in group_df.groupby(['task', 'run']):
            task_id, run_id = session_id
            bold_file, dm = results.pop(0)        
            glm_inputs.setdefault(group_id, {}).setdefault('bold', []).append(bold_file)
            glm_inputs.setdefault(group_id, {}).setdefault('design', []).append(dm)
        glm_inputs.setdefault(group_id, {}).setdefault(
            model_id, _make_contrasts(datasets, study_id, model_id, hrf_model, group_df))
        glm_inputs.setdefault(group_id, {}).setdefault(
            '%s_per_run' % model_id, _make_contrasts(
                datasets, study_id, model_id, hrf_model, group_df, per_run=True))
    return glm_inputs


def _make_contrasts(datasets, study_id, model_id, hrf_model, group_df, per_run=False):
    contrasts = {}
    model_contrasts = datasets[study_id]['models'][model_id]['contrasts']
    if per_run:
        model_contrasts_ = {}
        for session_id, run_df in group_df.groupby(['task', 'run']):
            task_id, run_id = session_id
            for con_id in model_contrasts:
                new_con_id = '%s_%s_%s' % (
                    con_id.split('_', 1)[0], run_id, con_id.split('_', 1)[1])
                if new_con_id.startswith(task_id):
                    model_contrasts_[new_con_id] = model_contrasts[con_id]
        model_contrasts = model_contrasts_

    for session_id, run_df in group_df.groupby(['task', 'run']):
        task_id, run_id = session_id

        for con_id in model_contrasts:
            con_val = model_contrasts[con_id]
            con_val = np.array(con_val).astype(np.float)
            if 'derivative' in hrf_model:
                con_val = np.insert(con_val, np.arange(con_val.size) + 1, 0).tolist()
            if (not con_id.startswith(task_id) and not per_run) or (
                (not con_id.startswith(task_id) or not run_id in con_id) and per_run):
                con_val = None
            contrasts.setdefault(con_id, []).append(con_val)

    return contrasts
    

def load_classification_inputs(study_dirs, img_ext='nii.gz', memory=Memory(None), n_jobs=1):
    """Returns data (almost) ready to be used for a
    classification of the timeseries.
    """
    # datasets, structural, functional, conditions, contrasts = \
    #     collect_openfmri(study_dirs, img_ext=img_ext, memory=memory, n_jobs=n_jobs)

    # functional['movement'] = np.nan
    glm_inputs = load_glm_inputs(study_dirs, hrf_model='canonical', drift_model='blank',
                               img_ext=img_ext, memory=memory, n_jobs=n_jobs)
    classif_inputs = {}
    for key in glm_inputs:
        classif_inputs[key] = {}
        classif_inputs[key]['bold'] = glm_inputs[key]['bold']
        classif_inputs[key]['target'] = []
        for dm in glm_inputs[key]['design']:
            cols = dm.columns.values
            cols = [col for col in cols if 'movement' not in col and 'constant' not in col]
            dm = dm[cols].values
            p = np.percentile(dm, 85)
            target = pd.DataFrame(dict(zip(cols, (dm > p).T.astype('int'))))
            classif_inputs[key]['target'].append(target)
    return classif_inputs


def _make_design_matrix(run_frame, hrf_model='canonical', drift_model='cosine', orthogonalize=None):
    # print ' %s' % ' '.join(run_frame[['study', 'subject', 'model', 'task', 'run']].values[0])
    bold_file = run_frame.preproc_bold.unique()[0]
    n_scans = nb.load(bold_file).shape[-1]
    tr = float(run_frame.TR.unique()[0])
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    movement_regressors = run_frame.movement.unique()[0]
    movement_regressors = np.recfromtxt(movement_regressors) \
        if isinstance(movement_regressors, basestring) else None
    names = []
    times = []
    durations = []
    amplitudes = []
    for condition_id, condition_name, condition_file in run_frame[['condition', 'condition_name', 'condition_file']].values:
        conditions = _csv_to_dict(condition_file)
        if condition_id == 'empty_evs':
            conditions = sorted(conditions.keys())
            for c in conditions:
                names.append(['cond%03i' % int(c)])
                times.append([0])
                durations.append([0])
                amplitudes.append([0])
        else:
            keys = sorted(conditions.keys())
            times.append(np.array(keys).astype('float'))
            names.append(['%s_%s' % (condition_id, condition_name)] * len(keys))
            durations.append([float(conditions[k][0]) for k in keys])
            amplitudes.append([float(conditions[k][1]) for k in keys])

    times = np.concatenate(times).ravel()
    order = np.argsort(times)
    times = times[order]
    names = np.concatenate(names)[order]
    durations = np.concatenate(durations)[order]
    amplitudes = np.concatenate(amplitudes)[order]

    if durations.sum() == 0:
        paradigm = EventRelatedParadigm(names, times, amplitudes)
    else:
        paradigm = BlockParadigm(names, times, durations, amplitudes)

    if movement_regressors is None:
        design_matrix = make_dmtx(
            frametimes, paradigm, hrf_model=hrf_model,
            drift_model=drift_model)
    else:
        mov_reg_names = ['movement_%i' % r
                     for r in range(movement_regressors.shape[1])]
        design_matrix = make_dmtx(
            frametimes, paradigm, hrf_model=hrf_model,
            drift_model=drift_model,
            add_regs=movement_regressors, add_reg_names=mov_reg_names)

    # orthogonalize regressors
    if orthogonalize is not None and not 'derivative' in hrf_model:
        task_id = run_frame.task.unique()[0]

        for x, y in orthogonalize[orthogonalize.index == task_id].values:
            x_ = design_matrix.matrix[:, x]
            y_ = design_matrix.matrix[:, y]
            z = _orthogonalize_vectors(x_, y_)
            design_matrix.matrix[:, x] = z

    return bold_file, pd.DataFrame(dict(zip(design_matrix.names, design_matrix.matrix.T)))


def _orthogonalize_vectors(x, y):
    x = np.array(x)
    y = np.array(y)

    s = np.dot(x, y) / np.sum(y ** 2)

    return (x - y * s)


if __name__ == '__main__':
    memory = Memory('/storage/workspace/yschwart/cache')
    base_dir = '/storage/workspace/yschwart/new_brainpedia'

    # glob intra_stats folders to get contrasts
    study_dirs = sorted(glob.glob(os.path.join(base_dir, 'intra_stats', 'ds001')))
    _, _, _, _, contrasts = collect_openfmri(study_dirs, memory=memory, n_jobs=-1)

    # glob preproc folders
    study_dirs = sorted(glob.glob(os.path.join(base_dir, 'preproc', 'ds001')))
    datasets, structural, functional, conditions, _ = collect_openfmri(study_dirs, memory=memory, n_jobs=-1)

    # we can merge dataframes!
    df = functional.merge(conditions)

    # we can filter the dataframes!
    functional = functional[functional.study == 'ds001']

    # computes design matrices for the given dataframes
    glm_inputs = load_glm_inputs(study_dirs, memory=memory, n_jobs=1)
    glm_inputs = load_glm_inputs(study_dirs, hrf_model='canonical with derivative', memory=memory, n_jobs=-1)

    # computes classification targets for the given dataframes
    classif_inputs = load_classification_inputs(study_dirs, memory=memory, n_jobs=-1)
