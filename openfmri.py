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

from joblib import Parallel, delayed
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
        # look for studid separators
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


def _collect_openfmri_dataset(study_dir):
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
        model[model_id]['conditions'] = _csv_to_dict(
            join_model('condition_key.txt'), key_end_pos=2)

        model[model_id]['contrasts'] = _csv_to_dict(
            join_model('task_contrasts.txt'), key_end_pos=2)

    # parse subject-level files and align it with study and model
    structural = {}
    functional = {}
    conditions = {}
    contrasts = {}

    for subject_dir in sorted(glob.glob(join_study('sub???'))):
        join_subject = partial(os.path.join, subject_dir)
        subject_id = os.path.basename(subject_dir.strip())

        # Anatomy data
        anat_file = join_subject('anatomy', 'highres001.nii.gz')
        anat_file = anat_file if os.path.exists(anat_file) else np.nan
        structural.setdefault('study', []).append(study_id)
        structural.setdefault('subject', []).append(subject_id)
        structural.setdefault('anatomy', []).append(anat_file)

        # BOLD data
        bold_files = join_subject('model', '*', 'BOLD', '*', 'bold.nii.gz')
        n_bold = 0
        for bold_file in sorted(glob.glob(bold_files)):
            model_id, _, session_id, _ = bold_file.split(os.path.sep)[-4:]
            raw_bold_file = join_subject('BOLD', session_id, 'bold.nii.gz')
            raw_bold_file = raw_bold_file if os.path.exists(raw_bold_file) else np.nan
            task_id, run_id = session_id.split('_')

            functional.setdefault('study', []).append(study_id)
            functional.setdefault('subject', []).append(subject_id)
            functional.setdefault('model', []).append(model_id)
            functional.setdefault('task', []).append(task_id)
            functional.setdefault('task_name', []).append(dataset[task_id])            
            functional.setdefault('run', []).append(run_id)
            functional.setdefault('raw_bold', []).append(raw_bold_file)
            functional.setdefault('preproc_bold', []).append(bold_file)
            functional.setdefault('TR', []).append(dataset['TR'])
            n_bold += 1

        if n_bold == 0:
            bold_files = join_subject('BOLD', '*', 'bold.nii.gz')
            for bold_file in sorted(glob.glob(bold_files)):
                functional.setdefault('study', []).append(study_id)
                functional.setdefault('subject', []).append(subject_id)
                functional.setdefault('model', []).append(model_id)
                functional.setdefault('task', []).append(task_id)
                functional.setdefault('task_name', []).append(dataset[task_id])            
                functional.setdefault('run', []).append(run_id)
                functional.setdefault('raw_bold', []).append(bold_file)
                functional.setdefault('preproc_bold', []).append(np.nan)
                functional.setdefault('TR', []).append(dataset['TR'])

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
            conditions.setdefault('task_name', []).append(dataset[task_id])            
            conditions.setdefault('run', []).append(run_id)
            conditions.setdefault('condition', []).append(cond_id)
            conditions.setdefault('condition_name', []).append(cond_name)
            conditions.setdefault('condition_file', []).append(cond_file)

        # contrast files
        contrast_files = join_subject('model', '*', '*_maps', '*.nii.gz')
        for contrast_file in sorted(glob.glob(contrast_files)):
            model_id, dtype, contrast_id = contrast_file.split(os.path.sep)[-3:]
            contrast_id = contrast_id.split('.nii.gz')[0]
            contrasts.setdefault('study', []).append(study_id)
            contrasts.setdefault('subject', []).append(subject_id)
            contrasts.setdefault('model', []).append(model_id)
            contrasts.setdefault('dtype', []).append(dtype)
            contrasts.setdefault('contrast', []).append(contrast_id)
            
    return (pd.DataFrame(structural), pd.DataFrame(functional),
            pd.DataFrame(conditions), pd.DataFrame(contrasts))


def collect_openfmri(study_dirs, n_jobs=1):
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
    datasets = Parallel(n_jobs=n_jobs, pre_dispatch='n_jobs')(
        delayed(_collect_openfmri_dataset)(study_dir) for study_dir in study_dirs)
    structural = pd.concat([d[0] for d in datasets], ignore_index=True)
    functional = pd.concat([d[1] for d in datasets], ignore_index=True)
    conditions = pd.concat([d[2] for d in datasets], ignore_index=True)
    contrasts = pd.concat([d[3] for d in datasets], ignore_index=True)
    return structural, functional, conditions, contrasts


def fetch_glm_data(functional, conditions, hrf_model='canonical',
                   drift_model='cosine', n_jobs=1):
    """Returns data (almost) ready to be used for a GLM.
    """
    main = functional.merge(conditions)

    # computing design matrices
    print 'Computing models...'
    results = Parallel(n_jobs=n_jobs, pre_dispatch='n_jobs')(
        delayed(_make_design_matrix)(run_df, hrf_model, drift_model)
        for group_id, group_df in main.groupby(['study', 'subject', 'model'])
        for run_id, run_df in group_df.groupby(['task', 'run'])
        )

    # collect results
    print 'Collecting...'
    designs = {}
    for group_id, group_df in main.groupby(['study', 'subject', 'model']):
        for run_id, run_df in group_df.groupby(['task', 'run']):
            bold_file, dm = results.pop(0)        
            designs.setdefault(group_id, {}).setdefault('bold', []).append(bold_file)
            designs.setdefault(group_id, {}).setdefault('design', []).append(dm)

    return designs

def fetch_classification_data(functional, conditions, n_jobs=1):
    """Returns data (almost) ready to be used for a
    classification of the timeseries.
    """
    glm_data = fetch_glm_data(functional, conditions, hrf_model='canonical',
                             drift_model='blank', n_jobs=n_jobs)
    classif_data = {}
    for key in glm_data:
        classif_data[key] = {}
        classif_data[key]['bold'] = glm_data[key]['bold']
        classif_data[key]['target'] = []
        for dm in glm_data[key]['design']:
            dm = dm.matrix[:, :-1]
            p = np.percentile(dm, 85)
            classif_data[key]['target'].append((dm > p).astype('int'))
    return classif_data


def _make_design_matrix(run_frame, hrf_model='canonical', drift_model='cosine'):
    print ' %s' % ' '.join(run_frame[['study', 'subject', 'model', 'task', 'run']].values[0])
    bold_file = run_frame.preproc_bold.unique()[0]
    n_scans = nb.load(bold_file).shape[-1]
    tr = float(run_frame.TR.unique()[0])
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)

    names = []
    times = []
    durations = []
    amplitudes = []
    for condition_id, condition_file in run_frame[['condition', 'condition_file']].values:
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
            names.append([condition_id] * len(keys))
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

    design_matrix = make_dmtx(
        frametimes, paradigm, hrf_model=hrf_model,
        drift_model=drift_model)

    return bold_file, design_matrix


if __name__ == '__main__':
    base_dir = '/storage/workspace/yschwart/new_brainpedia'

    # glob preproc folders
    study_dirs = sorted(glob.glob(os.path.join(base_dir, 'preproc', '*')))
    structural, functional, conditions, _ = collect_openfmri(study_dirs, n_jobs=-1)

    # glob intra_stats folders to get contrasts
    study_dirs = sorted(glob.glob(os.path.join(base_dir, 'intra_stats', '*')))
    _, _, _, contrasts = collect_openfmri(study_dirs, n_jobs=-1)

    # we can merge dataframes!
    df = functional.merge(conditions)

    # we can filter the dataframes!
    functional = functional[functional.study == 'HCP']

    # computes design matrices for the given dataframes
    designs = fetch_glm_data(functional, conditions, n_jobs=-1)

    # computes classification targets for the given dataframes
    classif = fetch_classification_data(functional, conditions, n_jobs=-1)
