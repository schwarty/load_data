import os
import re
import gzip
import shutil
import tempfile

import numpy as np
import pandas as pd
import nibabel as nb

from scipy.io import loadmat
from joblib import Memory, Parallel, delayed


default_memory = Memory(None)


def load_matfile(mat_file):
    """Load a matfile or gzipped matfile with the right arguments
    """
    if isinstance(mat_file, (str, unicode)):
        if mat_file.endswith('.gz'):
            mat_file = gzip.open(mat_file, 'rb')
        return loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    return mat_file


def load_conditions_onsets(mat_file, session_names=None,
                           condition_names=None, memory=default_memory):
    """Load conditions onsets

       Parameters
       ----------
       mat_file: str
           matfile location
       session_names: dict or None (optional)
           alternative session names, with sessions id as integers
           for the keys, and sessions names as str for the values.
           e.g., {1: 'session_one', 2: 'session_two'}
       condition_names: dict or None (optional)
           alternative condition names, with tuple of integers
           corresponding to the session_id and condition_id as keys.
           e.g., {(1, 1): 'session001_cond001_task',
                  (1, 2): 'session001_cond002_control'}
       memory: Memory (optional)
           joblib memory

       Returns
       -------
       conditions: DataFrame
           one line per event and the following columns:
           session_id, session_name, condition_id, condition_name,
           condition_default_name, onset, duration, amplitude
    """
    matfile = memory.cache(load_matfile)(mat_file)['SPM']
    session_names = dict() if session_names is None else session_names
    condition_names = dict() if condition_names is None else condition_names
    conditions = {}

    if hasattr(matfile.Sess, '__iter__'):
        sessions = matfile.Sess
    else:
        sessions = [matfile.Sess]

    for session_id, session in enumerate(sessions):
        for condition_id, u in enumerate(session.U):
            default_condition_name = str(u.name)
            onsets = u.ons.tolist()
            durations = u.dur.tolist()
            if not isinstance(onsets, list):
                onsets = [onsets]
                durations = [durations]
            n_events = len(onsets)
            amplitudes = [1] * n_events
            session_name = session_names.get(
                session_id + 1,
                'session%03i' % (session_id + 1))
            condition_name = condition_names.get(
                (session_id + 1, condition_id + 1),
                default_condition_name)
            for i in range(n_events):
                conditions.setdefault('session_id', []).append(session_id + 1)
                conditions.setdefault('session_name', []).append(session_name)
                conditions.setdefault('condition_id', []).append(
                    condition_id + 1)
                conditions.setdefault('condition_name', []).append(
                    condition_name)
                conditions.setdefault('condition_default_name', []).append(
                    default_condition_name)
                conditions.setdefault('onset', []).append(onsets[i])
                conditions.setdefault('duration', []).append(durations[i])
                conditions.setdefault('amplitude', []).append(amplitudes[i])

    return pd.DataFrame(conditions)


def save_conditions_onsets(conditions_onsets,
                           onsets_dir=tempfile.gettempdir()):
    """Save a DataFrame given by get_conditions_onsets to disk.
    """
    condition_files = []

    for cid, cframe in conditions_onsets.groupby(['session_name',
                                                  'condition_id']):
        session_id, condition_id = cid
        onset_dir = os.path.join(onsets_dir, session_id)
        if not os.path.exists(onset_dir):
            os.makedirs(onset_dir)
        fname = os.path.join(onset_dir, 'cond%03i.txt' % condition_id)
        cframe[['onset', 'duration', 'amplitude']].to_csv(
            fname, sep=' ', header=False, index=False)
        condition_files.append(fname)

    return condition_files


def get_task_contrasts(mat_file, contrast_names=None,
                       design_format='nipy', memory=default_memory):
    matfile = memory.cache(load_matfile)(mat_file)['SPM']
    mat_dir = os.path.abspath(os.path.split(mat_file)[0])
    n_scans, n_sessions = _get_n_scans(mat_file)

    stat_images = {}
    contrasts = {}
    condition_names = matfile.xX.name.astype(np.str)

    for c in matfile.xCon:
        contrast_id = str(c.name)
        if contrast_names is not None and contrast_id not in contrast_names:
            continue
        elif contrast_names is not None:
            contrast_id = contrast_names[contrast_id]

        try:
            c_map = os.path.join(mat_dir, str(c.Vcon.fname))
            t_map = os.path.join(mat_dir, str(c.Vspm.fname))
        except:
            c_map = ''
            t_map = ''

        stat_images.setdefault('contrast_id', []).append(contrast_id)
        stat_images.setdefault('c_maps', []).append(c_map)
        stat_images.setdefault('t_maps', []).append(t_map)
        condition_values = c.c

        if design_format == 'nipy':
            session_masking = _mask_session_design_matrix(condition_names,
                                                          n_sessions)
            for i, mask in enumerate(session_masking):
                contrasts.setdefault(i, {}).setdefault(
                    'index', []).append(contrast_id)
                for condition_id, cond_val in zip(condition_names[mask],
                                                  condition_values[mask]):
                    contrasts.setdefault(i, {}).setdefault(
                        condition_id, []).append(cond_val)
        else:
            contrasts.setdefault('index', []).append(contrast_id)
            for condition_id, cond_val in zip(condition_names,
                                              condition_values):
                contrasts.setdefault(condition_id, []).append(cond_val)

    if design_format == 'nipy':
        contrasts = [pd.DataFrame(c, index=c.pop('index'))
                     for c in contrasts.values()]
    else:
        contrasts = pd.DataFrame(contrasts, index=contrasts.pop('index'))

    return pd.DataFrame(stat_images), contrasts


def load_design_matrix(mat_file, design_format='nipy', memory=default_memory):
    """ Load design matrix from matfile.

        Parameters
        ----------
        mat_file: str
            Path of matfile
        design_format: str
            nipy: return a list of DataFrames, one per session
            spm : return a single DataFrame with all the design
    """
    matfile = memory.cache(load_matfile)(mat_file)['SPM']

    n_scans, n_sessions = _get_n_scans(mat_file)

    # condition names
    condition_names = matfile.xX.name.astype(np.str)

    # design matrix
    design_matrix = matfile.xX.X.tolist()           # xX: model

    if design_format == 'nipy':  # slicing design matrix per session
        design_matrices = []
        session_masking = _mask_session_design_matrix(
            condition_names, n_sessions)
        # split design matrix on the timepoint axis per session
        dm_vmasked = np.vsplit(design_matrix, np.cumsum(n_scans[:-1]))
        for mask, dm in zip(session_masking, dm_vmasked):
            # mask the session matrices with their regressors only
            dm_hmasked = pd.DataFrame(
                dict(zip(condition_names[mask], dm[:, mask].T)))
            design_matrices.append(dm_hmasked)
        return design_matrices

    else:  # plain design matrix
        return pd.DataFrame(dict(zip(condition_names, design_matrix)))


def get_bold_timeseries(mat_file, smoothed=False, memory=default_memory):
    matfile = memory.cache(load_matfile)(mat_file)['SPM']
    n_scans, n_sessions = _get_n_scans(mat_file)

    # bold timeseries paths
    timeseries = [_check_bold_path(p, mat_file) for p in matfile.xY.P]

    # remove the s from swa to get non-smoothed data
    if not smoothed:
        _timeseries = []
        for timepoint in timeseries:
            dirname, fname = os.path.split(timepoint)
            if fname.startswith('swa'):
                fname = fname[1:]
            _timeseries.append(os.path.join(dirname, fname))
        timeseries = _timeseries

    # split files per session
    if len(nb.load(timeseries[0]).shape) == 4:
        timeseries = np.unique(timeseries).tolist()
    else:
        timeseries = [
            t.tolist()
            for t in np.split(timeseries, np.cumsum(n_scans)[:-1])]

    return timeseries


def load_glm_inputs(mat_files, smoothed_bold=False, contrast_names=None,
                    design_format='nipy', memory=default_memory, n_jobs=1):

    glm_inputs = Parallel(n_jobs=n_jobs)(delayed(_parallel_load_glm_inputs)(
            mat_file, smoothed_bold, contrast_names, design_format, memory)
            for mat_file in mat_files)

    return dict(zip(mat_files, glm_inputs))


def _parallel_load_glm_inputs(mat_file, smoothed_bold=False,
                              contrast_names=None, design_format='nipy',
                              memory=default_memory):

    glm_inputs = {}

    design_matrix = load_design_matrix(mat_file, design_format)
    stat_images, contrasts = get_task_contrasts(mat_file, contrast_names,
                                                design_format)
    bold = get_bold_timeseries(mat_file, smoothed=smoothed_bold)

    glm_inputs['design'] = design_matrix
    glm_inputs['bold'] = bold

    if design_format == 'nipy':
        contrasts_ = {}
        for sess in contrasts:
            for c, cval in zip(sess.index, sess.values):
                contrasts_.setdefault(c, []).append(cval.tolist())
        tasks = np.unique([c.split('_', 1)[0]
                           for session in contrasts
                           for c in session.index.values
                           if c.startswith('task')])

    else:
        contrasts_ = dict(zip(contrasts.index, contrasts.values))

    # remove 2d contrasts
    contrasts = {}
    for c in contrasts_:
        if np.array(contrasts_[c]).ndim != 3:
            contrasts[c] = contrasts_[c]

    # make a model001_per_run
    if design_format == 'nipy' and tasks.size > 0:
        contrasts_per_run = {}
        for task_id in tasks:
            for c in contrasts:
                n_runs = np.array(contrasts[c]).shape[0]
                for run_id in range(n_runs):
                    contrast_id = '%s_run%03i_%s' % (
                        task_id, run_id + 1, c.split('_', 1)[1])
                    run_indices = np.ones(n_runs, dtype=np.bool)
                    run_indices[run_id] = False
                    run_indices = np.where(run_indices)[0]
                    con_val = contrasts[c][:]
                    for i in run_indices:
                        con_val[i] = None
                    contrasts_per_run[contrast_id] = con_val
        glm_inputs['model001_per_run'] = contrasts_per_run

    glm_inputs['model001'] = contrasts

    return glm_inputs


def _mask_session_design_matrix(condition_names, n_sessions):
    for i in range(n_sessions):
        yield np.array([name.startswith('Sn(%s)' % (i + 1))
                        for name in condition_names], dtype=np.bool)


def _check_bold_path(path, up_to_date_path=None):
    path = str(path).strip()

    # for data processed on Windows
    if '\\' in path:
        chunks = path.split('\\')
        if chunks[0] == '' or ':' in chunks[0]:
            chunks = chunks[1:]
        return os.path.join(*chunks)

    # SPM8 adds the volume number at the end of the path
    if re.match('.*,\d+', path):
        return path.split(',')[0]

    if up_to_date_path is not None:
        up_to_date_path = os.path.realpath(up_to_date_path)
        path_chunks = np.array(path.split('/'))
        up_to_date_path_chunks = np.array(up_to_date_path.split('/'))
        pivot = up_to_date_path_chunks[np.in1d(
            up_to_date_path_chunks, path_chunks)][-1]
        root = up_to_date_path.split(pivot)[0]
        leaf = os.path.join(
            os.path.split(path)[0].split(pivot)[1].strip('/'),
            os.path.split(path)[1])
        path = os.path.join(root, pivot, leaf)

    return path


def _get_n_scans(mat_file, memory=default_memory):
    matfile = memory.cache(load_matfile)(mat_file)['SPM']

    if not hasattr(matfile.nscan, '__iter__'):
        n_scans = [matfile.nscan]
    else:
        n_scans = matfile.nscan.tolist()
    n_sessions = len(n_scans)

    return n_scans, n_sessions
