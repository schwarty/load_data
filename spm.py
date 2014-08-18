import os
import gzip
import shutil
import tempfile

import numpy as np
import pandas as pd

from scipy.io import loadmat


def load_matfile(mat_file):
    if isinstance(mat_file, (str, unicode)):
        if mat_file.endswith('.gz'):
            mat_file = gzip.open(mat_file, 'rb')
        return loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    return mat_file


def load_conditions_onsets(mat_file, session_names=None, condition_names=None):
    matfile = load_matfile(mat_file)['SPM']
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


def get_task_contrasts(mat_file, has_derivative=True):
    matfile = load_matfile(mat_file)['SPM']
    mat_dir = os.path.abspath(os.path.split(mat_file)[0])

    stats = {}

    if hasattr(matfile.Sess, '__iter__'):
        sessions = matfile.Sess
    else:
        sessions = [matfile.Sess]

    conditions = []
    for session_id, session in enumerate(sessions):
        for i, u in enumerate(session.U):
            condition_id = 'cond%03i_%s' % (i + 1, str(u.name))
            conditions.append(condition_id)
            if has_derivative:
                conditions.append('%s_derivative' % condition_id)

    for c in matfile.xCon:
        image_id = str(c.name)

        try:
            c_map = os.path.join(mat_dir, str(c.Vcon.fname))
            t_map = os.path.join(mat_dir, str(c.Vspm.fname))
        except:
            c_map = ''
            t_map = ''

        stats.setdefault('image_id', []).append(image_id)
        stats.setdefault('c_maps', []).append(c_map)
        stats.setdefault('t_maps', []).append(t_map)

        conditions_values = c.c.tolist()[:len(conditions)]
        for condition_id, v in zip(conditions, conditions_values):
            stats.setdefault(condition_id, []).append(v)

    return pd.DataFrame(stats)
