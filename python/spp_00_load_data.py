from scipy.io import loadmat
import numpy as np
import os.path
# from utils import get_filename_with_ext

# FEAT_DIR = '../features_hjort_2hz'
FEAT_DIR = '../features_sliding_windows'
# FEAT_DIR = '../features'


def load_features(data_set, patient, list_set_features):

    print '-------------USING FEATURE DIRECTORY --------------'
    print FEAT_DIR
    print '---------------------------------------------------'

    X = np.array([])
    y = np.array([])
    afeatnames = []
    aFiles = []
    plabels = []
    p_labels_10min = []
    data_q = []

    for i, l in enumerate(list_set_features):
        mask = '{0}_{1}_{2}'.format(data_set, patient, l)
        filename = get_files_with_mask(FEAT_DIR, mask)

        if len(filename) == 0:
            raise Exception('File on path: {0} with mask: {1} does not exist'.format(FEAT_DIR, mask))

        # if loading from 10 minutes segements p_labels_t = p_labels_10min_t
        X_t, y_t, feat_names_t, a_files_t, p_labels_t, data_q_t, p_labels_10min_t = get_data_from_mat(filename[0])
        # load_from_mat(filename[0])

        if i == 0:
            X = X_t
            y = y_t
            afeatnames = feat_names_t
            aFiles = a_files_t
            plabels = p_labels_t
            data_q = data_q_t
            p_labels_10min = p_labels_10min_t
        else:
            assert (sum(y - y_t) == 0)
            assert (aFiles == a_files_t) is True
            assert (sum(plabels - p_labels_t) == 0)
            assert (sum(data_q - data_q_t) == 0)
            assert (sum(p_labels_10min - p_labels_10min_t) == 0)

            X = np.hstack((X, X_t))
            afeatnames += feat_names_t

    n = plabels.shape[0]
    b_10min_segment = True if sum(plabels == p_labels_10min) == n else False

    data = dict()
    data['X'] = X
    data['y'] = y
    data['aFeatNames'] = afeatnames
    data['aFiles'] = aFiles
    data['plabels'] = plabels
    data['plabels_10min'] = p_labels_10min
    data['data_q'] = data_q
    data['b_10min_segment'] = b_10min_segment

    return data
    # return X, y, afeatnames, aFiles, plabels, data_q, p_labels_10min


def get_data_from_mat(filename):

    d = loadmat(filename)

    if 'X_win' in d:
        return get_from_win(d)
    else:
        return get_from_10_min(d)


def get_from_win(data):
    """
    :param data loaded matlab file
    """
    # d = loadmat(filename)

    featnames = data['aFeatNames'][0]
    featnames = [f[0] for f in featnames]

    aFiles = data['aFiles_win'][:, 0]
    aFiles = [get_filename_with_ext(f[0]) for f in aFiles]

    # plabels_win - similar to plabels (data from 1 hours has the same p)
    # plabels_10min - data divided into sliding windows has the same labels
    plabels_win = data['plabels_win'][:, 0]
    plabels_10min = data['plabels_10min'][:, 0]
    data_quality = data['data_quality_win'][:, 0]

    X = data['X_win']
    y = data['y_win']

    # print Xt.shape
    # print y.shape
    # print plabels.shape

    return X, y, featnames, aFiles, plabels_win, data_quality, plabels_10min


def get_from_10_min(data):
    """
    :param data loaded matlab file
    """
    # d = loadmat(filename)

    featnames = data['aFeatNames'][0]
    featnames = [f[0] for f in featnames]

    aFiles = data['aFiles'][:, 0]
    aFiles = [get_filename_with_ext(f[0]) for f in aFiles]

    if 'plabels' in data:
        plabels = data['plabels'][:, 0]
    else:
        plabels = None

    if 'data_quality' in data:
        data_quality = data['data_quality'][:, 0]
    else:
        data_quality = None

    X = data['X']
    y = data['y']

    # print Xt.shape
    # print y.shape
    # print plabels.shape

    return X, y, featnames, aFiles, plabels, data_quality, plabels


def get_files_with_mask(dir_name, mask):
    """
    Get all files in directory with specified mask
    :param dir_name:
    :param mask:
    :return: list of files
    :rtype: list()
    """
    files = directory_listing(dir_name)
    output = list()
    for f in files:
        if mask in f:
            output.append(f)

    return output


def get_filename_with_ext(infile):
    dummy, file_name_with_ext = os.path.split(infile)
    return file_name_with_ext


def get_filename_and_ext(infile):
    file1 = get_filename_with_ext(infile)
    file_name, ext = os.path.splitext(file1)
    return file_name, ext


def directory_listing(dir_name):
    output = []
    for dirname, dummy, filenames in os.walk(dir_name):
        for filename in filenames:
            output.append(os.path.join(dirname, filename))

    return output


def write_removed_features(nsubject, feat_select, feat_to_remove):
    name = 'patient_{0}_{1}_feat_to_remove.txt'.format(nsubject, '_'.join(feat_select))
    with open(name, 'w+') as fw:
        for item in feat_to_remove:
            fw.write("%s\n" % item)


def load_removed_features(subject, list_set_features):

    feat_to_remove = list()
    for i, l in enumerate(list_set_features):
        mask = 'patient_{0}_{1}_feat_to_remove'.format(subject, l)
        filename = get_files_with_mask(FEAT_DIR, mask)

        if len(filename) == 0:
            raise Exception('File on path: {0} with mask: {1} does not exist'.format(FEAT_DIR, mask))

        with open(filename[0], 'r') as fr:
            for s in fr.readlines():
                feat_to_remove.append(s.strip())

    return feat_to_remove
