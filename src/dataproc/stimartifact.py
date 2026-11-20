import pdb

import numpy as np


def zero_runs(arr):
    '''
    Detecting subsequence of array indices that are 0. Return start and end indices.
    Taken from Stackoverflow: https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array

    Parameters:
        arr: numpy array in which to find zeros.

    Return:
        ranges: numpy array with start and end indices of zero runs. Size = number of runs x 2.
            0th column is start and 1st column is end.

    Date, Author: 25 June 2024, Krishnakant V. Saboo
    '''
    #

    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def tuple_union(tuplist):
    '''
    Take union of tuples that are overlapping.
    Based on the stackoverflow answer: https://stackoverflow.com/questions/15273693/union-of-multiple-ranges

    Parameters:
        tuplist: list of tuples. Each tuple is (start, end).

    Return:
        outlist: list of tuples after taking union of tuples that are overlapping. Each tuple is (start, end).

    Note: Tuples that start right after one tuple ends are also combined. E.g., tuple (9,17) and (18,23) results in (9, 23).
    Date, Author: 25 June 2024, Krishnakant V. Saboo
    '''

    outlist = []

    for begin, end in sorted(tuplist): # go over all the tuples in the sorted list of tuples
        # if the last tuple in the list ends after current tuple begins, then they are overlapping.
        # -1 handles the case when the two tuples are right next to each other. e.g., (10,15) and (16,18)
        if outlist and outlist[-1][1] >= begin - 1:
            outlist[-1][1] = max(outlist[-1][1], end) # max of last tuple in list's end and end of current tuple

        else:
            outlist.append([begin, end])  # include current tuple in the list because it doesn't overlap
            # included as list because (begin, end) would give error => tuple doesn't allow index assignment

    return outlist



def detect_stim_artifacts(signalmat, params):
    '''
    Detect stimulation artifacts in ECoG signal. Return start and end time/indices of stimulation artifacts.
    Has functionality to remove buffer on either side of artifact signal to account for spill-over due to filtering
    of raw ECoG signal with stimulation artifacts.

    Subsequence of ECoG recordings in a channel that have the same value for >= MIN_ARTIFACT_LENGTH are detected as
    stimulations.

    Parameters:
        signalmat: numpy array of raw ECoG signal. [Size: time x channels]
        params: dictionary with parameters for stimulation artifact detection.
            Must contain keys - NUMCHAN, MIN_ARTIFACT_LENGTH, POST_STIM_BUFFER

    Returns:
        common_stimarts_buff_range: numpy array of start and end time/indices of common range of stimulation artifacts
            across channels + buffer after stimulation ends. Obtained by taking union of individual ranges in each channel.
            [Size: num of stimulations x 2] column 0 is start index, and column 1 is end index.
        common_stimarts_range: numpy array of start and end time/indices of common range of stimulation artifacts
            across channels. Obtained by taking union of individual ranges in each channel.
            [Size: num of stimulations x 2] column 0 is start index, and column 1 is end index.
        zrunsdict: dictionary with stimulation artifact indices in each channel. Doesn't include buffer after stimulation.
            Includes detections that are < MIN_ARTIFACT_LENGTH also.
            Key - 'chan-{channel number}'. E.g., 'chan-1'.

    Date, Author: 25 June 2024, Krishnakant V. Saboo
    '''

    NUMCHAN = params['NUMCHAN']
    MIN_ARTIFACT_LENGTH = params['MIN_ARTIFACT_LENGTH']
    POST_STIM_BUFFER = params['POST_STIM_BUFFER']
    len_signal = signalmat.shape[0]

    detections_list = []  # detections that are longer than a certain threshold length
    buff_detections_list = [] # detections longer than threshold + post-stim buffer included
    zrunsdict = {}  # all detections of successive samples that have the same value

    for channum in range(NUMCHAN):
        key = f'chan-{channum}'

        # take difference between successive values. Places with 0 have stim because amplitude is held constant.
        data_diff = np.diff(signalmat[:, channum], n=1)
        zruns = zero_runs(data_diff)
        # update end index of zero runs by 1 because zero-runs are detected on diffed array.
        # Diffed array has 0s at one lesser index than the original array for the duration when data is constant.
        # E.g., artifact from index=3 to index=10 is given as [3,11] in zruns for compatibility with numpy indexing.
        zruns[:,1] += 1
        zrunsdict[key] = zruns.copy()  # save all the detections also

        # find detections >= minimum artifact length
        zrundet = zruns[(zruns[:,1] - zruns[:,0]) >= MIN_ARTIFACT_LENGTH]

        # convert the detections to a list of tuples
        chan_detect_tuples = [tuple(zrundet[ii,:]) for ii in range(zrundet.shape[0])]
        detections_list.extend(chan_detect_tuples)

        # convert the detections + post-stim buffer to a list of tuples
        chan_buff_detect_tuples = [tuple([zrundet[ii,0], min(zrundet[ii,1] + POST_STIM_BUFFER, len_signal)])
                                   for ii in range(zrundet.shape[0])]
        buff_detections_list.extend(chan_buff_detect_tuples)

    # combine the artifact intervals across all the channels
    common_stimarts_range = np.array(tuple_union(detections_list))
    common_stimarts_buff_range = np.array(tuple_union(buff_detections_list))

    return common_stimarts_buff_range, common_stimarts_range, zrunsdict


def remove_stim_artifacts_from_signal(signalmat, signalmat_filtered, stimartparams):
    '''
    Remove stimulation artifacts from ECoG signal. If there is no stimulation detected in the ECoG signal,
    then NO signal is removed.

    Parameters:
        signalmat: Raw ECoG signal numpy array. [Size: time x channels]
        signalmat_filtered: Filtered ECoG signal numpy array. Must be at least 2D with time as 1st axis [Size: time x ...]
        stimartparams: dictionary with parameters for removing stimulation artifacts.
            Must contain keys - SIDE_CLIP_LEN, NUMCHAN, MIN_ARTIFACT_LENGTH, POST_STIM_BUFFER

    Returns:
        signalmat_filtered_clipped: ECoG signal with stimulation artifacts removed. Numpy array [Size: time (clipped) x bands x channels]
            Clipped ECoG signal has discontinuities due to some timepoints being removed.
            E.g., if signal has timepoint [0,...,99] and stimulation artifacts is between [16,...,30], then the clipped
            signal will correspond to timepints [0,...,15,31,...,99] and will be of length = 100 - 30 = 70.

    Date, Author: last week of June 2024, Krishnakant V. Saboo
    '''

    num_T = signalmat.shape[0]  # number of time points in a recording
    SIDE_CLIP_LEN = stimartparams.SIDE_CLIP_LEN  # remove SIDE_CLIP_LEN samples (e.g, 1s=250 samples) of data from either side
    keep_time_indices = np.arange(num_T, dtype=float) # indices of all the times to keep

    # times that contain stimulation artifacts
    stimart_times_list, ignore, ignore2 = detect_stim_artifacts(signalmat=signalmat, params=stimartparams)

    # time points to remove in filtered signal
    if len(stimart_times_list) > 0: # only do for loop if stimulation artifacts were detected in the data
        for ii in range(stimart_times_list.shape[0]):
            # clip 1s before stim artifact begins or till start of signal, whichever is larger
            t_clip_start = max(0, stimart_times_list[ii][0] - SIDE_CLIP_LEN)
            # clip 1s after stim artifact ends or till max len of signal, whichever is smaller
            t_clip_end = min(stimart_times_list[ii][1] + SIDE_CLIP_LEN, num_T)

            keep_time_indices[t_clip_start:t_clip_end] = np.NaN

    # drop the indices that are NaN
    keep_time_indices = keep_time_indices[~np.isnan(keep_time_indices)]
    keep_time_indices = keep_time_indices.astype(int)  # convert to int to allow indexing

    signalmat_filtered_clipped = signalmat_filtered[keep_time_indices].copy() # only keep times that are not artifacty or beside artifacts
    return signalmat_filtered_clipped, stimart_times_list


def mask_stim_artifacts_in_signal(signalmat, stimartparams, maskval=0):
    '''
    Remove stimulation artifacts from ECoG signal. If there is no stimulation detected in the ECoG signal,
    then NO signal is removed.

    Parameters:
        signalmat: Raw ECoG signal numpy array. [Size: time x channels]
        signalmat_filtered: Filtered ECoG signal numpy array. Must be at least 2D with time as 1st axis [Size: time x ...]
        stimartparams: dictionary with parameters for removing stimulation artifacts.
            Must contain keys - SIDE_CLIP_LEN, NUMCHAN, MIN_ARTIFACT_LENGTH, POST_STIM_BUFFER

    Returns:
        signalmat_filtered_clipped: ECoG signal with stimulation artifacts removed. Numpy array [Size: time (clipped) x bands x channels]
            Clipped ECoG signal has discontinuities due to some timepoints being removed.
            E.g., if signal has timepoint [0,...,99] and stimulation artifacts is between [16,...,30], then the clipped
            signal will correspond to timepints [0,...,15,31,...,99] and will be of length = 100 - 30 = 70.

    Date, Author: last week of June 2024, Krishnakant V. Saboo
    '''

    num_T = signalmat.shape[0]  # number of time points in a recording
    SIDE_CLIP_LEN = stimartparams.SIDE_CLIP_LEN  # remove SIDE_CLIP_LEN samples (e.g, 1s=250 samples) of data from either side
    mask = np.array(num_T*[True]) # mask of all the times to keep; True=keep

    # times that contain stimulation artifacts
    stimart_times_list, ignore, ignore2 = detect_stim_artifacts(signalmat=signalmat, params=stimartparams)

    # time points to remove in filtered signal
    if len(stimart_times_list) > 0: # only do for loop if stimulation artifacts were detected in the data
        for ii in range(stimart_times_list.shape[0]):
            # clip 1s before stim artifact begins or till start of signal, whichever is larger
            t_clip_start = max(0, stimart_times_list[ii][0] - SIDE_CLIP_LEN)
            # clip 1s after stim artifact ends or till max len of signal, whichever is smaller
            t_clip_end = min(stimart_times_list[ii][1] + SIDE_CLIP_LEN, num_T)

            mask[t_clip_start:t_clip_end] = False

    # replace signal at masked timepoints with maskval
    signalmat_masked = signalmat.copy()
    signalmat_masked[~mask] = maskval # replace mask=False entries with 0

    return signalmat_masked, mask, stimart_times_list