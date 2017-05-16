import itertools
import numpy

from six.moves import range


FS = 44100            # samples/second
DEFAULT_WINDOW_SIZE = 2048    # fourier window size
OUTPUT_SIZE = 128               # number of distinct notes
STRIDE = 512          # samples between windows
WPS = FS / float(512)   # windows/second
features = 0
labels = 1


def note_to_class(note):
    return note - 21


def get_splits(all_inds):
    test_inds = ['2303','2382', '1819']
    valid_inds = ['2131', '2384', '1792',
                  '2514', '2567', '1876']
    train_inds = [ind for ind in all_inds 
                  if ind not in test_inds and ind not in test_inds]
    return train_inds, valid_inds, test_inds


def load_in_memory(filename):
    with open(filename, 'rb') as f:
        train_data = dict(numpy.load(f))
        all_inds = train_data.keys()
        train_inds, valid_inds, test_inds = get_splits(all_inds)

        test_data = {}
        for ind in test_inds: # test set
            if ind in train_data:
                test_data[ind] = train_data.pop(ind)
            
        valid_data = {}
        for ind in valid_inds:
            valid_data[ind] = train_data.pop(ind)

        return train_data, valid_data, test_data


def create_test_in_memory(test_data, step=128, fs=11000, window_size=4096,
                          output_size=84):
    n_files = len(test_data)
    pos_per_file = 7500
    Xtest = numpy.empty([n_files * pos_per_file, window_size])
    Ytest = numpy.zeros([n_files * pos_per_file, output_size])

    for i, ind in enumerate(test_data):
        print(ind)
        audio = test_data[ind][features]

        for j in range(pos_per_file):
            if j % 1000 == 0:
                print(j)
            index = fs + j * step # start from one second to give us some wiggle room for larger segments
            Xtest[pos_per_file * i + j] = audio[index: index + window_size]
            
            # label stuff that's on in the center of the window
            s = int((index + window_size / 2))
            for label in test_data[ind][labels][s]:
                note = label.data[1]
                Ytest[pos_per_file * i + j, note_to_class(note)] = 1
    return Xtest, Ytest


def create_test(files, music_file, window_size=DEFAULT_WINDOW_SIZE, 
                output_dim=OUTPUT_SIZE, fs=FS, resample=None, step=512, note_to_class=None):
    # create the test set
    n_files = len(files)
    max_length = 7500
    Xtest = numpy.empty([n_files * max_length, window_size])
    Ytest = numpy.zeros([n_files * max_length, output_dim])

    for i, ind in enumerate(files):
        print('.. aggregating', ind)
        Ycur = IntervalTree(
            [Interval(start_time, end_time, note_id) 
             for _, end_time, _, _, note_id, _, start_time 
             in music_file[ind]['labels'][:]])
        dt = music_file[ind]['data'][:]
        if resample:
            dt = resampy.resample(dt, fs, resample)
        for j in range(7500):
            if j % 1000 == 0:
                print('.. segment', j)
            # start from one second to give us some wiggle room for larger
            # segments
            s = fs + j * step
            Xtest[7500 * i + j] = dt[s: s + window_size]
            
            # label stuff that's on in the center of the window
            for label in Ycur[s + window_size / 2]:
                Ytest[7500 * i + j, note_to_class(label.data)] = 1
    return Xtest, Ytest


def music_net_iterator(train_data, rng, window_size=4096, output_size=84,
                       complex_=False):
    channels = 2 if complex_ else 1
    Xmb = numpy.zeros([len(train_data), window_size, 2])

    while True:
        Ymb = numpy.zeros([len(train_data), output_size])
        for j, ind in enumerate(train_data):
            s = rng.randint(window_size / 2, 
                            len(train_data[ind][features]) - window_size / 2)
            Xmb[j, :, 0] = train_data[ind][features][s - window_size / 2:
                                                     s + window_size / 2]
            for label in train_data[ind][labels][s]:
                note = label.data[1]
                Ymb[j, note_to_class(note)] = 1
        yield Xmb, Ymb

