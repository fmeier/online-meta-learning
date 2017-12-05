
import glob
import os
import abc
import threading

try:
    import cPickle as pkl
except:
    import pickle as pkl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_utils import device
from tf_utils import defines


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    with device.device_cpu():
        name = var.op.name.replace(':', '_')
        with tf.name_scope('summaries/{}'.format(name)):
            mean = tf.reduce_mean(var)
            summaries = []
            summaries.append(tf.summary.scalar('mean/{}'.format(name), mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            summaries.append(tf.summary.scalar(
                'stddev/{}'.format(name), stddev))
            summaries.append(tf.summary.scalar('max/{}'.format(name),
                                               tf.reduce_max(var)))
            summaries.append(tf.summary.scalar('min/{}'.format(name),
                                               tf.reduce_min(var)))

            # This results sometimes in NaNs.
            summaries.append(tf.summary.histogram(name, var))
            return summaries


class SaveTensorDict(object):

    def __init__(self, dir_path, name, only_if_cpu):
        self._dir_path = os.path.join(dir_path, defines.SUMMARIES)

        # We use this for the summary and : is not allowed.
        self._name = name.replace(':', '_')
        # We use this for the file path, therefore the filename has to
        # not contain slashes.
        self._name = self._name.replace('/', '__')
        self._only_if_cpu = only_if_cpu

    def create_save_summary(self, input_tensors):
        """Create an summary op to save tensors from a dict of input tensors.

        Args:
            input_tensors: A list/dict or single tensor which can be
                used for plotting.

        Returns:
            A summary op which is already registered in the graph.

        """
        if not isinstance(input_tensors, dict):
            raise Exception('Input_tensors has to be a dict')

        if self._only_if_cpu:
            if device.maybe_is_tensor_gpu(input_tensors.values()):
                return None

        if not os.path.exists(self._dir_path):
            os.makedirs(self._dir_path)

        self._file_path = os.path.join(self._dir_path, self._name)

        input_tensors['step'] = tf.train.get_global_step()
        # We sort the keys such that we have a stable map
        keys = sorted(input_tensors.keys())
        inp = []
        for key in keys:
            inp.append(input_tensors[key])

        def _custom_summary(*input_tensors):
            data = {}
            for key, value in zip(keys, input_tensors):
                data[key] = value
            step = data['step']
            with open(self._file_path + '_' + str(step) + '.pkl', 'w') as fo:
                pkl.dump(data, fo)
            return step

        with device.device_cpu():
            summary_op = tf.py_func(_custom_summary,
                                    inp=inp,
                                    Tout=tf.int64)
            return tf.summary.scalar(
                'SaveTensorDict/' + self._name, summary_op)


class PlotImageSummary(object):
    """Interface in order to use matplotlib to update a summary.
    """
    __metaclass__ = abc.ABCMeta

    _figure_count = 0
    _lock = threading.RLock()

    def __init__(self, name, only_if_cpu):
        self._fig = None
        self._name = name
        self._only_if_cpu = only_if_cpu

    @abc.abstractmethod
    def init_plot(self, input_tensors):
        """Init the plot such that we can update the data in order to be fast.

        Args:
            input_tensors: A list/dict or single tensor which can be
                used for plotting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_plot(self, input_tensors):
        """Update teh data of the plot.

        Args:
            input_tensors: A list of tensors which can be used for plotting.
        """

    def get_plot_buffer(self):
        """Redraw the canvas and return the image as a string for numpy.
        """
        self._fig.canvas.draw()
        return self._fig.canvas.tostring_rgb()

    def create_image_summary(self, input_tensors):
        """Create an image summary from a list of input tensors.

        Args:
            input_tensors: A list/dict or single tensor which can be
                used for plotting.

        Returns:
            A summary op which is already registered in the graph.

        """

        if self._only_if_cpu:
            if device.maybe_is_tensor_gpu(input_tensors):
                return None

        # We need to register a new figure.
        figure_count = PlotImageSummary._figure_count
        # We typically do not want to have the interactive backend.
        plt.switch_backend('Agg')

        self._fig = plt.figure(figure_count)
        ncols, nrows = self._fig.canvas.get_width_height()
        shape = (1, nrows, ncols, 3)

        self._figure_count += 1
        self.init_plot(input_tensors)

        def _custom_summary(*args):

            # Matplotlib is inherently single threaded and ops can be
            # executed in parallel so we need to lock the figure update.
            with PlotImageSummary._lock:
                plt.figure(figure_count)
                self.update_plot(args)
                buf = self.get_plot_buffer()
            return np.fromstring(buf, dtype=np.uint8).reshape(shape)

        if not isinstance(input_tensors, list):
            input_tensors = [input_tensors]

        with device.device_cpu():
            summary_op = tf.py_func(_custom_summary,
                                    inp=input_tensors,
                                    Tout=tf.uint8)
            return tf.summary.image(self._name, summary_op)


class PlotByTime(PlotImageSummary):

    def __init__(self, name, only_if_cpu=False):
        super(PlotByTime, self).__init__(name, only_if_cpu)

    def init_plot(self, input_tensors):
        """We initial a plot for a single tensor.
        """
        assert isinstance(input_tensors, tf.Tensor), (
            'We only support a single tensor but received {}'.format(
                input_tensors))
        assert input_tensors.get_shape().ndims == 2, (
            'We only support rank 2 tensors but got {}.'.format(
                input_tensors.ndim))

        self._axis = self._fig.add_subplot(1, 1, 1)
        self._lines = []

        # We register all lines in order to simply update the data later.
        for line in xrange(input_tensors.get_shape().as_list()[0]):
            self._lines.append(
                mpl.lines.Line2D([], [],
                                 label=str(line)))
            self._axis.add_line(self._lines[-1])
        self._y = []
        plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)

    def update_plot(self, input_tensors):
        data = input_tensors[0]
        self._y.append(data)
        x = np.arange(len(self._y))
        y = np.array(self._y).reshape(len(self._y), -1)
        for line in xrange(self._size):
            self._lines[line].set_data(x, y[:, line])
        self._axis.set_ylim((y.min(), y.max()))
        self._axis.set_xlim((0, len(self._y)))


class PlotBins(PlotImageSummary):

    def __init__(self, name, min_val, max_val, only_if_cpu=False):
        super(PlotBins, self).__init__(name, only_if_cpu)
        self._min_val = min_val
        self._max_val = max_val

    def init_plot(self, input_tensors):
        assert isinstance(input_tensors, tf.Tensor), (
            'We only support a single tensor but received {}'.format(
                input_tensors))

        assert input_tensors.get_shape().ndims <= 1, (
            'We only support rank 2 tensors but got {}.'.format(
                input_tensors.ndim))

        self._x = np.linspace(self._min_val, self._max_val,
                              input_tensors.get_shape().as_list()[0])
        self._axis = self._fig.add_subplot(1, 1, 1)
        self._line = mpl.lines.Line2D([], [], color='black')
        self._axis.add_line(self._line)
        self._axis.set_xlim((self._min_val, self._max_val))

    def update_plot(self, input_tensors):
        data = input_tensors[0].ravel()
        self._line.set_data(self._x, data)
        self._axis.set_ylim((0, data.max()))


class PlotScatterXY(PlotImageSummary):

    def __init__(self, name, only_if_cpu=False):
        super(PlotScatterXY, self).__init__(name, only_if_cpu)

    def init_plot(self, input_tensors, colors=None):
        """We initial a plot for a single tensor.
        """
        assert isinstance(input_tensors, list), (
            'We only support a single tensor but received {}'.format(
                input_tensors))

        self._axis = self._fig.add_subplot(1, 1, 1)
        self._scat = []

        # We register all lines in order to simply update the data later.

        x = np.zeros(np.prod(input_tensors[0].get_shape().as_list()))
        for pos in xrange(len(input_tensors) - 1):
            self._scat.append(
                self._axis.scatter(x, x, label=str(pos)))
        plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)

    def update_plot(self, input_tensors):
        min_y = np.inf
        max_y = -np.inf
        for pos, scat in enumerate(self._scat):
            cur_input_tensor = input_tensors[pos + 1]
            min_y = min(min_y, cur_input_tensor.min())
            max_y = max(max_y, cur_input_tensor.max())
            scat._offsets[:, 0] = input_tensors[0].ravel()
            scat._offsets[:, 1] = input_tensors[pos + 1].ravel()

        self._axis.set_xlim((input_tensors[0].min(),
                             input_tensors[0].max()))

        self._axis.set_ylim((min_y, max_y))


def get_file_path_pattern(dir_path_experiment, pattern):
    root_path = os.path.abspath(dir_path_experiment)
    # Notice do not start with a / when joining paths, it will do a root join.
    glob_path = os.path.join(root_path, pattern)
    return glob.glob(glob_path)


def get_summary_events(dir_path_experiment, key_name=None):
    root_path = os.path.abspath(dir_path_experiment)
    files = get_file_path_pattern(dir_path_experiment,
                                  pattern='*/*events.out.tfevents.*')
    result = {}
    for file_path in files:
        rel_path = os.path.relpath(file_path, root_path)
        try:
            result[rel_path] = read_from_summary_file(file_path, key_name)
        except:
            pass
    return result


def read_from_summary_file(file_path, key_name=None):
    data = []
    for e in tf.train.summary_iterator(file_path):
        if key_name is None:
            element = {}
        for v in e.summary.value:
            if key_name is not None:
                if key_name in v.tag:
                    data.append(v.simple_value)
            else:
                print(v)
                element[v.tag] = v.simple_value
        if key_name is None:
            if element:
                data.append(element)

    return data


def get_summary_save_tensor_dict(dir_path_experiment, summary_name):
    dir_path_experiment = os.path.abspath(dir_path_experiment)
    files = get_file_path_pattern(dir_path_experiment,
                                  pattern='*/summaries/*{}*.pkl'.format(
                                      summary_name))

    files += get_file_path_pattern(dir_path_experiment,
                                   pattern='summaries/*{}*.pkl'.format(
                                       summary_name))

    result = {}
    for file_path in files:
        rel_path = os.path.relpath(file_path, dir_path_experiment)
        experiment_name = os.path.dirname(rel_path)

        # We add the new step to our dict.
        # The step is always available since the save_tensor_dict will
        # add this automatically.
        data = result.get(experiment_name, {})
        with open(file_path, 'r') as fi:
            save_tensor_dict = pkl.load(fi)
        data[save_tensor_dict['step']] = save_tensor_dict
        result[experiment_name] = data
    for key, value in result.items():

        # The keys are always the same for one save_tensor_dict.
        data_keys = value.values()[0].keys()
        result_np = {}

        # The keys are the steps so we sort them.
        indices = sorted(value.keys())

        for data_key in data_keys:
            result_np[data_key] = np.stack(
                [value[index][data_key] for index in indices])
        result[key] = result_np
    return result
