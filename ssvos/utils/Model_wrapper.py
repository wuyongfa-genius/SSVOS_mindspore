"""Wrap Model with a start_epoch arg."""
import os
import math
import numpy as np
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import RunContext
from mindspore.common import Tensor
from mindspore.parallel._ps_context import _is_role_pserver


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class Model_with_start_states(Model):
    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", boost_level="O0", start_epoch=0, start_step=0, **kwargs):
        super().__init__(network, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics, eval_network=eval_network,
                         eval_indexes=eval_indexes, amp_level=amp_level, boost_level=boost_level, **kwargs)
        self.start_epoch = start_epoch
        self.start_step = start_step
    
    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None, sink_size=-1):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        is_graph = (context.get_context("mode") == context.GRAPH_MODE)
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())
            train_dataset.__total_batch__ = epoch * sink_size

        ## NOTE here has been modified
        cb_params.cur_step_num = self.start_step
        cb_params.dataset_sink_mode = True

        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        dataset_helper = None
        if hasattr(train_dataset, '_dataset_helper'):
            dataset_helper = train_dataset._dataset_helper
        ## NOTE here has been modified
        total_epochs = epoch - self.start_epoch
        ## NOTE here has been modified
        for i in range(total_epochs):
            ## NOTE here has been modified
            cb_params.cur_epoch_num = i + 1 + self.start_epoch
            list_callback.epoch_begin(run_context)
            dataset_helper, train_network = self._exec_preprocess(is_train=True,
                                                                  dataset=train_dataset,
                                                                  dataset_sink_mode=True,
                                                                  sink_size=sink_size,
                                                                  epoch_num=epoch_num,
                                                                  dataset_helper=dataset_helper)

            self._train_network = train_network
            cb_params.train_network = self._train_network

            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                cb_params.train_dataset_element = inputs
                list_callback.step_begin(run_context)
                outputs = self._train_network(*inputs)
                if is_graph:
                    cb_params.cur_step_num += dataset_helper.sink_size()
                else:
                    cb_params.cur_step_num += 1
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)
                if _is_role_pserver():
                    os._exit(0)

            dataset_helper.continue_send()
            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break
        dataset_helper.stop_send()
        dataset_helper.release()

        list_callback.end(run_context)

    def _train_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        dataset_helper, _ = self._exec_preprocess(is_train=True,
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=False,
                                                  epoch_num=epoch)
        ## NOTE here has been modified
        cb_params.cur_step_num = self.start_step
        cb_params.dataset_sink_mode = False
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        ## NOTE here has been modified
        total_epochs = epoch-self.start_epoch
        ## NOTE here has been modified
        for i in range(total_epochs):
            ## NOTE here has been modified
            cb_params.cur_epoch_num = i + 1 + self.start_epoch

            list_callback.epoch_begin(run_context)

            for next_element in dataset_helper:
                len_element = len(next_element)
                next_element = _transfer_tensor_to_tuple(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError("When 'loss_fn' is not None, 'train_dataset' should return "
                                     "two elements, but got {}, please check the number of elements "
                                     "returned by 'train_dataset'".format(len_element))
                cb_params.cur_step_num += 1

                cb_params.train_dataset_element = next_element
                list_callback.step_begin(run_context)
                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    _, overflow, _ = outputs
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                list_callback.step_end(run_context)
                if _is_role_pserver():
                    os._exit(0)
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

            train_dataset.reset()

            # if param is cache enable, flush data from cache to host before epoch end
            self._flush_from_cache(cb_params)

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.end(run_context)