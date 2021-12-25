"""Implement some callbacks frequently used"""
import os
import time
import numpy as np
from mindspore import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.parallel._cell_wrapper import destroy_allgather_cell
from mindspore import communication as dist
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindspore.train.serialization import save_checkpoint
from ssvos.utils.log_utils import master_only_info
from mindspore import log as logger


class MyModelCheckpoint(Callback):
    def __init__(self,
                 directory,
                 append_dict: dict={},
                 interval=1,
                 prefix='ckpt',
                 master_only=True,
                 by_epoch=True,
                 get_network_fn=None,
                 rank=0
                 ):
        super().__init__()
        assert master_only, "Make sure that you only write at master device"
        self.interval = interval
        self.prefix = prefix
        self.master_only = master_only
        self.by_epoch = by_epoch
        self.get_network_fn = get_network_fn
        self.rank = rank

        self._append_dict = append_dict
        self._last_epoch = 0

        # write operation is not safe when in distribute training,
        # so we need to make independent dirs per device
        self.directory = directory
        self.rank = rank
        if rank == 0:
            os.makedirs(self.directory, exist_ok=True)

    def _save_ckpt(self, cb_params, force=False):
        cur_epoch = cb_params.cur_epoch_num
        global_step = cb_params.cur_step_num
        if cur_epoch == self._last_epoch + self.interval or force:
            master_only_info("[INFO] Start saving checkpoint...", rank=self.rank)
            ckpt_path = os.path.join(
                self.directory, f'{self.prefix}_epoch_{cur_epoch}.ckpt')
            if self.get_network_fn is not None:
                network = self.get_network_fn(cb_params.train_network)
            else:
                network = cb_params.train_network
            # save epoch_num
            self._append_dict.update(epoch=cur_epoch)
            self._append_dict.update(global_step=global_step)
            save_checkpoint(network, ckpt_path, append_dict=self._append_dict)
            master_only_info(f'[INFO] Checkpoint saved at {ckpt_path}.', rank=self.rank)
            # update last epoch
            self._last_epoch = cur_epoch

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if self.rank == 0:
            self._save_ckpt(cb_params)

    def end(self, run_context):
        cb_params = run_context.original_args()
        if self.rank == 0:
            self._save_ckpt(cb_params, force=True)
        destroy_allgather_cell()


class MindSightLoggerCallback(Callback):
    def __init__(self,
                 directory,
                 log_interval=1,
                 master_only=True,
                 rank = 0
                 ):
        super().__init__()
        assert master_only, "Make sure that you only write at master device"
        self.log_interval = log_interval
        self.master_only = master_only
        self.rank = rank

        # write operation is not safe when in distribute training,
        # so we need to make independent dirs per device
        self.directory = directory
        if rank == 0:
            os.makedirs(self.directory, exist_ok=True)

        self._temp_optimizer = None
        self._is_parse_loss_success = True
        self._last_step = 0

    def __enter__(self):
        # init your summary record in here, when the train script run, it will be inited before training
        if self.rank == 0:
            self.summary_record = SummaryRecord(self.directory)
        return self

    def __exit__(self, *exc_args):
        # Note: you must close the summary record, it will release the process pool resource
        # else your training script will not exit from training.
        if self.rank == 0:
            self.summary_record.close()

    def _get_optimizer(self, cb_params):
        """
        Get optimizer from the cb_params or parse from the network.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.

        Returns:
            Union[Optimizer, None], if parse optimizer success, will return a optimizer, else return None.
        """
        # 'optimizer_failed' means find optimizer failed, so we will not collect data about optimizer.
        optimizer_failed = 'Failed'
        if self._temp_optimizer == optimizer_failed:
            return None

        if self._temp_optimizer is not None:
            return self._temp_optimizer

        optimizer = cb_params.optimizer
        if optimizer is None:
            network = cb_params.train_network if cb_params.mode == 'train' else cb_params.eval_network
            optimizer = self._parse_optimizer_by_network(network)

        if optimizer is None or not isinstance(optimizer, Optimizer):
            logger.warning("Can not find optimizer in network, or the optimizer does not inherit MindSpore's "
                           "optimizer, so we will not collect data about optimizer in SummaryCollector.")
            optimizer = None

        self._temp_optimizer = optimizer if optimizer is not None else optimizer_failed

        return optimizer

    @staticmethod
    def _parse_optimizer_by_network(network):
        """Parse optimizer from network, if parse success will return a optimizer, else return None."""
        optimizer = None
        for _, cell in network.cells_and_names():
            if isinstance(cell, Optimizer):
                return cell
            try:
                optimizer = getattr(cell, 'optimizer')
            except AttributeError:
                continue

            if not isinstance(optimizer, Optimizer):
                continue

            # Optimizer found successfully
            break

        return optimizer

    def _get_loss(self, cb_params):
        """
        Get loss from the network output.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.

        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        if not self._is_parse_loss_success:
            # If parsing has failed before, avoid repeating it
            return None

        output = cb_params.net_outputs
        if output is None:
            logger.warning(
                "Can not find any output by this network, so SummaryCollector will not collect loss.")
            self._is_parse_loss_success = False
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            logger.warning(
                "The output type could not be identified, so no loss was recorded in SummaryCollector.")
            self._is_parse_loss_success = False
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = Tensor(np.mean(loss.asnumpy()))
        return loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        global_step = cb_params.cur_step_num
        if global_step == self._last_step + self.log_interval:
            if self.rank == 0:
                cur_lr = self._get_optimizer(cb_params).get_lr()
                loss = self._get_loss(cb_params)
                self.summary_record.add_value(
                    'scalar', 'global_step', Tensor(global_step))
                self.summary_record.add_value('scalar', 'loss', Tensor(loss))
                self.summary_record.add_value('scalar', 'lr', Tensor(cur_lr))
                self.summary_record.record(global_step)
            self._last_step = global_step


class ConsoleLoggerCallBack(Callback):
    def __init__(self,
                 log_interval=1,
                 master_only=True,
                 rank=0):
        super().__init__()
        assert master_only, "Make sure that you only write at master device"
        self.log_interval = log_interval
        self.rank = rank

        self._temp_optimizer = None
        self._is_parse_loss_success = True
        self._last_step = 0

    def _get_optimizer(self, cb_params):
        """
        Get optimizer from the cb_params or parse from the network.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.

        Returns:
            Union[Optimizer, None], if parse optimizer success, will return a optimizer, else return None.
        """
        # 'optimizer_failed' means find optimizer failed, so we will not collect data about optimizer.
        optimizer_failed = 'Failed'
        if self._temp_optimizer == optimizer_failed:
            return None

        if self._temp_optimizer is not None:
            return self._temp_optimizer

        optimizer = cb_params.optimizer
        if optimizer is None:
            network = cb_params.train_network if cb_params.mode == 'train' else cb_params.eval_network
            optimizer = self._parse_optimizer_by_network(network)

        if optimizer is None or not isinstance(optimizer, Optimizer):
            logger.warning("Can not find optimizer in network, or the optimizer does not inherit MindSpore's "
                           "optimizer, so we will not collect data about optimizer in SummaryCollector.")
            optimizer = None

        self._temp_optimizer = optimizer if optimizer is not None else optimizer_failed

        return optimizer

    @staticmethod
    def _parse_optimizer_by_network(network):
        """Parse optimizer from network, if parse success will return a optimizer, else return None."""
        optimizer = None
        for _, cell in network.cells_and_names():
            if isinstance(cell, Optimizer):
                return cell
            try:
                optimizer = getattr(cell, 'optimizer')
            except AttributeError:
                continue

            if not isinstance(optimizer, Optimizer):
                continue

            # Optimizer found successfully
            break

        return optimizer

    def _get_loss(self, cb_params):
        """
        Get loss from the network output.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.

        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        if not self._is_parse_loss_success:
            # If parsing has failed before, avoid repeating it
            return None

        output = cb_params.net_outputs
        if output is None:
            logger.warning(
                "Can not find any output by this network, so SummaryCollector will not collect loss.")
            self._is_parse_loss_success = False
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            logger.warning(
                "The output type could not be identified, so no loss was recorded in SummaryCollector.")
            self._is_parse_loss_success = False
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = Tensor(np.mean(loss.asnumpy()))
        return loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        batch_num = cb_params.batch_num
        epoch = cb_params.cur_epoch_num
        global_step = cb_params.cur_step_num
        step = (global_step - 1) % batch_num + 1
        # cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        if global_step == self._last_step + self.log_interval:
            cur_lr = self._get_optimizer(cb_params).get_lr()
            loss = self._get_loss(cb_params)
            if isinstance(loss, (tuple, list)):
                if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                    loss = loss[0]

            if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
                loss = np.mean(loss.asnumpy())

            log_str = f' Epoch [{epoch}][{step}/{batch_num}] lr: {cur_lr:}, loss: {loss}'
            master_only_info(log_str, rank=self.rank)
            self._last_step = global_step

        