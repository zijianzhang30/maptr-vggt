from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad == None and val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))


@HOOKS.register_module()
class DistillWeightScheduleHook(Hook):
    """Apply a piecewise-constant distillation weight by training epoch."""

    def __init__(self, schedule, target='pre_lss_distiller'):
        self.schedule = sorted(
            [(int(start_epoch), float(weight))
             for start_epoch, weight in schedule],
            key=lambda item: item[0])
        self.target = target

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        weight = self.schedule[0][1]
        for start_epoch, scheduled_weight in self.schedule:
            if epoch < start_epoch:
                break
            weight = scheduled_weight

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        distiller = getattr(model, self.target, None)
        if distiller is None:
            raise AttributeError(
                f'Model has no active distiller named {self.target!r}')
        distiller.loss_weight = weight
        runner.logger.info(
            f'{self.target}.loss_weight set to {weight:.6g} for epoch {epoch}')

