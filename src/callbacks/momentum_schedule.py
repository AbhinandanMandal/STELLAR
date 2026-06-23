import lightning as L


class MomentumScheduleCallback(L.Callback):
    """Keep the EMA teacher-momentum schedule correct, including across resumes.

    ``STELLARModel`` anneals the teacher momentum using an internal
    ``_step_counter`` buffer that is intentionally *not* saved in the checkpoint
    (so it never interferes with strict loading of released inference weights).
    This callback re-syncs that counter from the trainer's ``global_step`` --
    which Lightning *does* restore on resume -- so the cosine momentum schedule
    continues from exactly where it left off.

    When ``teacher_momentum_schedule_steps`` is not set on the model, it is
    filled in automatically with the total number of optimizer steps for the run
    (correctly accounting for epochs, devices and gradient accumulation).

    The callback is a no-op unless the model has ``momentum_teacher=True``, so it
    is safe to leave enabled in every config.
    """

    def __init__(self, auto_schedule_steps: bool = True):
        super().__init__()
        self.auto_schedule_steps = auto_schedule_steps

    @staticmethod
    def _stellar(pl_module):
        # Olympus wraps the model as ``pl_module.model``; fall back to the module itself.
        return getattr(pl_module, "model", pl_module)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        model = self._stellar(pl_module)
        if not getattr(model, "momentum_teacher", False):
            return

        # Auto-fill the schedule length with the total number of optimizer steps.
        if self.auto_schedule_steps and not model.teacher_momentum_schedule_steps:
            try:
                model.teacher_momentum_schedule_steps = int(trainer.estimated_stepping_batches)
            except Exception:
                pass

        # Resume: align the internal counter with the restored global step.
        if hasattr(model, "_step_counter"):
            model._step_counter.fill_(int(trainer.global_step))
        print(f"[MomentumScheduleCallback] schedule_steps="
              f"{model.teacher_momentum_schedule_steps} "
              f"start_step={int(trainer.global_step)}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        model = self._stellar(pl_module)
        if getattr(model, "momentum_teacher", False) and hasattr(model, "_step_counter"):
            model._step_counter.fill_(int(trainer.global_step))
