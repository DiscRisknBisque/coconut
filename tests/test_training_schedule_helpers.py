from run import (
    _optimizer_update_steps,
    _remaining_epochs_in_current_stage,
    _scheduler_step_counts,
)


def test_optimizer_update_steps_uses_ceil_for_remainder_batch():
    assert _optimizer_update_steps(10, 3) == 4
    assert _optimizer_update_steps(9, 3) == 3
    assert _optimizer_update_steps(0, 3) == 0


def test_remaining_epochs_in_current_stage_handles_stage_boundary_and_resume():
    assert _remaining_epochs_in_current_stage(
        epoch=0,
        num_epochs=12,
        epochs_per_stage=5,
        single_stage_schedule=False,
    ) == 5
    assert _remaining_epochs_in_current_stage(
        epoch=4,
        num_epochs=12,
        epochs_per_stage=5,
        single_stage_schedule=False,
    ) == 1
    assert _remaining_epochs_in_current_stage(
        epoch=5,
        num_epochs=12,
        epochs_per_stage=5,
        single_stage_schedule=False,
    ) == 5
    assert _remaining_epochs_in_current_stage(
        epoch=11,
        num_epochs=12,
        epochs_per_stage=5,
        single_stage_schedule=False,
    ) == 1


def test_scheduler_step_counts_stage_reset_mode_uses_remaining_stage_epochs():
    updates_per_epoch, total_steps, warmup_steps = _scheduler_step_counts(
        num_batches=7,
        gradient_accumulation_steps=4,
        epoch=6,
        num_epochs=20,
        reset_optimizer=True,
        epochs_per_stage=5,
        single_stage_schedule=False,
        lr_warmup_ratio=0.1,
    )

    # epoch=6 is stage 1 (epochs 5-9), so 4 epochs remain in current stage
    assert updates_per_epoch == 2
    assert total_steps == 8
    assert warmup_steps == 0


def test_scheduler_step_counts_no_reset_mode_uses_remaining_run_epochs():
    updates_per_epoch, total_steps, warmup_steps = _scheduler_step_counts(
        num_batches=7,
        gradient_accumulation_steps=4,
        epoch=6,
        num_epochs=20,
        reset_optimizer=False,
        epochs_per_stage=5,
        single_stage_schedule=False,
        lr_warmup_ratio=0.1,
    )

    assert updates_per_epoch == 2
    assert total_steps == 28
    assert warmup_steps == 2


def test_scheduler_step_counts_clamps_warmup_ratio():
    _, total_steps, warmup_steps = _scheduler_step_counts(
        num_batches=5,
        gradient_accumulation_steps=2,
        epoch=0,
        num_epochs=3,
        reset_optimizer=True,
        epochs_per_stage=5,
        single_stage_schedule=True,
        lr_warmup_ratio=2.0,
    )

    assert total_steps == 9
    assert warmup_steps == total_steps
