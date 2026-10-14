import pytest
from le_snail.evaluation import Trial, summarize


def test_fall_can_recover_and_finish_crawl():
    trial = Trial()
    trial.update(0, 0, 0, 1)
    trial.update(10, 0.2, 0, 0)
    assert not trial.finished
    trial.update(20, 0.25, 0, 1)
    assert trial.update(30, 0.51, 0.02, 1)


def test_backtracking_and_repeated_gripper_motion_do_not_count():
    trial = Trial()
    for t, x in enumerate([0., .2, 0., .2, 0., .2, 0.]):
        trial.update(t, x, 0, 1)
    assert not trial.success
    trial.update(120, 0.2, 0, 1)
    assert trial.reason == 'timeout'


def test_goal_after_deadline_fails():
    trial = Trial()
    trial.update(0, 0, 0, 1)
    assert not trial.update(120.01, 0.6, 0, 1)
    assert trial.reason == 'timeout'


def test_goal_exactly_at_deadline_counts():
    trial = Trial()
    trial.update(0, 0, 0, 1)
    assert trial.update(120., 0.5, 0.15, 1)


def test_crossing_lane_and_returning_cannot_pass():
    trial = Trial()
    trial.update(0, 0, 0, 1)
    trial.update(2, 0.1, 0.151, 1)
    assert not trial.update(3, 0.6, 0, 1)
    assert trial.reason == 'left_lane'


def test_recovery_requires_forward_motion_after_standing_up():
    trial = Trial('left')
    trial.update(0, 0, 0, 0)
    trial.update(1, .1, 0, 1)
    trial.update(2, .12, 0, 1)
    assert not trial.success
    assert not trial.update(3, .13, 0, 1)
    assert trial.update(4, .18, 0, 1)


def test_starting_upright_cannot_pass_a_recovery_test():
    trial = Trial('right')
    trial.update(0, 0, 0, 1)
    assert trial.finished and trial.reason == 'recovery_did_not_start_fallen'


def test_fleeting_upright_pose_during_tumbling_does_not_pass():
    trial = Trial()
    trial.update(0, 0, 0, 0)
    trial.update(1, .55, 0, 1)
    trial.update(1.1, .55, 0, 0)
    assert not trial.success


def test_crawl_success_does_not_substitute_for_both_recovery_sides():
    trials = [Trial(finished=True, success=True) for _ in range(5)]
    report = summarize(trials)
    assert report['groups']['crawl']['passed']
    assert not report['passed']


def test_nonfinite_state_fails_closed():
    trial = Trial()
    trial.update(0, float('nan'), 0, 1)
    assert trial.finished and not trial.success
