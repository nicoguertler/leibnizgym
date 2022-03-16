"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      Reward terms defined for Trifinger environment.
"""

from typing import Tuple
# leibnizgym
from leibnizgym.utils.mdp import RewardTerm
from leibnizgym.utils.torch_utils import quat_mul, quat_conjugate, quat_diff_rad
# python
import torch


def linear_schedule_interpolation(step: float, sched_start: float, sched_end: float) -> float:
    val = (step - sched_start) / (sched_end - sched_start)
    val = max(0.0, min(1.0, val))
    return val


@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + 2 + (-scaled).exp())

@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """ Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [pos_offset_local, torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device)],
        dim=-1
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 0:3]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal


@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)):
    num_envs = pose.shape[0]

    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)

    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf


class ObjectDistanceReward(RewardTerm):
    """Reward encouraging movement of the object towards the goal position."""

    def __init__(self, name: str = "object_dist", **kwargs):
        self.name = name
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", 2000)

        self.sched_start = float(kwargs.pop("thresh_sched_start", 0))
        self.sched_end = float(kwargs.pop("thresh_sched_end", 0))
        self.sched_enabled = self.sched_start != self.sched_end

        # initialize base class
        super(ObjectDistanceReward, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, dt: float, curr_sched_step: float, object_state, goal_state) -> torch.Tensor:
        # evaluate the coefficient from scheduling
        if self.sched_enabled:
            # sched_val = linear_schedule_interpolation(curr_sched_step, self.sched_start, self.sched_end)
            sched_val = 1.0 if self.sched_start <= curr_sched_step <= self.sched_end else 0.0
        else:
            sched_val = 1.0

        dist = torch.norm(object_state[:, 0:3] - goal_state[:, 0:3], p=2, dim=-1)
        return self.weight * dt * sched_val * lgsk_kernel(dist)

class ObjectKeypointReward(RewardTerm):
    """Reward encouraging alignment of between keypoints on object and goal."""

    def __init__(self, name: str = "object_keypoint", **kwargs):
        self.name = name
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", 2000)

        # initialize base class
        super(ObjectKeypointReward, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, dt: float, object_state, goal_state) -> torch.Tensor:
        object_keypoints = gen_keypoints(object_state[:, 0:7])
        goal_keypoints = gen_keypoints(goal_state[:, 0:7])

        delta = object_keypoints - goal_keypoints

        dist_l2 = torch.norm(delta, p=2, dim=-1)

        keypoints_kernel_sum = lgsk_kernel(dist_l2, scale=30.).mean(dim=-1)

        return self.weight * dt * keypoints_kernel_sum

class ObjectMoveReward(RewardTerm):
    """Encourages movement of object towards goal."""

    def __init__(self, name: str = "object_move_reward", **kwargs):
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", -750)
        # initialize base class
        super(ObjectMoveReward, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, object_state: torch.Tensor,
                last_object_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: simulation timestep.
            object_state: object state, shape: (N, 13).
            last_object_state: last timestep object state, shape: (N, 13).
            goal_state: goal state, shape: (N, 7).

        Returns:
            The computed reward, shape: (N,)
        """
        curr_norms = torch.norm(object_state[:, 0:3]-goal_state[:, 0:3], dim=-1)
        prev_norms = torch.norm(last_object_state[:, 0:3]-goal_state[:, 0:3], dim=-1)

        return self.weight * (curr_norms - prev_norms)


class ObjectRotationReward(RewardTerm):
    """Reward encouraging movement of the object towards the goal orientation.

    @note Also includes a scheduling term to allow learning of rotation after eg reaching and picking.
    """

    sched_start: float # Timesteps before this reward begins being blended in.
    sched_end: float # Timsteps when this reward is finished being linearly blended in.

    def __init__(self, name: str = "rot_dist", **kwargs):
        self.name = name
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", 100)
        self.epsilon = kwargs.pop("epsilon", 0.1)
        self.scale = kwargs.pop("scale", 1.0)
        # schedule from 50M to 100M steps

        self.sched_start = float(kwargs.pop("thresh_sched_start", 0))
        self.sched_end = float(kwargs.pop("thresh_sched_end", 0))
        self.sched_enabled = self.sched_start != self.sched_end

        # initialize base class
        super(ObjectRotationReward, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, dt: float, curr_sched_step: float, object_state: torch.Tensor,
                goal_state: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        # evaluate the coefficient from scheduling
        if self.sched_enabled:
            # sched_val = linear_schedule_interpolation(curr_sched_step, self.sched_start, self.sched_end)
            sched_val =  1.0 if self.sched_start <= curr_sched_step <= self.sched_end else 0.0
        else:
            sched_val = 1.0

        # extract quaternion orientation
        # Todo (@arthur): add ability to ignore pitch
        quat_a = object_state[:, 3:7]
        quat_b = goal_state[:, 3:7]

        angles = quat_diff_rad(quat_a, quat_b)
        # rot_rew = (sched_val * dt) * lgsk_kernel(angles, scale=3.)
        # rot_rew = (sched_val * dt) / torch.abs(3. * angles + 0.01)
        rot_rew = sched_val * dt / (self.scale*torch.abs(angles) + self.scale)

        return self.weight * rot_rew


class ObjectRotationDeltaReward(RewardTerm):
    """Reward encouraging movement of the object towards the goal orientation with a delta reward.

    @note Also includes a scheduling term to allow learning of rotation after eg reaching and picking.
    """

    sched_start: float # Timesteps before this reward begins being blended in.
    sched_end: float # Timsteps when this reward is finished being linearly blended in.

    def __init__(self, name: str = "rot_dist_delta", **kwargs):
        self.name = name
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", 100)
        # schedule from 50M to 100M steps

        self.sched_start = float(kwargs.pop("linear_schedule_start", 0))
        self.sched_end = float(kwargs.pop("linear_schedule_end", 0))
        self.sched_enabled = self.sched_start != self.sched_end

        # initialize base class
        super(ObjectRotationDeltaReward, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, dt: float, curr_sched_step: float, object_state: torch.Tensor, last_object_state: torch.Tensor,
                goal_state: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        # evaluate the coefficient from scheduling
        if self.sched_enabled:
            sched_val = linear_schedule_interpolation(curr_sched_step, self.sched_start, self.sched_end)
        else:
            sched_val = 1.0

        # extract quaternion orientation
        # Todo (@arthur): add ability to ignore pitch
        quat_a = object_state[:, 3:7]
        last_quat_a = last_object_state[:, 3:7]
        quat_b = goal_state[:, 3:7]

        last_angles = torch.abs(quat_diff_rad(last_quat_a, quat_b))
        angles = torch.abs(quat_diff_rad(quat_a, quat_b))
        rot_delta_rew = sched_val* (angles - last_angles)

        return self.weight * rot_delta_rew


class FingerReachObjectRatePenalty(RewardTerm):
    """Reward encouraging movement of fingers towards the object."""

    def __init__(self, name: str = "finger_reach_object_rate", **kwargs):
        # norm type
        self._norm_p = kwargs.pop("norm_p", 2)
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", -250)

        self.sched_start = float(kwargs.pop("thresh_sched_start", 0))
        self.sched_end = float(kwargs.pop("thresh_sched_end", 0))
        self.sched_enabled = self.sched_start != self.sched_end

        # initialize base class
        super(FingerReachObjectRatePenalty, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, curr_sched_step: float, fingertip_state: torch.Tensor, last_fingertip_state: torch.Tensor,
                object_state: torch.Tensor, last_object_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curr_sched_step: Current step in scheduling.
            fingertip_state: root state tensor for each fingertip, shape (N, 3, 13)
            last_fingertip_state: root state tensor for each fingertip in last timestep, shape (N, 3, 13)
            object_state: root state tensor for the object, shape (N, 13)
            last_object_state: root state tensor for the object in last timestep, shape (N, 13)

        Returns:
            The computed reward, shape: (N,)
        """
        # distance from each finger to the centroid of the object, shape (N, 3).
        curr_norms = torch.stack([
            torch.norm(fingertip_state[:, i, 0:3] - object_state[:, 0:3], p=self._norm_p, dim=-1)
            for i in range(3)
        ], dim=-1)
        # distance from each finger to the centroid of the object in the last timestep, shape (N, 3).
        prev_norms = torch.stack([
            torch.norm(last_fingertip_state[:, i, 0:3] - last_object_state[:, 0:3], p=self._norm_p, dim=-1)
            for i in range(3)
        ], dim=-1)

        if self.sched_enabled:
            # sched_val = linear_schedule_interpolation(curr_sched_step, self.sched_start, self.sched_end)
            sched_val = 1.0 if self.sched_start <= curr_sched_step <= self.sched_end else 0.0
        else:
            sched_val = 1.0

        return self.weight * sched_val * (curr_norms - prev_norms).sum(dim=-1)


class FingertipMovementPenalty(RewardTerm):
    """Reward penalising the movement of fingers."""

    def __init__(self, name: str = "finger_move_penalty", **kwargs):
        # args
        activate = kwargs.pop("activate")
        weight = kwargs.pop("weight", -1.0e-4)
        # initialize base class
        super(FingertipMovementPenalty, self).__init__(name, activate, weight)

    @torch.jit.export
    def compute(self, dt: float, fingertip_state: torch.Tensor,
                last_fingertip_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: simulation timestep.
            fingertip_state: fingertip state, shape: (N, num. of fingers, 13).
            last_fingertip_state: last timestep fingertip state, shape: (N, num. of fingers, 13)

        Returns:
            The computed reward, shape: (N,)
        """
        # compute velocity
        fingertip_vel = (fingertip_state[:, :, 0:3] - last_fingertip_state[:, :, 0:3]) / dt

        return self.weight * fingertip_vel.pow(2).view(-1, 9).sum(dim=-1)


# Mapping of reward terms from names to term
REWARD_TERMS_MAPPING = {
    "finger_reach_object_rate": FingerReachObjectRatePenalty,
    "finger_move_penalty": FingertipMovementPenalty,
    "object_dist": ObjectDistanceReward,
    "object_keypoint": ObjectKeypointReward,
    "object_rot": ObjectRotationReward,
    "object_rot_delta": ObjectRotationDeltaReward,
    "object_move": ObjectMoveReward,
}

# EOF
