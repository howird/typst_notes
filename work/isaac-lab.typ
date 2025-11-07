= IsaacLab

== Creating an Environment

- here we use the `DirectRLEnv` as we want more control than the managed one,
  and don't want to rely on a framework
- `step()`:
  + Pre-process the actions:
    - `_pre_physics_step(actions: Float[Array, "num_envs, action_dim"]) -> None`:
      here we usually just load `actions` passed into `step()` to `self.actions`
  + Apply the actions to the simulator and step through the physics in a
    decimated manner (i.e. apply the same action `decimation` times)
    - `_apply_actions() -> None`: generally, we compute the forces to apply to
      joints, and apply them with
      `self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)`
    - note: for stability, environment steps forward at a fixed time-step, while
      the physics simulation is decimated at a lower time-step. the environment
      time-step is computed as: $"decimation" times "physics_dt"$
      - `decimation`: number of simulation steps per environment step
      - `physics_dt`: physics time-step
  + Compute the reward and done signals.:
    - `_get_rewards() -> Float[Array, "num_envs"]`
    - `_get_dones() -> tuple[Bool[Array, "num_envs"], Bool[Array, "num_envs"]]`:
      (termination, timeout)
  + Reset environments that have terminated or reached the maximum episode
    length
    - `_reset_idx(env_ids: Sequence[int])`: default implementation calls
      `self.scene.reset(env_ids)` and if configured applies randomizations to
      reset scenes
  + Apply interval events if they are enabled
  + Compute observations: `_get_observations()`

- here is what must be implemented, the following methods are called from within
  `step()` and should not be called outside of this class:
  - `_get_observations() -> VecEnvObs`
  - `_get_states() -> VecEnvObs | None`
  - `_setup_scene() -> None`: not required but generally should be done

== Locomotion Environment

- `_compute_intermediate_values`: this is a helper function which computes
  values that would be relevant to the environments, such as variables that
  determine termination conditions,
  - called at the start of `_get_dones()` and at the end of `_reset_idx()`
-
