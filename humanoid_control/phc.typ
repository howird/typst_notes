= Perpetual Humanoid Control

== Overview

This paper introduces the *Perpetual Humanoid Controller (PHC)*, a physics-based
motion imitator designed for real-time, resilient control of simulated avatars.
It tackles several key challenges in humanoid animation and control.

=== Challenges

- *Challenge: Catastrophic Forgetting in Large-Scale Motion Imitation*
  - *Problem:* When a single reinforcement learning policy is trained on a vast and
    diverse motion dataset (like AMASS, with over 10,000 clips), it tends to "forget"
    how to perform previously learned motions as it learns new, more difficult ones.
  - *Hypothesis:* By progressively allocating new, dedicated network capacity to
    harder tasks, the controller can learn difficult motions without overwriting
    existing skills.
  - *Approach:* The paper proposes the *Progressive Multiplicative Control Policy
    (PMCP)*. This method first trains a baseline "primitive" network on the entire
    dataset. It then identifies the motion sequences that this primitive fails to
    imitate "hard negative mining" and trains a new, separate primitive specifically
    on this harder subset. This process can be repeated, creating a stack of
    specialized primitives. Finally, a "composer" network is trained to dynamically
    combine the outputs of these frozen primitives to control the humanoid.
  - *Alternative Solutions:*
    - *Residual Force Control (RFC):* Using external, non-physical forces to stabilize
      the character. This is effective but can lead to unrealistic artifacts like
      floating or flying.
    - *Mixture of Experts (MoE):* Using multiple expert policies, though these have
      not yet been scaled successfully to the largest motion datasets.

- *Challenge: Handling Noisy Inputs and Recovering from Falls*
  - *Problem:* Controllers often fail when given noisy inputs, such as pose
    estimations from a live video feed, causing the humanoid to fall. Traditional
    methods simply reset the simulation, which is disruptive and ineffective if the
    reference pose is also unreliable.
  - *Hypothesis:* A dedicated recovery skill, learned as a separate task, can allow
    the humanoid to get up and return to the reference motion in a natural,
    human-like way, creating a more robust and seamless experience.
  - *Approach:* Fail-state recovery is treated as another task for a specialized
    primitive ($P^(F)$). When the humanoid falls or is too far from the reference,
    this primitive takes over. It uses a simplified point-goal objective, where the
    goal is just the root position of the reference motion, ignoring the full-body
    pose. This allows the humanoid to get up and walk back to the target location
    before resuming detailed motion imitation. The *Adversarial Motion Prior (AMP)*
    reward is used to ensure the recovery motions appear natural.
  - *Alternative Solutions:*
    - *Resetting:* Teleporting the character to a reference pose upon failure, which
      can be jarring and lead to a vicious cycle of falling and resetting.
    - *Floating-base models:* Using a humanoid model that cannot fall, which
      compromises physical realism.
    - *Non-tracking recovery policies:* Using a recovery policy that does not have
      access to the reference motion, which can result in unnatural, jittery
      movements.

- *Challenge: Requiring Full Rotational Pose Data*
  - *Problem:* Most imitation methods require both 3D joint positions and joint
    rotations as input. However, estimating accurate joint rotations from images is
    a more difficult task than estimating 3D keypoint positions.
  - *Hypothesis:* A controller can be effectively trained using only 3D keypoint
    data, making it more robust to the outputs of common pose estimators and
    simplifying the input requirements.
  - *Approach:* The paper develops and evaluates a "keypoint-based" version of the
    controller. This variant takes only 3D keypoint locations ($hat(p)_( 1:T )$) as
    input and learns to generate the necessary joint torques to match those target
    positions, effectively performing a physics-aware form of inverse kinematics.
  - *Alternative Solutions:* The standard approach is to rely on both joint
    rotations and positions for motion imitation.

=== Proposed Component: Perpetual Humanoid Controller (PHC)

- *High-Level Description:* The PHC is a goal-conditioned policy that learns to
  generate physically realistic character motion. It uses a *Progressive
  Multiplicative Control Policy (PMCP)* architecture, which consists of several
  frozen, pre-trained "primitive" networks and a "composer" network that learns to
  weight their outputs.
- *Inputs:*
  - *Current State ($s_t$):* The character's proprioceptive information (e.g.,
    current joint positions $q_t$ and velocities $dot(q)_t$ from the physics
    simulation).
  - *Goal State ($s_t^g$):* The target reference motion for the next timestep. This
    can be in the form of:
    - Full pose data (joint rotations $hat(theta)_( t+1 )$ and positions $hat(p)_( t+1 )$).
    - Keypoint-only data (3D joint positions $hat(p)_( t+1 )$).
- *Output:*
  - *Action ($a_t$):* A set of target joint angles that are fed into a
    Proportional-Derivative (PD) controller, which then computes the joint torques
    to be applied in the physics simulation.

=== Non-Novel Dependencies

- *Environments/Simulators:*
  - *Isaac Gym:* A GPU-based physics simulation environment used for all training
    and evaluation.
- *Kinematic Models:*
  - *SMPL (Skinned Multi-Person Linear Model):* The kinematic structure and body
    model used for the humanoid character.
- *Datasets:*
  - *AMASS:* A large-scale MoCap dataset used for training the controller.
  - *Human3.6M (H36M):* A dataset used for evaluating the controller on unseen MoCap
    and on noisy inputs from video.
- *Pre-trained Models (for evaluation/demos):*
  - *MeTRAbs:* A model used to estimate 3D keypoints from video.
  - *HybrIK:* A model used to estimate 3D human pose and shape (including joint
    rotations) from video.
  - *Motion Diffusion Model (MDM):* A text-to-motion generative model used to create
    reference clips from language prompts.
  - *YOLOv8 & OCSort:* Used for multi-person detection and tracking in real-time
    video demos.

=== Glaring Assumptions

- For real-time video-based avatar control, the system assumes the camera is
  stationary, level with the ground, and contains no pitch or roll. Camera height
  must be manually adjusted at the start of a session.

== Problem Formulation

The project frames humanoid control as a *goal-conditioned reinforcement
learning problem*. The core objective is to train a policy $pi$ that maximizes
the expected discounted sum of rewards by successfully imitating a reference
motion. This is formalized as a Markov Decision Process (MDP), $cal(M)= angle.l cal(S),cal(A),cal(T),cal(R), gamma angle.r$.

- *State Space ($cal(S)$)*: The state $s_t$ at any time $t$ is composed of the
  character's intrinsic state and its goal: $s_t = (s_t^p, s_t^g)$.
  - *Proprioceptive State ($s_t^p$)*: Contains the character's physical properties,
    such as its 3D body pose ($q_t$), velocity ($dot(q)_t$), and optionally its body
    shape parameters ($beta$).
  - *Goal State ($s_t^g$)*: Encodes the reference motion target. This changes based
    on the controller type and situation.
    - For *rotation-based imitation*, the goal $s_t^"g-rot"$ includes the difference
      between the target and current pose, velocity, and angular velocity, as well as
      the absolute target pose.
    - For *keypoint-based imitation*, the goal $s_t^"g-kp"$ is simpler, containing
      only the difference in position and velocity, and the absolute target position.
    - During *fail-state recovery*, the goal switches based on distance, as defined in
      *Equation (3)*:

$
  s_t^g = cases(
    s_t^"g-imitation" & "if "norm(hat(p)_t^0 - p_t^0)_2 <= 0 . 5 "m", s_t^"g-Fail" & "otherwise",

  )
$

Here, $s_t^"g-Fail"$ is a simplified goal that only provides the relative
position and orientation to the target root, encouraging the character to get up
and navigate back to the reference motion.

- *Action Space ($cal(A)$)*: The policy's action, $a_t in RR^( 23 times 3 )$,
  represents the target joint angles for the 23 actuated joints of the SMPL
  humanoid model. These targets are used by a Proportional-Derivative (PD)
  controller to compute the final joint torques $tau$.

- *Reward Function ($cal(R)$)*: The total reward $r_t$ guides the learning process
  and is a weighted sum of three distinct components, defined by *Equation (1)*:

$
  r_t = 0 . 5 r_t^g + 0 . 5 r_t^"amp" + r_t^"energy"
$

- *Task Reward ($r_t^g$)*: This measures how well the character performs the
  current objective.
  - For motion imitation, the reward $r_t^"g-imitation"$ is calculated with
    *Equation (2)*, which rewards minimizing the error between the simulated and
    reference joint positions, rotations, and velocities:

$
  r_t^"g-imitation" = w_(j p) e^(-100norm(hat(p)_t - p_t)^2) + w_(j r) e^(-10norm(hat(theta)_t minus.circle q_t)^2) + w_(j v) e^(-0 . 1norm(hat(v)_t - v_t)^2) + w_(j omega) e^(-0 . 1norm(hat(omega)_t - omega_t)^2)
$

- For fail-state recovery, the reward $r_t^"g-recover"$ is calculated using
  *Equation (4)*, which combines a point-goal reward (based on reducing distance
  to the target) with the style and energy terms.
- *Style Reward ($r_t^"amp"$)*: An adversarial reward from the *Adversarial Motion
  Prior (AMP)* framework. A discriminator network is trained to distinguish real
  human motion from the character's simulated motion, pushing the policy to
  produce more natural and human-like behavior.
- *Energy Penalty ($r_t^"energy"$)*: A penalty on the product of joint torque and
  angular velocity, which discourages jerky, high-frequency motions.

- *Policy ($pi$)*: The final controller, *PHC*, uses a *Progressive Multiplicative
  Control Policy (PMCP)* architecture. This policy is a combination of multiple "primitive"
  networks, each specializing in a different task or difficulty level, and a "composer"
  network that dynamically weights their outputs. The final action is sampled from
  a Gaussian distribution whose parameters are a weighted product of the
  distributions from all primitives, calculated via *Equation (6)*.

== Pipeline
=== Progressive Imitation Primitive Training

This stage iteratively trains $K$ primitives, each becoming an expert on
progressively harder motions.

- *Inputs (for iteration $k$):*
  - *Primitive Network $cal(P)^(k)$:* A 2-layer MLP [1024, 512]. For $k>1$, its
    weights are initialized from the converged $cal(P)^(k-1)$.
  - *Hard Motion Subset $hat(Q)_"hard"^(k)$:* A collection of motion clips that the
    previous primitive, $cal(P)^(k-1)$, failed to imitate. For the first iteration,
    this is the entire dataset $hat(Q)$.
  - *Value Network $V$ and Discriminator $cal(D)$:* Continuously fine-tuned networks
    from the previous iteration.

- *Process:*
  1. *Train Primitive:* The primitive $cal(P)^k$ is trained using PPO on the motion
    subset $hat(Q)_"hard"^k$ until its performance on this subset converges.
  2. *Reward Calculation:* During training rollouts, the reward for each step is
    calculated using *Equation (1)*, with the task reward being the imitation term $r_t^"g-imitation"$ from
    *Equation (2)*.
  3. *Hard Negative Mining:* Once $cal(P)^k$ is converged, it's evaluated on its
    training data $hat(Q)_"hard"^k$. All sequences it still fails on are collected
    to form the input for the next iteration, $hat(Q)_"hard"^(k+1)$.
  4. *Freeze Weights:* The weights of the converged primitive $cal(P)^k$ are frozen.
    It will no longer be trained.

- *Outputs:*
  - A set of $K$ frozen, expert imitation primitives $cal(P)^(1), ..., cal(P)^(K)$.

=== Fail-State Recovery Primitive Training

This stage trains a dedicated primitive for getting up from falls and returning
to the reference motion.

- *Inputs:*
  - *Recovery Primitive Network $cal(P)^(F)$:* A new MLP, initialized with the
    weights of the final imitation primitive $cal(P)^(K)$.
  - *Locomotion Dataset $Q^"loco"$:* A hand-picked subset of AMASS containing simple
    walking and running motions.
  - The fine-tuned Value Network $V$ and Discriminator $cal(D)$.

- *Process:*
  1. *Train Primitive:* $cal(P)^(F)$ is trained using PPO on the $Q^"loco"$ dataset.
  2. *Specialized Training:* Episodes are initialized by randomly dropping the
    humanoid or placing it far from the reference to generate failure states.
  3. *Reward Calculation:* The reward is calculated using *Equation (4)*, which
    prioritizes reaching the goal destination ($r_t^"g-point"$) while maintaining a
    natural gait via the AMP reward.

- *Output:*
  - A frozen, trained fail-state recovery primitive $cal(P)^(F)$.

=== Composer Training

The final stage trains a composer network to dynamically combine all the learned
primitives.

- *Inputs:*
  - *Frozen Primitives $cal(P)^((1), ..., cal(P)^(K), cal(P)^(F))$:* The complete
    set of expert networks.
  - *Composer Network $cal(C)$:* A 2-layer MLP [1024, 512].
    - *Input:* State $s_t$.
    - *Output:* A weight vector $w_t in RR^( K+1 )$.
  - *Full Motion Dataset $hat(Q)$*.

- *Process:*
  1. *Train Composer:* With all primitives frozen, only the composer network $cal(C)$ is
    trained using PPO on the entire dataset $hat(Q)$.
  2. *Action Generation:* During training, for a given state $s_t$, the composer
    outputs weights $w_t$. The final action is determined by combining the outputs
    of all primitives according to these weights, using the formulation in *Equation
    (6)*.
  3. *Dynamic Reward:* The reward function dynamically switches between the imitation
    reward (*Equation (2)*) and the recovery reward (*Equation (4)*) based on the
    character's distance to the reference motion, as governed by *Equation (3)*.

- *Output:*
  - The fully trained *Perpetual Humanoid Controller (PHC)*, comprising the frozen
    primitives and the trained composer.

== Discussion

Here is a detailed outline of the main questions, experiments, results, and
limitations discussed in the paper's evaluation sections.

=== Motion Imitation on High-Quality Data

==== ❓ Main Question
How well does the Perpetual Humanoid Controller (PHC), without external forces,
imitate high-quality Motion Capture (MoCap) data compared to the
state-of-the-art method (UHC), both with and without its standard stabilizing
forces?

==== 🔬 Experiment Design
- *Task*: The models were tasked with imitating clean MoCap sequences from the
  AMASS and H36M datasets.
- *Baselines*: The primary baseline was the Universal Humanoid Controller (UHC).
  PHC was compared against two versions of UHC:
  1. *UHC with RFC*: The standard implementation using Residual Force Control, a
    non-physical external force that acts as a "hand of God" to prevent falls.
  2. *UHC without RFC*: A version with external forces disabled to provide a
    physically fair comparison.
- *PHC Variants*: Both the primary rotation-based PHC (`Ours`) and the
  keypoint-only version (`Ours-kp`) were evaluated.

==== 📊 Metrics Used
- *Success Rate (Succ)*: The percentage of test sequences completed without the
  humanoid's joints deviating more than 0.5m on average from the reference motion.
  This measures the controller's stability and tracking ability.
- *Position Error (MPJPE)*: The mean per-joint position error, measured in
  millimeters (mm) both globally ($E_"g-mpjpe"$) and relative to the character's
  root ($E_"mpjpe"$). This quantifies imitation accuracy.
- *Physics-based Error*: The difference in velocity ($E_"vel"$) and acceleration ($E_"acc"$)
  between the simulation and the reference motion, which indicates physical
  realism and smoothness.

==== 📈 Results and Significance
- *PHC surpasses the SOTA while being physically pure.* The proposed controller (`Ours`)
  achieved a *98.9% success rate* on the AMASS training set and *96.4%* on the
  AMASS test set. This was higher than UHC *with* its external force assistance
  (97.0% and 96.4% respectively) and drastically better than UHC when its
  assistance was removed (84.5% and 62.6%).
- *The motion is smoother and more realistic.* PHC had a significantly lower
  acceleration error ($E_"acc"$) compared to UHC without RFC, indicating it avoids
  the high-frequency jittery movements that the baseline resorts to for balance.
- *The keypoint-only controller is a strong alternative.* The `Ours-kp` variant
  performed on par with the full-pose version, validating that high-quality motion
  imitation can be achieved using only 3D keypoint data, which is often easier to
  acquire from video.
- *Significance*: These results show that it is possible to train a single,
  physically realistic policy that excels at imitating a massive motion dataset
  without relying on external, non-physical stabilizing forces.

==== ⚠️ Limitations
- The controller still fails to imitate about 1% of the training data.
- These failure cases are typically highly dynamic and acrobatic motions like
  backflips or high jumps. The authors hypothesize that learning these motions
  requires more planning and sequence-level context than the single-frame goal
  provides.

=== Robustness to Noisy Inputs from Video

==== ❓ Main Question
How robust is PHC when driven by noisy and imperfect motion data generated by
real-time, off-the-shelf video pose estimators?

==== 🔬 Experiment Design
- *Task*: Imitate motion sequences generated by running video pose estimators on
  the H36M dataset. These estimators, HybrIK and MeTRAbs, are per-frame models and
  produce noisy output, especially in the depth axis.
- *Input Types*:
  - Joint rotations from HybrIK combined with root position from MeTRAbs.
  - 3D keypoints directly from MeTRAbs (for the `Ours-kp` model).
- *Baselines*: The same UHC and PHC variants from the first experiment were used.

==== 📊 Metrics Used
The same metrics were used: Success Rate (Succ) and Mean Per-Joint Position
Error (global and root-relative).

==== 📈 Results and Significance
- *PHC is highly robust to noise.* On this challenging task, PHC achieved a
  success rate of *~90%*, massively outperforming both UHC with RFC (58.1%) and
  UHC without RFC (18.1%).
- *The keypoint-based model performed best.* `Ours-kp`, when fed keypoints
  directly from MeTRAbs, had the highest success rate (*91.9%*). This suggests
  that directly estimating keypoints from images is a more reliable input source
  than estimating joint rotations, and PHC is robust enough to leverage this.
- *Significance*: This is a critical result for practical applications. It
  validates that PHC can be used as a "drop-in" solution to drive a physically
  simulated avatar directly from a webcam in real-time, successfully bridging the
  gap between noisy computer vision output and stable physics-based animation.

==== ⚠️ Limitations
- There is a noticeable degradation in performance in the live demo compared to
  offline tests.
- This is due to the inherent challenges of real-time processing, including noisy
  depth estimates, difficulty in estimating velocity from jittery inputs, and
  fluctuating framerates.
- The real-time system assumes a stationary, level camera, which must be manually
  configured.

=== Fail-State Recovery Capability

==== ❓ Main Question
Can the controller reliably recover from common failure states (falling over or
being far from the reference) and naturally return to its imitation task?

==== 🔬 Experiment Design
- *Task*: Measure the success rate of recovery from three artificially created
  failure scenarios within 5 and 10 seconds.
- *Scenarios*:
  1. *Fallen-State*: The character is dropped on the ground.
  2. *Far-State*: The character is placed 3 meters away from the target.
  3. *Fallen + Far-State*: A combination of both.
- *Models*: Both the rotation-based and keypoint-based PHC models were evaluated.

==== 📊 Metrics Used
- *Recovery Success Rate (Succ-5s, Succ-10s)*: The percentage of 1000 random
  trials where the controller successfully got up (if fallen) and moved to within
  0.5m of the reference motion to resume tracking.

==== 📈 Results and Significance
- *Recovery is extremely reliable.* Both PHC models recovered with a very high
  success rate (*>92%* within 10 seconds) across all scenarios, including the most
  difficult combined "Fallen + Far" state.
- *Significance*: This confirms the controller's "perpetual" nature. It can
  operate continuously without requiring disruptive manual or automatic resets,
  making it ideal for long-running simulations and live user-driven avatar
  applications where errors and falls are inevitable.

==== ⚠️ Limitations
- While functionally effective, the *naturalness* of the recovery motion could be
  improved.
- The walking gait during recovery sometimes exhibits artifacts like asymmetric
  motion, which is a known side effect of the AMP reward signal.
- The transition from the recovery behavior back to the imitation task can
  sometimes be abrupt or "jolting".

=== Architectural Component Importance

==== ❓ Main Question
Which specific design choices in the PHC architecture are most responsible for
its performance and robustness, especially its ability to learn new skills
without forgetting old ones?

==== 🔬 Experiment Design
- *Task*: An ablation study was performed by systematically removing key
  components from the full PHC model and measuring the drop in performance on the
  noisy video input task.
- *Ablated Components*:
  - *Relaxed Early Termination (RET)*: The rule change allowing more foot freedom
    for balance.
  - *Multiplicative Control Policy (MCP)*: The use of a composer to combine
    primitives.
  - *Progressive Training (PMCP/PNN)*: The core idea of training primitives
    sequentially on failure cases.
  - *Fail-State Recovery Task*: The inclusion of the dedicated recovery primitive.
- *Additional Ablations*: The appendix explored the impact of using MOE vs. MCP,
  the number of primitives, and different PNN connection types.

==== 📊 Metrics Used
Success Rate (Succ) and Position Error (MPJPE) on the noisy H36M-Test-Video
dataset.

==== 📈 Results and Significance
- *Progressive training is the most critical component.* The single biggest
  performance gain (from *66.2% to 86.9%* success) came from adding the
  progressive training pipeline (PMCP). This shows that intelligently allocating
  new network capacity to harder tasks is far more effective than simply making
  the network bigger.
- *New skills can be added without catastrophic forgetting.* Adding the fail-state
  recovery task *after* the imitation primitives were trained did not degrade
  imitation performance. This validates that the PMCP framework successfully
  isolates tasks and prevents the learning of a new skill from erasing an old one.
- *More primitives lead to better performance (to a point).* Performance scaled
  positively with the number of primitives, as each new primitive was able to
  master a new set of difficult motions that the previous ones could not.
- *Significance*: This study provides strong evidence that the proposed *PMCP
  architecture is the key innovation*. It offers a generalizable framework for
  multi-task learning and curriculum learning in complex RL domains, effectively
  solving the catastrophic forgetting problem for this use case.

==== ⚠️ Limitations
- The primary drawback of the progressive training procedure is that it is
  time-consuming, taking approximately one week on a high-end GPU to train the
  full model.

