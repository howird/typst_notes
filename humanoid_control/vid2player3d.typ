= Learning Physically Simulated Tennis Skills

== Overview

This paper introduces a system that enables physically simulated 3D characters
to learn a diverse set of tennis skills by observing unannotated broadcast
videos. The goal is to produce controllers that can synthesize realistic,
long-horizon tennis rallies, capturing different player styles without explicit
mocap data or stroke labels.

=== Challenges & Solutions

- *Challenge: High-Cost Data Acquisition & Low-Quality Video Motion*
  - Traditional motion capture (mocap) is expensive and impractical for large-scale
    sports data collection. Conversely, while broadcast video is abundant, the 3D
    motion extracted from it using standard pose estimators is often noisy and
    physically implausible, exhibiting artifacts like jitter and foot-sliding.
  - *Hypothesis:* Physics-based imitation can be used to "clean" the noisy kinematic
    motion data, resulting in a physically plausible and smoother motion dataset
    that is a better foundation for learning complex skills.
  - *Approach:*
    1. First, extract initial kinematic motion ($MM_"kin"$) from videos using
      off-the-shelf pose estimation models.
    2. Then, train a low-level imitation policy using deep reinforcement learning (DRL)
      to track this noisy motion. The resulting physically-corrected motion ($MM_"corr"$)
      is dynamically consistent and free of major artifacts.
  - *Alternative:* Training directly on the noisy kinematic data. An ablation study
    showed this approach (w/o PhysicsCorr) leads to significantly worse motion
    quality and lower task performance.

- *Challenge: Learning Diverse, Unlabelled Skills*
  - The source videos contain a wide variety of tennis strokes (serves, topspins,
    slices), but these actions are not explicitly labeled. The challenge is to learn
    a single versatile controller that can execute these different skills as
    required by the game state.
  - *Hypothesis:* A hierarchical control system, where a high-level policy learns to
    select actions in a learned motion embedding, can implicitly learn to generate
    appropriate strokes based on simple task rewards (e.g., hitting the ball to a
    target), without needing explicit stroke annotations.
  - *Approach:*
    1. A conditional VAE (MVAE) is trained on the physically-corrected motion data ($MM_"corr"$)
      to create a low-dimensional motion embedding.
    2. A high-level policy is trained with DRL to select latent codes ($z$) from this
      embedding to generate full-body motion sequences that accomplish a task, such as
      returning a ball with a specific spin.
  - *Alternative:* An end-to-end DRL policy. The paper builds on prior work favoring
    hierarchical models for composing complex skills.

- *Challenge: Precision Control with Imperfect Motion Data*
  - Even after physics-based correction, subtle but critical motions, particularly
    wrist and hand movements, are not accurately reconstructed from video due to
    motion blur and occlusion. This lack of precision prevents the character from
    successfully hitting the ball with accuracy.
  - *Hypothesis:* A hybrid control scheme that allows the high-level policy to
    directly control critical joints (like the wrist) while leveraging the motion
    embedding for general body movement will enable the precision required for the
    task.
  - *Approach:* The high-level policy's action simultaneously outputs a latent code
    for the MVAE (for body motion) and corrective Euler angles for the wrist and
    elbow joints. This allows it to fine-tune the racket orientation at contact.
    Simple kinematic heuristics are also applied to ensure the character's head
    tracks the ball and to enforce two-handed grips.
  - *Alternative:* Relying solely on the MVAE to generate the full-body motion. The
    paper's ablation study (w/o HybridCtr) demonstrated that this results in a
    drastic drop in task performance, with the bounce-in rate dropping by nearly
    half and position error increasing significantly.

=== Proposed System at a Glance

- *Component:* A hierarchical, four-stage video imitation pipeline.
- *Input:* Raw, unannotated broadcast tennis videos.
- *Output:* A versatile controller for a physically simulated character capable of
  playing extended tennis rallies with a diverse set of shots and styles.

=== Dependencies for Reproduction

- *Environments/Datasets:*
  - *Tennis Videos:* A collection of 13 US Open matches from the Tennis Channel
    (2017-2021) featuring Roger Federer, Novak Djokovic, and Rafael Nadal. In total,
    this comprises ~279 minutes of motion data.
  - *AMASS Dataset:* A large-scale mocap database used for pre-training the general
    low-level motion imitation policy.
  - *Issac Gym:* A GPU-accelerated physics simulation environment used for training
    all policies.
- *Pre-trained Models / Algorithms:*
  - *YoloV4:* For player detection and tracking.
  - *ViTPose:* For 2D pose keypoint extraction.
  - *HybrIK:* For estimating 3D human pose and SMPL shape parameters from single
    images.
  - *GLAMR (like approach):* For global root trajectory optimization.
  - *SMPL Model:* The character's body is based on the Skinned Multi-Person Linear
    model.

=== Glaring Assumptions

- The physics model for the tennis equipment is a simplification of reality. The
  racket head is modeled as a flat cylinder and the grip is directly attached to
  the character's wrist joint, which may limit the faithful reproduction of
  nuanced, professional-level swing dynamics.
- The system assumes the ball's future trajectory can be accurately predicted and
  is provided as an input to the high-level policy. This provides the agent with
  perfect foresight, which is an advantage not available in real-world play where
  trajectory prediction is an acquired skill involving uncertainty.

== Problem Formulation

The project is formulated as a hierarchical reinforcement learning problem,
where low-level and high-level policies are trained as Markov Decision Processes
(MDPs).

=== Low-Level Imitation Policy

This policy learns to control a physically simulated character to mimic a
reference motion. It is defined by an MDP, $cal(M)=(cal(S),cal(A),cal(T),r,gamma)$.

- *States ($s_t$)*: The state describes the character's physical status and the
  target motion it should follow. It includes:
  - Current simulated joint positions ($p_t$), linear velocities ($dot(p)_t$), local
    rotations ($q_t$), and angular velocities ($dot(q)_t$).
  - Target kinematic joint positions ($hat(p)_t+1$) and rotations ($hat(q)_t+1$) for
    the next frame from the reference motion.

- *Actions ($a_t$)*: The action is a combination of target joint angles for PD
  controllers and supplementary root forces.
  - $a_t = (u_t, eta_t)$, where $u_t$ is the target joint angles and $eta_t$ represents
    residual forces and torques applied to the character's root.
  - The target angles $u_t$ are used by Proportional-Derivative (PD) controllers to
    compute actuation torques $tau_t$ for each non-root joint:

$
  tau_t = k_p dot.op(u_t - q_t^(n r)) - k_d dot.op dot(q)_t^(n r) quad"Eq. 1"
$

where $k_p$ and $k_d$ are fixed gain parameters.

- *Reward Function ($r_t$)*: The reward encourages close tracking of the reference
  motion while penalizing excessive energy use. It is a weighted sum of five
  terms:

$
  r_t = omega_o r_t^o + omega_v r_t^v + omega_P r_t^p + omega_k r_t^k + omega_e r_t^e quad"Eq. 2"
$

- *Joint Rotation Reward ($r_t^o$)*: Measures the geodesic distance between
  simulated ($q_t^j$) and reference ($hat(q)_t^j$) joint rotations.

$
  r_t^o = exp [ -alpha_o sum_j (lr(|lr(|q_t^j minus.circle hat(q)_t^j |)|)^2)] quad"Eq. 3"
$

- *Joint Velocity Reward ($r_t^v$)*: Measures the difference between simulated ($dot(q)_t^j$)
  and reference ($hat(dot(q))_t^j$) joint velocities.

$
  r_t^v = exp [ -alpha_v sum_j (lr(|lr(|dot(q)_t^j - hat(dot(q))_t^j |)|)^2)] quad"Eq. 4"
$

- *Joint Position Reward ($r_t^p$)*: Measures the distance between simulated ($x_t^j$)
  and reference ($hat(x)_t^j$) 3D world joint positions.

$
  r_t^p = exp [ -alpha_P sum_j (lr(|lr(|x_t^j - hat(x)_t^j |)|)^2)] quad"Eq. 5"
$

- *Keypoint Reward ($r_t^k$)*: Measures the distance between projected 2D joint
  positions ($overline(x)_t^j$) and the original 2D keypoints ($hat(overline(x))_t^j$)
  from video annotation.

$
  r_t^k = exp [ -alpha_k sum_j (lr(|lr(|macron(x)_t^j - hat(macron(x))_t^j |)|)^2)] quad"Eq. 6"
$

- *Power Penalty ($r_t^e$)*: Penalizes the energy expenditure computed from joint
  velocities and torques.

$
  r_t^e = - sum_j (lr(|lr(|dot(q)_t^j dot.op tau_t^j |)|)^2) quad"Eq. 7"
$

=== High-Level Motion Planning Policy

This policy learns to perform the tennis task (e.g., hit a ball to a target) by
selecting motions from a learned embedding. It is also formulated as an MDP.

- *States ($s_t$)*: Describes the character, the ball, and the task goal.
  - *Character State*: The character's current pose, represented with the same
    features used to train the motion embedding (global root position/orientation,
    relative joint positions, etc.).
  - *Ball State*: The ball's predicted position for the next 10 frames.
  - *Control Targets*: The desired bounce position on the opponent's court and the
    desired spin (topspin or backspin).

- *Actions ($a_t$)*: Consists of a latent code to generate the main body motion
  and joint corrections for precision.
  - $a_t = (z, c)$, where $z$ is a latent code for the MVAE model and $c$ is a
    vector of three Euler angle corrections for the wrist and elbow joints of the
    swing arm.

- *Reward Function*: A two-part reward function guides the policy before and after
  ball contact.
  - *Racket Position Reward (Pre-Contact)*: Encourages the policy to move the racket
    ($x_t^r$) close to the ball ($x_t^b$) at the ideal contact phase ($theta_t approx pi$).

$
  r_t^r = exp(-alpha_r lr(|lr(|x_t^r - x_t^b |)|)^2) dot.op exp(-alpha_theta lr(|lr(|theta_t - pi|)|)^2) quad"Eq. 8"
$

- *Ball Position Reward (Post-Contact)*: Rewards hitting the ball to the target
  bounce position ($hat(x)^b$) with the correct spin ($hat(s)^b$).

$
  r_t^b = cases(
    -10 & "if " s^b != hat(s)^b, exp(-alpha_b lr(|lr(|macron(x)^b - hat(x)^b |)|)^2) & "if " s^b = hat(s)^b,

  ) quad"Eq. 9"
$

where $overline(x)^b$ is the actual bounce position and $s^b$ is the actual spin
direction.

=== Physics Modeling of Tennis

The flight of the tennis ball is governed by aerodynamic forces.

- *Forces*: The simulation adds external air drag ($F_d$) and Magnus force ($F_M$)
  to the ball.

$
  F_d = (C_d A v^2)/2, quad F_M = (C_L A v^2)/2 quad"Eq. 10"
$

where $v$ is ball velocity, $A = pi rho R^2$ is a constant based on air density $rho$ and
ball radius $R$, $C_d$ is the drag coefficient, and $C_L$ is the lift
coefficient.
- *Lift Coefficient ($C_L$)*: This term, which models the effect of spin, is
  calculated as:

$
  C_L = 1/(2 + v /v_"spin")
$

where $v_"spin"$ is the magnitude of the ball's spin velocity.

== Pipeline

The system is implemented in four sequential stages, as shown in Figure 2 of the
paper.

=== Video Annotation

This stage processes raw video to create a dataset of kinematic motions.

- *Input*: Raw broadcast tennis videos (1080p, 30fps).
- *Process*:
  1. Off-the-shelf models (YoloV4, ViTPose, HybrIK) are used to detect players and
    estimate their 2D keypoints and 3D SMPL poses in camera coordinates.
  2. The camera matrix is estimated from court lines to transform poses into a global
    court coordinate system. The root trajectory is refined via an optimization to
    minimize re-projection error.
  3. Key events (player ID, ball contact times) are manually annotated to aid in
    later stages.
- *Output*: A kinematic motion dataset ($MM_"kin"$) containing time-series data of
  SMPL poses and global trajectories for each player shot.

=== Low-Level Imitation

This stage corrects the noisy kinematic data and creates a reusable motion
tracking policy.

- *Inputs*:
  - The kinematic motion dataset $MM_"kin"$ from Stage 1.
  - The AMASS mocap dataset for pre-training.
  - A physically simulated character (72-DOF, SMPL-based) in the Isaac Gym
    environment.
- *Process*:
  1. A DRL policy is first trained on the large-scale AMASS dataset to learn to
    imitate general human motions.
  2. This policy is then fine-tuned on the tennis-specific $MM_"kin"$ dataset to
    specialize it for tennis movements.
  3. During fine-tuning, the policy is optimized to maximize the reward from *Eq. 2*,
    which drives the simulated character to follow the reference motion from $MM_"kin"$ as
    closely as possible while remaining physically plausible. The torques used in
    the power penalty (*Eq. 7*) are calculated using *Eq. 1*.
- *Outputs*:
  - *A trained low-level imitation policy ($pi_"low"$)*, which can make the
    character track any given reference motion.
  - *A physically corrected motion dataset ($MM_"corr"$)*, created by using the
    final $pi_"low"$ to track every motion in $MM_"kin"$ and recording the resulting
    physically-plausible simulated motion.

=== Motion Embedding

This stage learns a compressed, generative representation of tennis motions.

- *Input*: The physically corrected motion dataset $MM_"corr"$ from Stage
  2.
- *Process*: A conditional VAE (MVAE) is trained on $MM_"corr"$ to learn a
  low-dimensional latent space. The MVAE learns to predict the character's pose in
  the next frame based on the current pose and a latent vector $z$.
- *Output*: *A trained MVAE model* (specifically, its decoder). The decoder can
  take a character pose and a latent vector $z$ and autoregressively generate a
  sequence of novel, human-like tennis motions.

=== High-Level Motion Planning

This final stage trains an agent to use the learned motion model to accomplish
the tennis task.

- *Inputs*:
  - The trained MVAE decoder from Stage 3.
  - The trained low-level policy $pi_"low"$ from Stage 2.
  - The current state of the simulation (character pose, future ball trajectory,
    task goal).
- *Process*:
  1. A high-level DRL policy is trained using a curriculum that gradually increases
    task difficulty.
  2. At each step, the policy receives the current state and outputs a latent code $z$ and
    wrist/elbow joint corrections.
  3. The MVAE decoder generates a target body pose from $z$, which is then modified
    by the joint corrections and other kinematic heuristics (e.g., head-tracking).
  4. This final target pose is passed as a reference to the low-level policy $pi_"low"$,
    which computes the physical torques (*Eq. 1*) to move the simulated character.
  5. The high-level policy receives a reward based on the outcome of the shot,
    calculated using *Eq. 8* and *Eq. 9*. The ball physics governing the outcome are
    defined by *Eq. 10*.
- *Output*: *A trained high-level policy ($pi_"high"$)* that serves as the final
  controller. It can generate actions to make the character play extended rallies
  against an opponent.

== Discussion

=== How effectively can the system learn diverse and complex tennis skills from video?

- *Experiment*: A single controller was trained using 80 minutes of Roger
  Federer's motion data. This controller was then tested in its ability to execute
  a wide variety of shots—including serves, topspin forehands, topspin backhands,
  and backhand slices—and to hit incoming balls to a target location with a
  specified spin (topspin or backspin).
- *Metrics*:
  - *Task Performance*: `Hit rate` (fraction of times the racket contacts the ball), `Bounce-in rate` (fraction
    of shots that land in the opponent's court), and `Bounce position error` (average
    distance from the actual bounce spot to the target).
  - *Qualitative Analysis*: Visual inspection of the diversity of the generated
    motions, as shown in the paper's figures.
- *Results & Significance*: The controller performed with high accuracy, achieving
  a median hit rate of 92% and a median bounce-in rate of 85% across 10,000 test
  sessions. The median bounce position error was less than two meters. Critically,
  the controller learned to perform the appropriate swing for different spins
  (e.g., a topspin backhand vs. a slice backhand) without any explicit stroke
  labels in the training data. This result is significant because it validates
  that a hierarchical approach with simple, task-oriented rewards is sufficient to
  learn a complex and versatile set of skills from unannotated video.
- *Limitations*: While the controller successfully captured the broad
  characteristics of tennis strokes, it failed to reproduce the fine-grained,
  nuanced details of a professional athlete's motion. For instance, the generated
  motions had shorter follow-throughs, less realistic wrist pronation, and did not
  capture how players steady the racket with their non-swinging hand during
  preparation. The authors suggest this is due to limitations in both the motion
  extraction from video and the fidelity of the physics simulation.

=== Can the system capture the distinct playing styles of different players?

- *Experiment*: To test for style capture, the system was used to train three
  separate controllers, each based on video data from a different professional
  player: Roger Federer (right-handed, one-handed backhand), Novak Djokovic
  (right-handed, two-handed backhand), and Rafael Nadal (left-handed, two-handed
  backhand).
- *Metrics*:
  - *Qualitative Analysis*: Visual comparison of the motion generated by each
    controller to see if it reflected the known, signature style of the
    corresponding player.
  - *Task Performance*: The hit rate, bounce-in rate, and bounce position error were
    measured for all three controllers to ensure they were all high-performing.
- *Results & Significance*: The system successfully learned effective controllers
  for all three players, each achieving high task performance. Qualitatively, the
  generated animations captured the coarse, defining attributes of each player's
  style, such as their dominant hand and whether they used a one- or two-handed
  backhand. The significance lies in demonstrating that the model can learn
  personalized motion styles directly from large-scale video data, opening the
  door to creating simulated characters that behave and appear like specific,
  real-world individuals.
- *Limitations*: The limitation is the same as above: the system captures only "gross
  aspects" of player style, not the subtle details of professional footwork and
  swings.

=== How do the proposed solutions handle low-quality motion data from videos?

This question was investigated through two key ablation studies targeting the
paper's core contributions.

==== Importance of Physics-Based Motion Correction

- *Ablation*: A controller was trained by bypassing the physics-correction step.
  The motion embedding was built directly from the noisy, kinematically-estimated
  motion (`w/o PhysicsCorr` condition).
- *Metrics*:
  - *Motion Quality*: `Jitter` (third derivative of joint positions) and `Foot sliding` (displacement
    of grounded feet between frames).
  - *Task Performance*: Hit rate, bounce-in rate, and bounce position error.
- *Results & Significance*: Without physics correction, the final animated motion
  showed significantly more jitter and foot sliding. Task performance also
  dropped, especially the bounce position error, indicating less precise control.
  This ablation confirms the authors' hypothesis that "cleaning" the noisy video
  data with a physics-based imitation policy is a critical step for achieving both
  high-quality motion and high task performance.

==== Effectiveness of Hybrid Wrist Control

- *Ablation*: A controller was trained where the high-level policy could *only*
  select motions from the learned embedding and was not allowed to output direct
  joint corrections for the wrist (`w/o HybridCtr` condition).
- *Metrics*: Task Performance (hit rate, bounce-in rate, bounce position error).
- *Results & Significance*: While the agent could still hit the ball often, the
  *bounce-in rate was nearly cut in half*, and the bounce position error increased
  dramatically. This result is highly significant as it proves that motion data
  from video is too imprecise (especially for fast-moving or occluded parts like
  the wrist) for high-precision tasks. The proposed hybrid control, which allows
  the RL agent to override the reference motion, is essential to compensate for
  these perception errors and achieve accurate control of the racket.

=== What is the trade-off between physical realism and task performance?

- *Experiment*: The system was trained with and without *residual force control*.
  Residual forces are external forces the policy can apply to the character's
  root, which helps in tracking agile motions but is not strictly physically
  realistic, as it can compensate for imperfections in the physics model.
- *Metrics*:
  - *Motion Quality*: Foot sliding.
  - *Task Performance*: Hit rate, bounce-in rate, and bounce position error.
- *Results & Significance*: Removing residual forces produced more realistic
  motion, with *40% less foot sliding*. However, this came at a cost: the hit rate
  dropped by 12% and the bounce-in rate dropped by 15%. This highlights a
  fundamental trade-off in the field: perfect physical realism can sometimes
  hinder performance on a task if the simulation model isn't perfect. The paper's
  system allows users to make this trade-off based on their needs.
- *Limitations*: The need for residual forces to achieve the highest performance
  indicates that the current physics model and character actuation are not
  perfectly matched to a real human, leaving room for future work to improve
  simulation fidelity.

=== How does the amount of training data affect performance?

- *Experiment*: The system was retrained multiple times using progressively larger
  subsets of Federer's video data, ranging from 12.5% (~10 minutes) to 100% (80
  minutes).
- *Metrics*:
  - *Task Performance*: Hit rate and bounce-in rate.
  - *Motion Quality*: Jitter of the generated motion.
- *Results & Significance*: Performance scaled directly with the amount of data.
  Both task performance and motion quality consistently improved as more motion
  data was used to build the motion embedding. Performance was significantly worse
  with only 10 minutes of data, as the learned motion space was too sparse for
  effective planning. This result strongly supports the paper's core motivation:
  that leveraging *large-scale* video datasets is a viable and effective path to
  creating more capable and realistic character controllers.
