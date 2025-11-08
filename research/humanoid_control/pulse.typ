= Universal Humanoid Motion Representation

This paper introduces *PULSE* (Physics-based Universal motion Latent SpacE), a
universal motion representation for physics-based humanoid control, akin to a
foundation model for motion. It can be reused for a wide variety of downstream
tasks, from locomotion and terrain traversal to free-form VR motion tracking,
without task-specific modifications.

== Overview

=== Challenges and Solutions

- *Challenge 1: Limited Scope of Existing Motion Representations*
  - *Problem*: Prior methods learn from small, specialized motion datasets
    (e.g., only locomotion), resulting in latent spaces that can only produce a
    narrow range of behaviors and fail to generalize to complex, free-form
    tasks. Attempts to use large, diverse datasets like AMASS had not yielded
    satisfactory results.
  - *Hypothesis*: A universal motion representation can be created by first
    training a "master" policy that can imitate a comprehensive range of
    motions, and then distilling its skills into a probabilistic latent space.
    The authors posit that using *direct online distillation* from a powerful,
    pre-trained imitator is the key to effectively scaling to large,
    unstructured datasets.
  - *Approach*:
    1. First, an imitator policy, *PHC+*, is trained via reinforcement learning
      to successfully imitate nearly all motions from the large-scale AMASS
      dataset.
    2. Then, PULSE is trained to learn this comprehensive skill set via *online
      distillation* from the frozen, pre-trained PHC+ teacher. PULSE uses a
      variational information bottleneck to model the distribution of motor
      skills.
  - *Alternative Solutions Considered*:
    - *Adversarial Learning* (e.g., ASE, CALM): Effective on small, curated
      datasets but struggles to capture the motor skill diversity in large,
      unorganized datasets like AMASS.
    - *Kinematics-based Latent Spaces* (e.g., HuMoR): Learned without physics
      simulation, these models can generate physically implausible motions.

- *Challenge 2: Inefficient Exploration for Downstream Tasks*
  - *Problem*: When using a latent space for hierarchical RL, exploration can be
    highly inefficient. Randomly sampling from a large latent space often
    produces incoherent actions, which slows down the learning of new tasks.
  - *Hypothesis*: A learnable prior conditioned on the agent's own state
    (proprioception) can provide a much better starting point for exploration
    than random noise. Sampling from this informed prior will generate more
    coherent, human-like behaviors, improving sampling efficiency and
    accelerating learning for downstream tasks.
  - *Approach*: PULSE jointly learns a *conditional prior*, $cal(R)(z_t|s_t^p)$,
    that models the distribution of likely actions based on the humanoid's
    current pose and velocities ($s_t^p$). For new tasks, the high-level policy
    learns to output a residual action that perturbs the mean of this prior,
    guiding exploration towards plausible motions.
  - *Alternative Solutions Considered*:
    - *Standard VAE Prior*: Using a simple zero-mean Gaussian prior ignores the
      current state, leading to unguided and difficult exploration.
    - *Supervising with the Prior*: Directly forcing the task policy's
      distribution to match the prior's can create an adverse feedback loop if
      the policy explores states the prior has not seen, making the prior's
      guidance uninformative.

=== High-Level Model Overview: PULSE

PULSE is an encoder-decoder model that learns a probabilistic latent space for
humanoid motor skills.

- *Inputs (during training/distillation):*
  - *Proprioception* ($s_t^p$): The humanoid's current 3D body pose ($q_t$) and
    velocities ($dot(q)_t$).
  - *Imitation Goal* ($s_t^"g-mimic"$): The target reference motion frame from
    the AMASS dataset.

- *Outputs (from internal components):*
  - The *Encoder ($cal(E)$)* outputs a distribution over the latent space,
    $cal(N)(mu_t^e,sigma_t^e)$.
  - The *Conditional Prior ($cal(R)$)* outputs a prior distribution conditioned
    on proprioception, $cal(N)(mu_t^p,sigma_t^p)$.
  - The *Decoder ($cal(D)$)* takes a latent code $z_t$ and proprioception
    $s_t^p$ to produce the final motor action $a_t$ (PD controller targets).

- *Usage in Downstream Tasks:*
  1. The Encoder ($cal(E)$) is discarded. The Prior ($cal(R)$) and Decoder
    ($cal(D)$) are frozen.
  2. A new high-level task policy ($pi_"task"$) is trained.
  3. This policy takes the current proprioception ($s_t^p$) and a task-specific
    goal ($s_t^g$) as input and outputs a residual latent action.
  4. This residual is added to the mean ($mu_t^p$) from the frozen Prior. The
    resulting latent code is passed to the frozen Decoder to generate the final
    motor command.

=== Dependencies for Reproduction

- *Datasets*:
  - *AMASS (Archive of Motion Capture as Surface Shapes)*: A cleaned version is
    used for training both the PHC+ teacher and the PULSE model. A subset
    containing only locomotion is used for initial state sampling in some tasks.
  - *QuestSim Dataset*: A real-world motion dataset used for evaluating the VR
    controller tracking task.
- *Pre-trained Models/Architectures*:
  - *PHC (Perpetual Humanoid Controller)*: The work heavily builds on this
    model, creating an improved version called *PHC+* which serves as the
    "teacher" for knowledge distillation.
  - *SMPL (Skinned Multi-Person Linear Model)*: The humanoid's kinematic
    structure is based on the SMPL mean shape.
- *Simulation Environment*:
  - *Isaac Gym*: The GPU-based physics simulator used for all experiments.

== Problem Formulation

The overarching goal is to learn a universal, physics-based humanoid motion
representation, *PULSE*, that can be reused across a wide variety of downstream
control tasks. This is framed as a knowledge distillation problem, where skills
from a powerful "teacher" imitation policy are transferred to a structured,
probabilistic latent space model (the "student").

=== State and Action Definitions
The system is modeled as a goal-conditioned Markov Decision Process (MDP),
$cal(M) = angle.l cal(S), cal(A), cal(T), cal(R), gamma angle.r$.

- *Humanoid Pose & Motion:*
  - The full-body pose at time $t$ is $q_t = (theta_t, p_t)$, where
    $theta_t in RR^J times 6$ are the 6D rotations and $p_t in RR^J times 3$ are
    the 3D positions for $J$ joints.
  - The motion is described by velocities $dot(q)_t = (omega_t, v_t)$, where
    $omega_t in RR^J times 3$ and $v_t in RR^J times 3$ are the angular and
    linear velocities.
- *State Space $cal(S)$:*
  - The agent's internal, or *proprioceptive*, state is
    $s_t^p = (q_t, dot(q)_t)$.
  - The full state for the policy is $s_t = (s_t^p, s_t^g)$, which includes a
    task-specific goal state $s_t^g$.
- *Action Space $cal(A)$:*
  - The action $a_t in RR^23 times 3$ specifies the targets for a
    Proportional-Derivative (PD) controller for each of the 23 actuated joints.

=== Learning the Latent Space via a Variational Information Bottleneck
The core of PULSE is a student policy $pi_"PULSE" = (cal(E), cal(D), cal(R))$
that learns from a pre-trained teacher $pi_"PHC+"$. It consists of three
components:
1. An *Encoder* $cal(E)$ that maps the current state and an imitation goal to a
  latent distribution.
2. A *Decoder* $cal(D)$ that maps the state and a latent code to a motor action.
3. A learnable *Conditional Prior* $cal(R)$ that models the distribution of
  latent codes based only on the humanoid's proprioception.

The encoder and prior are modeled as diagonal Gaussian distributions:

$
  cal(E)(z_t |s_t^p, s_t^(g "-mimic")) = cal(N)(mu_t^e, sigma_t^e) quad, quad cal(R)(z_t |s_t^p) = cal(N)(mu_t^p, sigma_t^p) quad(1)
$

The objective is to maximize the evidence lower bound (ELBO) of the
log-likelihood of producing the teacher's action:

$
  log P(a_t |s_t^p, s_t^(g "-mimic")) >= EE_cal(E) [log cal(D)(a_t |s_t^p, z_t)] - D_(K L)(cal(E)(z_t |s_t^p, s_t^(g "-mimic"))||cal(R)(z_t |s_t^p)) quad(2)
$

This is optimized using a practical loss function that combines a supervised
action-matching term, a KL-divergence term from the ELBO, and a temporal
smoothing regularizer:

$
  cal(L) = underbrace(||a_t^"PHC+" - a_t ||_2^2, cal(L)_"action") + alpha underbrace(||mu_t^e - mu_(t - 1)^e ||_2^2, cal(L)_"regu") + beta underbrace(D_(K L)(cal(E)||cal(R)), cal(L)_(K L)) quad(3)
$

=== Hierarchical Control for Downstream Tasks
For a new task, the trained Decoder $cal(D)$ and Prior $cal(R)$ are frozen. A
new high-level policy $pi_"task"$ learns to output a residual action in the
latent space. The final action is computed by adding this residual to the mean
of the learned prior:

$
  a_t^"task" = cal(D)(pi_"task" (z_t |s_t^p, s_t^g) + mu_t^p) quad(4)
$

This formulation guides exploration towards plausible, human-like motions.

== Implementation Pipeline

The project is implemented in a three-stage pipeline.

=== Train the Teacher Imitator (PHC+)
- *Goal:* Create a single, robust policy that can imitate the vast majority of
  motions from the large-scale AMASS dataset.
- *Inputs:*
  - *AMASS Dataset $hat(Q)$:* A large collection of motion capture sequences,
    cleaned to remove corrupted data.
  - *Physics Simulator:* Isaac Gym.
- *Process:*
  1. A goal-conditioned policy $pi_"PHC+"$ is trained using Reinforcement
    Learning (PPO).
  2. At each step, the goal $s_t^"g-mimic"$ is the next frame from a reference
    motion in $hat(Q)$.
  3. A progressive training scheme with hard-negative mining is used. The policy
    is iteratively trained on sequences it previously failed to imitate,
    improving its overall capability.
- *Outputs:*
  - *Frozen Teacher Policy $pi_"PHC+"$:* A neural network that maps a state and
    imitation goal to a motor action.
    - *Input Tensor (State):* A flattened vector representing
      $(s_t^p, s_t^"g-mimic")$.
    - *Output Tensor (Action $a_t^"PHC+"$):* Shape $(N, 69)$, where $N$ is the
      batch size and 69 corresponds to the PD targets for the 23 joints.

=== Distill Skills into PULSE distill
- *Goal:* Transfer the comprehensive motor skills from the frozen teacher
  $pi_"PHC+"$ into the structured latent space of the student model, PULSE.
- *Inputs:*
  - *Frozen Teacher Policy $pi_"PHC+"$* from Stage 1.
  - *AMASS Dataset $hat(Q)$* (for providing initial states and imitation goals).
  - *Physics Simulator.*
- *Process (Online Distillation):*
  1. Initialize the three networks of the student policy: Encoder $cal(E)$,
    Decoder $cal(D)$, and Prior $cal(R)$.
  2. Roll out episodes using the *student policy*. For each step $t$ in the
    simulation: a. The student's *Encoder* $cal(E)$ takes the state $s_t^p$ and
    goal $s_t^"g-mimic"$ to produce a latent distribution
    $cal(N)(mu_t^e, sigma_t^e)$. b. A latent code $z_t$ is sampled. c. The
    student's *Decoder* $cal(D)$ uses $s_t^p$ and $z_t$ to generate the
    student's action $a_t$, which is executed in the simulator.
  3. *Annotate Experience:* For the same state $(s_t^p, s_t^"g-mimic")$, query
    the frozen *teacher policy* $pi_"PHC+"$ to get the "expert" action
    $a_t^"PHC+"$.
  4. *Update Student:* The student networks are updated via supervised learning
    using the loss function from *Equation 3*. This loss forces the student's
    action $a_t$ to match the teacher's $a_t^"PHC+"$, while also regularizing
    the latent space.
- *Outputs:*
  - *Frozen Decoder $cal(D)$ and Prior $cal(R)$:* These two networks now form
    the universal motion representation. The Encoder $cal(E)$ is discarded.
    - *Latent Code $z_t$:* Tensor of shape $(N, 32)$.
    - *Prior Mean $mu_t^p$:* Tensor of shape $(N, 32)$.
    - *Decoder Output Action $a_t$:* Tensor of shape $(N, 69)$.

=== Apply to Downstream Tasks with Hierarchical Control
- *Goal:* Efficiently solve new, arbitrary tasks by using the learned motion
  representation as a structured action space.
- *Inputs:*
  - *Frozen Decoder $cal(D)$ and Prior $cal(R)$* from Stage 2.
  - *New Task Definition:* A reward function and goal state $s_t^g$ (e.g., for
    terrain traversal, this includes a height map and target trajectory).
  - *Physics Simulator.*
- *Process:*
  1. Define a new, high-level policy $pi_"task"$ that operates in the latent
    space.
  2. During RL training for the new task: a. The high-level policy $pi_"task"$
    observes the state $(s_t^p, s_t^g)$ and outputs a *residual latent action*
    $z_t^"res"$. b. The frozen *Prior* $cal(R)$ calculates the prior mean
    $mu_t^p$ from the current proprioception $s_t^p$. c. The final latent action
    is computed by applying *Equation 4*: $z_t = z_t^"res" + mu_t^p$. d. The
    frozen *Decoder* $cal(D)$ converts the final latent action $z_t$ into the
    low-level motor torque action $a_t^"task"$, which is applied in the
    simulator.
  3. The high-level policy $pi_"task"$ is updated with PPO to maximize the
    task-specific reward.
- *Outputs:*
  - *Trained Task Policy $pi_"task"$:* A specialized high-level controller for
    the new task.
    - *Input Tensor (State):* A flattened vector representing $(s_t^p, s_t^g)$.
    - *Output Tensor (Residual Action $z_t^"res"$):* Shape $(N, 32)$.

== Discussion

Here is a detailed outline of the primary questions the paper aimed to answer,
the experiments designed to address them, and the corresponding results and
limitations.

=== Fidelity: Can the model effectively distill and retain skills?

*Question:* How well does the distilled model, PULSE, retain the comprehensive
motor skills from its "teacher," PHC+, after being compressed into a
probabilistic latent space?

- *Experiment Design:*
  - A direct comparison of motion imitation quality between the original teacher
    (PHC+), the distilled student (PULSE), and the authors' previous model
    (PHC).
  - The evaluation was performed on both the training and test splits of the
    cleaned AMASS motion capture dataset.

- *Metrics Used:*
  - *Success Rate (Succ):* The percentage of motion sequences completed
    successfully without significant deviation.
  - *Position Error ($E_g"-mpjpe", E_"mpjpe"$):* Global and root-relative mean
    per-joint position error, measured in millimeters.
  - *Physics-based Error ($E_"acc", E_"vel"$):* Error in joint acceleration and
    velocity.

- *Results & Significance:*
  - PULSE successfully retained most of the teacher's abilities, achieving a
    *99.8%* success rate on the training data and *97.1%* on the test data. This
    was only a minor drop from PHC+'s near-perfect scores.
  - *This is significant because it validates the core premise of the paper:* it
    is possible to use a variational information bottleneck to distill the vast
    skills of a complex, universal imitator into a structured latent space
    without catastrophic loss of performance. The resulting decoder inherits the
    ability to perform a wide range of motions from the AMASS dataset.

- *Limitations:*
  - The distillation process is a *"lossy compression"*. PULSE does not achieve
    a 100% success rate like its teacher, and its position and velocity errors
    are slightly higher. This indicates that the information bottleneck, while
    enabling probabilistic modeling, introduces a trade-off that sacrifices some
    imitation fidelity.

=== Generalization: Is the latent space effective for new tasks?

*Question:* How effective is the PULSE latent space as a foundation for training
policies on new, complex generative tasks compared to other state-of-the-art
latent space models and training from scratch?

- *Experiment Design:*
  - Four generative downstream tasks were used: controlling *speed*, *reaching*
    for a point, *striking* a target, and navigating complex *terrain* (stairs,
    slopes, obstacles).
  - PULSE's performance was compared against two adversarial latent space models
    (*ASE* and *CALM*) and a baseline policy *trained from scratch* without any
    pre-trained representation.

- *Metrics Used:*
  - *Normalized Return:* Training curves showing the undiscounted task reward
    over millions of samples, normalized by the maximum possible reward.
  - *Qualitative Assessment:* Visual inspection of the resulting motions to
    evaluate their naturalness and realism.

- *Results & Significance:*
  - PULSE *outperformed all baselines* in final performance and convergence
    speed across all four tasks.
  - Critically, policies trained with PULSE produced *natural and human-like
    behaviors* (e.g., jumping-like motions on stairs) without needing an
    explicit style or adversarial reward, which was required by other methods.
  - *This demonstrates the primary value of PULSE:* it provides a highly
    informative and structured action space where exploration is more efficient
    and naturally biased towards plausible human motion, accelerating learning
    for complex tasks. It also showed that adversarial methods like ASE and CALM
    failed to effectively capture the diversity of the large AMASS dataset.

- *Limitations:*
  - While training from scratch was generally outperformed, it was the closest
    competitor in terms of reward. The paper observes that this can lead to
    unnatural behavior but does not provide a quantitative metric for
    "naturalness," relying on qualitative video evidence.

=== Universality: Can the model handle unconstrained, free-form motion?

*Question:* Is the PULSE representation truly "universal," or is it limited to
generative tasks? Can it handle a highly unconstrained estimation task like VR
controller tracking?

- *Experiment Design:*
  - A *VR controller tracking* task was set up, where the policy must infer
    full-body motion from only three 6DOF inputs (headset and two hand
    controllers).
  - PULSE was compared against other latent space models (ASE, CALM) and a
    specialized policy trained from scratch on this specific task.
  - Evaluation was done on the AMASS dataset and a real-world dataset
    (QuestSim).

- *Metrics Used:*
  - Success Rate, position/velocity/acceleration errors (on AMASS), and global
    hand position error (on the real-world dataset).
  - Training curves were compared to show convergence speed.

- *Results & Significance:*
  - PULSE *significantly outperformed* other latent space models, validating
    that its representation captures a wide enough range of motor skills for
    free-form motion.
  - Its performance was *comparable to the specialized from-scratch policy*, and
    it was able to successfully track all 14 real-world sequences.
  - Furthermore, using PULSE as a foundation allowed the policy to *converge
    faster* than training from scratch.
  - *This result is significant* as it confirms the "universal" claim. The
    latent space is not just a collection of predefined skills but a flexible
    foundation that can be adapted to challenging estimation tasks.

- *Limitations:*
  - The specialized from-scratch policy still achieved slightly lower tracking
    errors, indicating that a universal model may *trade peak performance for
    generality*. The paper notes that PULSE sometimes sacrifices tracking
    precision in favor of maintaining stability, unlike the specialized policy.

=== 4. Ablations: Which components of the framework are essential? 🔧

*Question:* What are the individual contributions of the key design choices in
PULSE, such as the learnable prior, the residual action formulation, and the
temporal regularization?

- *Experiment Design:*
  - A series of ablation studies were conducted on the challenging VR controller
    tracking task. Different versions of PULSE were trained with specific
    components removed:
    1. Without the temporal regularization term ($cal(L)_"regu"$).
    2. Without a learnable prior (using a standard VAE's zero-mean Gaussian
      prior).
    3. With a learnable prior, but not using it as a residual base for the
      action.
    4. Training PULSE with a mixed RL and supervised objective instead of pure
      distillation.

- *Metrics Used:*
  - Success Rate and imitation error metrics on the AMASS test set for the VR
    tracking task.

- *Results & Significance:*
  - *All tested components proved to be crucial.* Removing any of them caused a
    significant drop in performance.
  - The *learnable prior* and its use as a *residual action base* were essential
    for effective exploration; without them, the task was too hard to solve.
  - The *temporal regularization* ($cal(L)_"regu"$) was vital for creating a
    smooth and continuous latent space that is easier for an RL policy to
    navigate.
  - Mixing RL objectives into the distillation phase had a *negative effect*,
    creating a noisy latent space and worse performance.
  - *These results strongly justify the paper's specific architectural choices,*
    showing that the combination of these components is what makes the framework
    successful.

- *Limitations:*
  - These ablations were performed only on the most challenging downstream task
    (VR tracking) because of its sensitivity. The authors note that simpler
    generative tasks were less affected by the expressiveness of the latent
    space, so the measured impact might not be as dramatic for easier tasks.
