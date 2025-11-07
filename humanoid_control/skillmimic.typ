= SkillMimic

== Overview

=== Overview of SkillMimic

This paper introduces *SkillMimic*, a data-driven framework for teaching
physically simulated humanoids to perform a wide variety of reusable basketball
skills by imitating human demonstrations. The core idea is to learn skills from
a dataset of human-object interaction (HOI) motions, enabling a single policy to
master diverse skills and reuse them to accomplish more complex tasks.

=== Challenges and Solutions

- *Challenge 1: Learning diverse skills without manual reward engineering.*
  - *Problem*: Traditional reinforcement learning requires manually designing
    specific and labor-intensive reward functions for each individual skill, a
    process that does not scale or generalize well.
  - *Hypothesis*: Just as humans learn by observing and practicing, a simulated
    agent can learn a variety of skills from a dataset of demonstrations by
    mimicking both human and object motions simultaneously.
  - *Approach*: The paper proposes `SkillMimic`, which uses a unified,
    data-driven imitation reward configuration to learn multiple skills (e.g.,
    dribbling, shooting, layups) with a single policy. This removes the need for
    skill-specific reward design.
  - *Alternative*: The alternative is the traditional approach of designing
    unique, complex reward functions for every desired basketball skill, which
    the paper notes would demand significant engineering effort.

- *Challenge 2: Achieving precise and physically correct human-object contact.*
  - *Problem*: Standard imitation rewards focus only on kinematic properties
    (positions, velocities) and are insufficient for ensuring correct physical
    contact, often leading to physically implausible or failed interactions
    (e.g., using the head to control the ball).
  - *Hypothesis*: Explicitly modeling and rewarding correct contact states is
    critical for learning precise and successful physical interactions.
  - *Approach*: The paper introduces the *Contact Graph (CG)*, a simple and
    general method to model the contact state between body parts (hands, rest of
    body) and the object (basketball). A corresponding *Contact Graph Reward
    (CGR)* is used to explicitly encourage the agent to replicate the contact
    patterns from the reference motion, preventing local optima where kinematics
    are matched but contact is wrong.
  - *Alternative*: Relying solely on kinematic rewards, which the paper shows
    leads to suboptimal and incorrect contact behaviors.

- *Challenge 3: Accomplishing complex, long-horizon tasks.*
  - *Problem*: Learning a complex sequence of actions, such as dribbling down
    the court and scoring a layup, is extremely difficult to train from scratch
    with simple, goal-based rewards.
  - *Hypothesis*: A library of pre-learned, reusable skills can serve as a
    powerful "skill prior," simplifying the learning process for complex tasks
    by allowing a higher-level policy to simply choose which skill to execute.
  - *Approach*: A hierarchical learning solution is proposed where a *High-Level
    Controller (HLC)* is trained on top of the fixed, pre-trained `SkillMimic`
    policy. The HLC observes the task state (e.g., basket position) and selects
    which low-level skill (e.g., 'dribble forward', 'layup') to activate,
    allowing it to solve complex tasks using only simple goal-related rewards.
  - *Alternative*: Training a policy from scratch (PPO) or using hierarchical
    methods with general motion priors (ASE) instead of discrete interaction
    skills, both of which failed to converge on the complex tasks presented.

=== Proposed Component: Skill Policy

- *Component*: The core of the framework is the *Skill Policy*, a single neural
  network trained with reinforcement learning.
- *Function*: It learns to perform a wide variety of basketball skills by
  imitating a motion-capture dataset.
- *Inputs*:
  - *HOI State ($s_t$)*: The current state of the simulation, including the
    humanoid's local body positions, rotations, velocities, and fingertip
    contact forces, as well as the basketball's local position, rotation, and
    velocities.
  - *Skill Label ($c_j$)*: A one-hot encoded vector specifying the desired skill
    to execute (e.g., "pickup", "shot", "dribble left").
- *Output*:
  - *Action ($a_t$)*: A set of target joint rotations that are fed into PD
    controllers to generate torques and actuate the humanoid model.

=== Dependencies for Reproduction

- *Datasets*:
  - *BallPlay-M*: The primary dataset used for training, containing 35 minutes
    of diverse basketball skills captured with an optical motion capture system.
    This dataset is a novel contribution of the paper.
  - *BallPlay-V*: A secondary dataset estimated from monocular RGB videos, used
    to test robustness against data inaccuracies. This dataset is also a novel
    contribution.
  - *GRAB*: A pre-existing dataset of whole-body human grasping motions, used in
    ablation studies to demonstrate the generalizability of the proposed reward
    functions.
- *Physics Simulation Environment*:
  - *Isaac Gym*: All experiments are conducted within the Isaac Gym physics
    simulation platform, which enables massively parallel execution on a GPU.

=== Glaring Assumptions

- *Simulation Fidelity*: The method assumes that the physics simulation is a
  reasonable approximation of reality. Notably, the ball's density was modified
  to $1000~"kg"/"m"^3$ to "enhance stability and accelerate training
  convergence," which is a significant deviation from real-world physics.
- *Data Availability*: The entire data-driven framework is predicated on the
  existence of a comprehensive, well-labeled dataset of Human-Object Interaction
  clips for the desired skills.

=== Missing Perspectives from Introduction

- *Sim-to-Real Transfer*: The introduction and the paper as a whole focus
  exclusively on learning skills within a physics simulation. There is no
  discussion of the significant challenges associated with transferring these
  learned policies to a physical humanoid robot (the "sim-to-real gap").

=== Recommended Prerequisite Reading

- *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based
  Character Skills* by Peng et al. (2018). This paper is foundational for the
  concept of using reference motion data and imitation rewards to train agents
  for dynamic skills in a physics simulation, a core concept SkillMimic builds
  upon.
- *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*
  by Peng et al. (2021). This paper introduces the use of adversarial imitation
  learning, which improves upon the DeepMimic framework by enhancing versatility
  and reducing data alignment constraints, a technique relevant to the broader
  field of motion imitation.

== Problem Formulation

=== Problem Formulation

The project formulates the task of learning basketball skills as a *Markov
Decision Process (MDP)*. The goal is to train a policy that maximizes the
expected discounted return by imitating a collection of human-object interaction
(HOI) demonstrations.

==== Markov Decision Process (MDP)

The problem is defined by the tuple $cal(M)=s,a,f,r,gamma$, where:
- $s$ is the set of states.
- $a$ is the set of actions.
- $f(s_t+1|a_t,s_t)$ is the transition dynamics function, governed by the
  physics simulator.
- $r(s_t,a_t,s_t+1)$ is the reward function.
- $gamma in [0, 1]$ is the discount factor.

The objective is to learn a conditional policy $pi(a_t|s_t, c)$ that maximizes
the expected return $cal(R)(pi)$:

$
  cal(R)(pi) = EE_(p pi)(tau) [sum(t = 0)^(T - 1) gamma^t r_t ]
$

where $tau$ is the trajectory and $c$ is a condition representing the skill
label.

==== State and Action Space

- *State Space ($s_t$)*: The state observed by the policy is a concatenation of
  proprioceptive, contact, and object information, all transformed into the
  humanoid's root-local coordinate frame.

$
  s_t = o_t^"prop", o_t^f, o_t^"obj" quad (1)
$

- $o_t^"prop"$: Humanoid proprioception, including local body position,
  rotation, position velocity, and angular velocity.
- $o_t^f$: Net contact forces for all fingertips.
- $o_t^"obj"$: Object observation, including its local position, rotation,
  velocity, and angular velocity.

- *Action Space ($a_t$)*: The policy models a Gaussian distribution, and the
  action $a_t$ is a sample from this distribution. This action represents the
  target joint rotations for a full set of PD controllers that actuate the
  humanoid.

==== Unified HOI Imitation Reward

The reward function $r_t$ is a crucial component, designed to be unified across
all skills. It is calculated as a product of five sub-rewards to encourage
balanced learning.

$
  r_t = r_t^b dot.op r_t^o dot.op r_t^"rel" dot.op r_t^"reg" dot.op r_t^"cg" quad (2)
$

- *Contact Graph Reward ($r_t^"cg"$)*: This reward encourages precise imitation
  of physical contacts.

$
  r_t^"cg" = e x p(-sum_(j = 1)^J lambda^"cg" [j] dot.op e_t^"cg" [j]) quad (3)
$

where $e_t^"cg" = |s_t^"cg" - hat(s)_t^"cg"|$ is the element-wise absolute
difference between the simulated and reference contact states, and $lambda^"cg"$
is a sensitivity hyperparameter.

- *Body Kinematics Reward ($r_t^b$)*: Measures the similarity of the humanoid's
  motion to the reference.

$
  r_t^b = r_t^p dot.op r_t^r dot.op r_t^"pv" dot.op r_t^"rv" quad (4)
$

This includes rewards for joint position ($r_t^p$), rotation ($r_t^r$), position
velocity ($r_t^"pv"$), and angular velocity ($r_t^"rv"$). Each sub-reward
follows the general form:

$
  r_t^p = e x p(-lambda^p dot.op e_t^p) quad "where" quad e_t^p = "MSE"(s_t^p, hat(s)_t^p) quad (5)
$

Here, $s_t^p$ and $hat(s)_t^p$ are the simulated and reference positions,
respectively.

- *Object Kinematics Reward ($r_t^o$)*: Ensures the object's movement matches
  the reference.

$
  r_t^o = r_t^"op" dot.op r_t^"or" dot.op r_t^"opv" dot.op r_t^"orv" quad (6)
$

This includes rewards for object position ($r_t^"op"$), rotation ($r_t^"or"$),
position velocity ($r_t^"opv"$), and angular velocity ($r_t^"orv"$), calculated
similarly to Equation (5).

- *Relative Motion Reward ($r_t^"rel"$)*: Constrains the relative motion between
  the object and key body points to match the reference, also calculated using
  the form in Equation (5).

- *Velocity Regularization ($r_t^"reg"$)*: Suppresses high-frequency jitter in
  the humanoid's movements.

$
  r_t^"reg" = e x p(-lambda^"reg" dot.op e_t^"acc") quad (7)
$

where $e_t^"acc"$ is a measure of the simulated joint accelerations relative to
the reference velocities.

== Pipeline

The implementation follows a reinforcement learning loop where a policy is
trained to imitate skills from a motion dataset.

*Stage 1: Data Acquisition and Processing*
- *Description*: Real-world basketball skills are captured and processed into a
  structured dataset of human-object interactions.
- *Input*: Live performance of basketball skills by a human.
- *Process*: An optical motion capture system records the 3D positions of
  markers on the player and the ball at 120 fps. This raw data is processed to
  yield skeleton joint rotations and ball trajectories. Clips are labeled with
  the corresponding skill (e.g., "Pickup", "Layup").
- *Output*: The *BallPlay-M dataset*, a collection of motion clips. Each clip
  contains a sequence of reference states $hat(s)_0...N$ and a skill label
  $c_j$.

*Stage 2: Environment Initialization (RSI)*
- *Description*: At the beginning of each training episode, a parallel
  simulation environment is initialized to a specific state from the reference
  dataset.
- *Input*: The BallPlay-M dataset.
- *Process*: For each of the 2048 parallel environments, the system randomly
  selects a skill and a corresponding motion clip. It then selects a random
  frame from that clip to set the initial positions and velocities of the
  simulated humanoid and the basketball. This technique is known as *Reference
  State Initialization (RSI)*.
- *Output*: An initial simulation state $s_0$ for each parallel environment.

*Stage 3: State Observation*
- *Description*: The current state of the simulation is gathered and formatted
  into a tensor to be fed into the policy network.
- *Input*: The current full state of the physics simulation at timestep $t$.
- *Process*: The system queries the simulator for the humanoid's joint
  positions, velocities, and fingertip contact forces, as well as the ball's
  position and velocity. All coordinates are converted to be local to the
  humanoid's root. These are concatenated into a single flat vector.
- *Output*: A state tensor $s_t$ for each environment, as defined by *Equation
  (1)*.

*Stage 4: Action Generation*
- *Description*: The policy network takes the current state and skill label, and
  outputs an action.
- *Input*:
  - State tensor $s_t$ from Stage 3.
  - One-hot encoded skill label tensor $c_j$. Shape: `(2048, num_skills)`.
- *Process*: The state and skill label are concatenated and passed through a
  3-layer MLP. The network outputs the mean of a Gaussian distribution for each
  degree of freedom. An action is sampled from this distribution.
- *Output*: An action tensor $a_t$ representing target joint rotations. Shape:
  `(2048, num_DOFs)`, where `num_DOFs` is 156 for the BallPlay-M humanoid model.

*Stage 5: Physics Simulation*
- *Description*: The action is applied in the physics simulator to advance the
  state of the world.
- *Input*:
  - Action tensor $a_t$ from Stage 4.
  - The current simulation state.
- *Process*: The target joint rotations in $a_t$ are fed to PD controllers,
  which calculate the necessary torques to apply to the humanoid's joints. Isaac
  Gym's physics engine simulates the effect of these torques and gravity,
  calculating the state for the next timestep.
- *Output*: The new simulation state $s_t+1$.

*Stage 6: Reward Calculation*
- *Description*: The reward function is evaluated to provide a learning signal
  to the policy.
- *Input*:
  - The simulated state transition ($s_t -> s_t+1$).
  - The corresponding reference state $hat(s)_t+1$ from the motion clip.
- *Process*: The system calculates the five sub-rewards (body, object, relative,
  regularization, contact) by comparing the simulated state to the reference
  state using *Equations (3) through (7)*. The final reward $r_t$ is computed by
  multiplying these sub-rewards, as shown in *Equation (2)*.
- *Output*: A scalar reward value $r_t$ for each environment. Shape:
  `(2048, 1)`.

*Stage 7: Policy Update*
- *Description*: The collected experiences are used to update the policy
  network's weights via a reinforcement learning algorithm.
- *Input*: A batch of experience tuples $(s_t, c_j, a_t, r_t, s_t+1)$ collected
  from the parallel environments.
- *Process*: The project uses the *Proximal Policy Optimization (PPO)*
  algorithm. PPO uses the collected rewards to estimate an advantage function
  and then performs a clipped gradient update to the policy and value networks,
  aiming to maximize the expected future return.
- *Output*: Updated weights for the policy network. The process then repeats
  from Stage 3 with the new policy.

== Discussion

Here is a detailed outline of the primary questions investigated in the paper's
results and the limitations discussed.

=== How does SkillMimic compare to existing skill-learning methods?

This question assesses if the proposed unified, data-driven approach for
learning interaction skills is more effective than baselines that primarily
focus on imitating body motion or use simpler reward structures.

- *Experiments and Ablations*:
  - The primary method, *SkillMimic (SM)*, was compared against four baselines
    on four key basketball skills (Pickup, Dribble Forward, Shot, Layup).
  - *Body-Only Baselines*: DeepMimic (DM) and Adversarial Motion Priors (AMP),
    which imitate only body movements without explicit object perception.
  - *Reward-Ablation Baselines*: Two variants of SkillMimic were created to
    isolate the benefit of the proposed reward design. *SkillMimic-DM* used an
    additive reward structure inspired by DeepMimic, and *SkillMimic-AMP* used
    an adversarial reward.
  - Imitation quality was also evaluated on the *GRAB* and *BallPlay-V* datasets
    to test performance with different objects and noisy data.

- *Metrics Used*:
  - *Success Rate*: A skill-specific binary measure to determine if a task was
    accomplished (e.g., for "Pickup," if the ball was lifted above 1 meter).
  - *HOI Accuracy (Acc.)*: A per-frame metric deeming imitation accurate only if
    object position error, body position error, and contacts are all correct
    simultaneously.
  - *Position Error (MPJPE)*: The mean per-joint position error for the body
    ($E_"b-mpjpe"$) and object ($E_"o-mpjpe"$) to measure kinematic accuracy.
  - *Contact Error ($E_"cg"$)*: Measures the error in replicating the reference
    contact states.

- *Results and Significance*:
  - SkillMimic *significantly outperformed* all baselines across all basketball
    skills. The body-only methods (DM, AMP) failed almost completely,
    demonstrating that object perception and interaction rewards are essential.
  - The reward-ablation baselines (SM-DM, SM-AMP) showed some success but were
    unstable, highlighting the importance of SkillMimic's multiplicative reward
    structure and explicit contact modeling for avoiding local optima.
  - *Significance*: The results strongly validate the paper's core hypothesis: a
    unified framework that imitates both *human and object motion* while
    explicitly modeling *physical contact* is critical and superior for learning
    complex interaction skills.

=== Is the proposed Contact Graph Reward (CGR) essential?

This question is an ablation study designed to isolate and prove the importance
of explicitly rewarding correct physical contact, which is a key novelty of the
paper.

- *Experiments and Ablations*:
  - The full SkillMimic model was compared against a version with the Contact
    Graph Reward removed (`SkillMimic w/o CGR`).
  - The experiments were run on the GRAB and BallPlay-V datasets, which feature
    diverse interactions and potential data inaccuracies, making them good
    testbeds for this feature.

- *Metrics Used*:
  - *Qualitative Visuals*: The paper provides visual examples of failure cases
    when CGR is absent.
  - *Quantitative Metrics*: HOI Accuracy (Acc.) and Contact Error ($E_"cg"$)
    were used to measure the performance difference.

- *Results and Significance*:
  - Without CGR, the agent learned to exploit "kinematic local optima"—achieving
    a goal with physically incorrect contact, such as using its head to
    stabilize the ball or its hands to support itself on a table.
  - Quantitatively, removing CGR caused the Contact Error to *increase
    significantly* and the overall imitation Accuracy to drop sharply.
  - *Significance*: This ablation provides direct evidence that CGR is *critical
    for learning precise and physically plausible interactions*. Kinematic
    rewards alone are insufficient and can lead the agent to learn nonsensical
    behaviors.

=== Does SkillMimic scale with more data?

This question investigates whether the framework benefits from larger datasets,
a key property for any data-driven method. It was tested in two ways: scaling
data for a single skill and scaling the number of skills learned jointly.

- *Experiments and Ablations*:
  - *Single Skill Scaling*: The "pickup" skill was trained on four different
    dataset sizes: 1, 10, 40, and 131 motion clips. The policy's ability to
    generalize was tested by having it retrieve balls from random locations.
  - *Mixed vs. Individual Training*: Four skills were trained in two ways: (1)
    each skill trained with its own separate policy, and (2) all four skills
    trained jointly within a single policy.

- *Metrics Used*:
  - *Success Rate*: Used to evaluate the performance of the pickup skill and the
    success of both individual skills and zero-shot skill switching in the mixed
    training experiment.

- *Results and Significance*:
  - Pickup performance *improved dramatically* with more data, rising from a
    0.5% success rate with 1 clip to 85.6% with 131 clips, showcasing much
    better generalization.
  - *Mixed training* was superior to individual training. It improved the
    robustness of individual skills (preventing overfitting) and enabled
    *successful zero-shot skill switching*, which was not present in the
    training data.
  - *Significance*: These results demonstrate two crucial findings:
    1. The performance of SkillMimic is *scalable and improves with more data*,
      indicating its potential for learning even more complex skills given
      larger datasets.
    2. *Jointly training multiple skills* leads to better generalization and
      allows the policy to learn shared representations that facilitate smooth
      transitions between skills.

=== Can learned skills be reused to solve complex, long-horizon tasks?

This question evaluates the final stage of the proposed pipeline: using the
learned low-level skills as building blocks for a high-level controller to solve
more sophisticated tasks.

- *Experiments and Ablations*:
  - A High-Level Controller (HLC) was trained to select which skill the
    pre-trained SkillMimic policy should execute. This hierarchical approach was
    tested on four complex tasks: *Throwing, Heading, Circling, and Scoring*.
  - The proposed method was compared against two strong baselines: (1) training
    a policy from scratch (*PPO*) and (2) using a hierarchical method with
    general motion priors (*ASE*). All methods used the same simple, goal-based
    task rewards.

- *Metrics Used*:
  - *Success Rate*: Task-specific rules were used to measure the final
    performance of each method.
  - *Normalized Return*: Learning curves were plotted to compare the training
    speed and convergence of each method.

- *Results and Significance*:
  - The proposed method *achieved high success rates (80-93%)* and converged
    rapidly on all four complex tasks.
  - In stark contrast, both PPO and ASE *completely failed to learn*, showing
    near-zero success rates and flat learning curves.
  - *Significance*: This demonstrates the power of using the learned skills as
    *"skill priors."* By abstracting away low-level control, the HLC can solve a
    much simpler exploration problem, enabling it to learn long-horizon tasks
    that are intractable to learn from scratch with sparse, goal-based rewards.

=== What are the method's primary limitations?

This is a summary of the shortcomings identified by the authors in the
discussion section, which point toward future research directions.

- *Limitations Identified*:
  1. *Object Generalization*: The framework is not designed to handle
    generalization across objects with different shapes or properties; it is
    specialized for the basketball seen during training.
  2. *Reliance on Privileged Information*: The policy receives the exact state
    of the object directly from the simulator. It lacks a true perception system
    that could identify and track multiple objects in a more complex scene.
  3. *Handling of Conflicting Data*: Because the policy is memoryless, it cannot
    disambiguate situations where the same state can lead to multiple different
    outcomes in the training data (e.g., from a standing state, the human could
    continue standing or start to dribble). The current workaround is to
    manually separate such cases into different skill labels.
