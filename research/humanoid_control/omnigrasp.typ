= OmniGrasp

== Overview

This paper introduces *Omnigrasp*, a method for controlling a simulated,
dexterous humanoid to grasp a wide variety of objects and move them along
complex trajectories. The core of the method is a hierarchical reinforcement
learning (RL) framework that leverages a pre-trained, universal motion
representation to make the complex control problem tractable.

=== Challenges

- *Challenge 1: High-Dimensional Control*
  - Controlling a simulated humanoid with a high degree of freedom (153 DoF),
    including dexterous hands, is incredibly complex. It requires maintaining
    balance while executing precise arm and finger movements, which poses a
    severe exploration problem for standard RL algorithms.
  - *Approach:* The paper proposes using a pre-trained universal and dexterous
    humanoid motion representation, *PULSE-X*, as a structured action space for
    the RL policy. Instead of outputting low-level joint actuations directly,
    the policy outputs a low-dimensional latent code ($z_t in RR^48$). A
    pre-trained decoder then translates this code into natural, human-like joint
    actuations.
  - *Hypothesis:* A compact, structured action space that inherently produces
    coherent, human-like motion significantly simplifies the exploration problem
    and increases sample efficiency, enabling the agent to learn complex
    grasping tasks with a simple reward function.
  - *Alternatives Considered:*
    - *Disembodied Hand:* Using a floating hand simplifies the problem but lacks
      physical realism as it is not constrained by a body.
    - *Separate Controllers:* Training separate controllers for the body and
      hands introduces issues with instability and synchronization, and existing
      body imitators have tracking errors too large for precise grasping.
    - *Part-Based Priors:* Using separate motion priors for the body and hands
      adds system complexity and does not scale well to free-form motions like
      following random trajectories.

- *Challenge 2: Generalization Across Diverse Objects and Trajectories*
  - Prior work often focuses on simple, predefined trajectories (e.g., vertical
    lifts) or is limited to a single object or interaction sequence. Scaling a
    single policy to handle thousands of different object shapes and arbitrary
    trajectories is a significant hurdle.
  - *Approach:* The system is trained on over 1200 diverse objects and follows
    randomly generated 3D trajectories. This removes the dependency on limited
    and costly motion-captured (MoCap) human-object interaction data. For
    training efficiency with many objects, a simple hard-negative mining process
    is used to focus on objects the policy struggles with.
  - *Hypothesis:* By training on a massive diversity of synthetically generated
    trajectories and a large corpus of objects, the policy can learn a
    generalizable grasping skill that applies to unseen objects and novel
    trajectories without requiring paired human-object motion data.
  - *Alternatives Considered:*
    - *Specialized Policies:* Training one policy for each specific task or
      object, which is not scalable.
    - *Curriculum Learning:* Employing complex, structured curricula to
      gradually increase object or task difficulty.

=== Proposed Component: Omnigrasp

- *Description:* Omnigrasp is a *hierarchical reinforcement learning policy*
  trained with Proximal Policy Optimization (PPO). It functions as the
  high-level controller in a two-stage framework. The first stage involves
  training the PULSE-X motion representation, which is then frozen and used by
  the Omnigrasp policy in the second stage. The policy network itself is a Gated
  Recurrent Unit (GRU) to handle the sequential nature of the task.
- *Inputs (at test time):*
  - The object's mesh.
  - The desired object trajectory as a sequence of future poses
    ($hat(q)_(1:T)^"obj"$).
  - The policy's internal state includes the humanoid's proprioception (joint
    positions, velocities, contact forces) and a goal state containing an object
    shape latent code, the future object trajectory, and the relative position
    between the hands and the object.
- *Outputs:*
  - The Omnigrasp policy outputs a residual latent action
    ($z_t^"omnigrasp" in RR^48$).
  - This latent action is combined with the output of a prior network and fed
    into the frozen PULSE-X decoder ($cal(D)_"PULSE-X"$), which produces the
    final action: target positions for the Proportional-Derivative (PD)
    controllers of the humanoid's 51 actuated joints ($a_t in RR^(51 times 3)$).

=== Non-Novel Dependencies

- *Simulation Environment:*
  - *Isaac Gym:* Used for all high-performance, GPU-based physics simulations.
- *Datasets:*
  - *AMASS, GRAB, Re:InterHand:* These motion capture datasets were combined and
    augmented to create the "dexterous AMASS dataset" used to train the
    underlying PULSE-X motion representation.
  - *GRAB:* Used for training and for evaluating trajectory following on MoCap
    data.
  - *OakInk:* Provided a large repository of over 1700 diverse object meshes for
    training the grasping policy at scale.
  - *OMOMO:* Provided meshes for several large objects (e.g., chairs, lamps) to
    test the policy's ability to handle bigger items.
- *Pre-trained Models & Architectures:*
  - *PULSE / PHC:* The architectures for the motion representation (PULSE-X) and
    motion imitator (PHC-X) are extensions of these prior works.
  - *GrabNet / OakInk Grasp Generator:* An off-the-shelf grasp generator was
    used to create synthetic "pre-grasp" poses, which provide crucial reward
    signals during training.
  - *SMPL-X:* The kinematic structure of the simulated humanoid is based on this
    body model.
  - *Basis Point Set (BPS):* This method is used to encode object mesh geometry
    into a fixed-size latent vector for the policy's input.

=== Assumptions

- *Simulation-Based:* The entire method is developed and evaluated within a
  physics simulator. While potential for sim-to-real transfer is mentioned, it
  is not demonstrated, and the policy relies on perfect state information not
  available in the real world.
- *Known Object Geometry:* The system requires access to the complete 3D object
  mesh at test time to compute a shape embedding. It is not a vision-based
  system that operates on sensor data like images.
- *No Environmental Awareness:* The humanoid operates without awareness of its
  environment, aside from the target object. During training, a support table
  for the object is programmatically removed after a set time.

=== Prerequisite Knowledge

- *Luo et al., 2023, "Universal humanoid motion representations for
  physics-based control" [41]:* This paper introduces *PULSE*, the universal
  motion representation that Omnigrasp's *PULSE-X* directly extends.
  Understanding PULSE is essential to comprehending the action space and
  hierarchical control structure used in this work.
- *Schulman et al., 2017, "Proximal policy optimization algorithms" [67]:* A
  foundational understanding of the PPO algorithm is necessary to understand the
  training mechanism for the Omnigrasp policy.

== Problem Formulation

The paper formulates the task of grasping and transporting diverse objects as a
*goal-conditioned Reinforcement Learning (RL)* problem. The learning process is
modeled as a Markov Decision Process (MDP), defined by the tuple
$cal(M)=angle.l S,A,T,cal(R),gamma angle.r$.

- *State Space ($S$)*: The state $s_t in S$ at any time $t$ consists of two
  components: the humanoid's proprioception ($s_t^P$) and a goal state
  ($s_t^g$).
  - *Proprioception ($s_t^P$)*: This is defined as
    $s_t^P eq.delta (q_t, dot(q)_t, c_t)$, containing the humanoid's pose $q_t$,
    velocity $dot(q)_t$, and hand contact forces $c_t$.
    - The pose $q_t = (theta_t, p_t)$ includes joint rotations
      $theta_t in RR^(J times 6)$ and positions $p_t in RR^(J times 3)$ for all
      $J$ links.
    - The velocity $dot(q)_t = (omega_t, v_t)$ includes angular
      $omega_t in RR^(J times 3)$ and linear $v_t in RR^(J times 3)$ velocities.
  - *Goal State ($s_t^g$)*: This provides the policy with information about the
    object and the desired trajectory. It is defined as:

$
  s_t^g eq.delta(
    sigma^"obj", hat(q)_(t + 1)^"obj" plus.circle q_t^"obj", hat(v)_(t + 1 : phi.alt)^"obj" - v_t^"obj", p_t^"obj" - p_t^"hand",
  ) quad"Eq. 2"
$

where:
- $sigma^"obj" in RR^512$ is a latent code representing the object's shape,
  computed using a Basis Point Set (BPS).
- $hat(q)_(t+1)^"obj" plus.circle q_t^"obj"$ is the difference between the next
  reference object pose and the current object pose.
- $hat(v)_(t+1:phi)^"obj" - v_t^"obj"$ is the difference between future
  reference velocities and the current object velocity over a horizon $phi$.
- $p_t^"obj" - p_t^"hand"$ is the vector difference between the object's
  position and each hand joint's position.

- *Action Space ($A$)*: The framework uses a hierarchical action space. The
  high-level policy $pi_"Omnigrasp"$ operates in a low-dimensional latent space,
  and a pre-trained decoder maps this to high-dimensional joint actuations.
  - The final low-level action $a_t$ specifies the target for a
    Proportional-Derivative (PD) controller for each of the 51 actuated joints,
    resulting in an action space of $a_t in RR^(51 times 3)$.
  - The relationship between the policy's output ($z_t^"omnigrasp"$) and the
    final action ($a_t$) is:

$
  a_t = cal(D)_"PULSE-X" (pi_"Omnigrasp" (s_t) + mu_t^p) quad"Eq. 3"
$

where $pi_"Omnigrasp"(s_t)$ is the residual action from the policy, $mu_t^p$ is
the mean action from a pre-trained motion prior $cal(P)_"PULSE-X"$, and
$cal(D)_"PULSE-X"$ is the pre-trained action decoder. The policy's direct output
is a latent vector $z_t^"omnigrasp" in RR^48$.

- *Reward Function ($cal(R)$)*: A stepwise reward function
  $r_t = cal(R)(s_t^P, s_t^g)$ is designed to guide the policy through stages of
  reaching, grasping, and transporting without needing paired full-body motion
  data.

$
  r_t^"omnigrasp" =
  cases(
    r_t^"approach" & "if "norm(hat(p)^"pre-grasp" - p_t^"hand")_2 > 0 . 2 " and " t < lambda, r_t^"pre-grasp" & "if "norm(hat(p)^"pre-grasp" - p_t^"hand")_2 <= 0 . 2 " and " t < lambda, r_t^"obj" & "if " t >= lambda,
  )
  quad"Eq. 4"
$

where $lambda=1.5s$ is a time threshold for grasping to occur.
- *Approach Reward ($r_t^"approach"$)*: Encourages the hands to get closer to
  the pre-grasp position.
- *Pre-grasp Reward ($r_t^"pre-grasp"$)*: A more precise reward that encourages
  matching the hand's position and orientation to the synthetic pre-grasp pose
  once close enough.
- *Object Trajectory Reward ($r_t^"obj"$)*: This reward is active after the
  grasping phase and encourages the humanoid to hold the object and follow the
  reference trajectory. It is defined as:

$
  r_t^"obj" = (w_"op" exp(-100norm(hat(p)_t^"obj" - p_t^"obj")_2^2) + w_"or" exp(-100norm(hat(theta)_t^"obj" - theta_t^"obj")_2^2) + ...) dot.op II C + II C dot.op w_c quad ( "Eq. 5, simplified" )
$

This reward measures the difference between the current and reference object
pose and is gated by an indicator function $II (C )$ that is true only if the
humanoid's hands are in contact with the object.

- *Objective*: The goal is to train the policy $pi_"Omnigrasp"$ to maximize the
  expected discounted sum of future rewards, $EE[sum_(t=1)^T gamma^(t-1)r_t]$,
  using the Proximal Policy Optimization (PPO) algorithm.

== Pipeline

The project is implemented in two primary stages: first, acquiring a universal
motion model, and second, training the grasping policy using that model.

=== Acquiring the PULSE-X Motion Representation

This stage produces a robust, low-dimensional action space for the main task.
- *Step 1.1: Data Augmentation*
  - *Description*: Existing large-scale body motion datasets (AMASS) are
    augmented with articulated finger motions from hand-specific datasets (GRAB,
    Re:InterHand) to create a comprehensive "dexterous" motion dataset.
  - *Input*: Body motion sequences from AMASS; hand motion sequences from GRAB
    and Re:InterHand.
  - *Output*: The "dexterous AMASS" dataset, containing full-body motion
    sequences that include realistic finger articulations.

- *Step 1.2: Training the PHC-X Motion Imitator*
  - *Description*: A policy, PHC-X, is trained via RL to accurately mimic the
    motions in the dexterous AMASS dataset. This policy learns the low-level
    motor skills to produce a wide variety of human-like movements.
  - *Input*: The dexterous AMASS dataset.
  - *Output*: A trained PHC-X policy capable of imitating reference motions in
    the physics simulator.

- *Step 1.3: Distillation into PULSE-X*
  - *Description*: The skills learned by the PHC-X imitator are distilled into a
    compact, VAE-like latent representation called PULSE-X. This is done via an
    online distillation process where the VAE is trained to reconstruct the
    actions of the expert PHC-X policy.
  - *Input*: The trained PHC-X policy.
  - *Output*: The frozen components of PULSE-X:
    - *Prior ($cal(P)_"PULSE-X"$)*: A network that takes proprioception $s_t^P$
      and outputs a distribution over the latent space (mean $mu_t^p in RR^48$
      and variance).
    - *Decoder ($cal(D)_"PULSE-X"$)*: A network that takes proprioception
      $s_t^P$ and a latent code $z_t in RR^48$ and outputs a low-level PD target
      action $a_t in RR^(51 times 3)$.

=== Training the Omnigrasp Policy

This stage uses the pre-trained PULSE-X components to learn the grasping and
manipulation task. The pipeline follows Algorithm 1.

- *Step 2.1: Episode Initialization*
  - *Description*: At the start of each training episode, an object, its initial
    state, a synthetic pre-grasp pose, and a reference trajectory are randomly
    sampled.
  - *Input*: A dataset of object meshes $hat(O)$ and a 3D trajectory generator
    $cal(T)^3D$.
  - *Output*: Initial state for the humanoid and object, and a target trajectory
    $hat(q)_(1:T)^"obj"$ for the episode.

- *Step 2.2: State Formulation*
  - *Description*: The goal state $s_t^g$ is constructed for the current
    timestep by combining the object's shape encoding, its current state, and
    the future reference trajectory, as defined in *Equation (2)*.
  - *Input*: Humanoid proprioception $s_t^P$, current object state $q_t^"obj"$,
    and the reference trajectory.
  - *Output*: The full state vector $s_t = (s_t^P, s_t^g)$ for the policy.

- *Step 2.3: Hierarchical Action Generation*
  - *Description*: The policy computes a latent action, which is combined with
    the output of the motion prior and passed to the motion decoder to generate
    the final joint commands, as described in *Equation (3)*.
  - *Input*: The full state $s_t$, the Omnigrasp policy $pi_"Omnigrasp"$, and
    the frozen PULSE-X prior $cal(P)_"PULSE-X"$ and decoder $cal(D)_"PULSE-X"$.
  - *Process & Intermediate Outputs*:
    1. The Omnigrasp policy outputs a residual latent action:
      $z_t^"omnigrasp" in RR^48$.
    2. The PULSE-X prior outputs the prior's mean: $mu_t^p in RR^48$.
    3. The final latent code is computed: $z_t = z_t^"omnigrasp" + mu_t^p$.
  - *Output*: A low-level PD target action $a_t in RR^(51 times 3)$ from the
    PULSE-X decoder.

- *Step 2.4: Physics Simulation*
  - *Description*: The generated action $a_t$ is applied to the humanoid in the
    Isaac Gym simulator, which computes the physical evolution of the system.
  - *Input*: The current system state $s_t$ and the action $a_t$.
  - *Output*: The next system state $s_(t+1)$.

- *Step 2.5: Reward Calculation*
  - *Description*: A scalar reward $r_t$ is computed based on the outcome of the
    action, using the stepwise function from *Equation (4)* and *Equation (5)*.
    The reward logic dynamically shifts from guiding the humanoid to approach
    the object, to matching a pre-grasp, to following the object trajectory.
  - *Input*: The resulting state $s_(t+1)$, the pre-grasp pose, and the
    reference trajectory.
  - *Output*: A scalar reward $r_t$.

- *Step 2.6: Policy Update*
  - *Description*: The experience tuple $(s_t, z_t^"omnigrasp", r_t, s_(t+1))$
    is stored in a memory buffer. The PPO algorithm uses batches of these
    experiences to update the neural network weights of the Omnigrasp policy.
    This step includes a hard-negative mining curriculum, where objects that the
    policy frequently fails to grasp are sampled more often during training.
  - *Input*: A buffer of collected experiences.
  - *Output*: The updated Omnigrasp policy $pi_"Omnigrasp"$.

== Discussion

Here is a detailed outline of the main questions the paper aimed to answer in
its results, discussion, and limitations sections.

=== How does Omnigrasp perform against other state-of-the-art methods?
This question assesses the overall effectiveness of the proposed framework on a
standardized benchmark for full-body object manipulation.

- *Experiment Design*:
  - *Dataset and Task*: The *GRAB dataset* was used, which contains paired
    full-body and object motions. The evaluation was performed on two splits:
    *cross-object* (testing on 5 unseen objects) and *cross-subject* (testing on
    motions from an unseen person).
  - *Baselines*: The method was compared against four baselines:
    1. *Braun et al. [6]*: The previous state-of-the-art method for this task.
    2. *PPO-10B*: A version of the policy trained with standard reinforcement
      learning for a month (~10 billion samples) *without* the PULSE-X motion
      prior to show the benefit of the prior.
    3. *PHC*: A pure motion imitator that attempts to directly follow the
      ground-truth kinematic motion, highlighting the difficulty of direct
      imitation due to physical errors.
    4. *AMP*: An alternative RL method that uses an adversarial motion prior.

- *Results and Metrics*:
  - *Metrics*: Performance was measured using grasp success rate
    (*$"Succ"_"grasp"$*), full trajectory success rate (*$"Succ"_"traj"$*),
    percentage of trajectory targets reached (*TTR*), and errors in position
    (*$E_"pos"$*) and rotation (*$E_"rot"$*).
  - *Results*: Omnigrasp significantly outperformed all baselines on all
    metrics. For the cross-object task, it achieved a *100% grasp success rate*
    and *94.1% trajectory success rate*, compared to 46.6% grasp success for the
    previous SOTA. The baselines without a strong, hierarchical motion prior
    (PPO-10B and AMP) performed poorly.

- *Significance and Limitations*:
  - *Significance*: The results demonstrate that using a universal, pre-trained
    motion representation (PULSE-X) as the action space is the key to solving
    this complex control problem. It drastically improves sample efficiency and
    performance over learning from scratch.
  - *Limitations*: There is a notable gap between the grasp success rate and the
    full trajectory success rate, indicating that the humanoid can successfully
    pick up an object but may drop it while following a complex path.

=== Can the method scale to a large number of diverse objects and generalize?
This question tests the framework's ability to handle variety and its robustness
to objects it has never been trained on.

- *Experiment Design*:
  - *Dataset and Task*: The *OakInk dataset*, with over 1,700 diverse objects,
    was used for a large-scale lifting and holding task.
  - *Conditions*: Three training scenarios were tested to measure
    generalization:
    1. *Train on OakInk*: Assesses scalability on a large, diverse dataset.
    2. *Train on GRAB, test on OakInk*: A cross-dataset test to see if skills
      learned on 50 objects can generalize to over 1,000 completely different
      ones.
    3. *Train on GRAB + OakInk*: Assesses the benefit of combined training data.

- *Results and Metrics*:
  - *Metrics*: The same success rates and error metrics were used.
  - *Results*: The policy successfully learned to pick up *1,272 out of 1,330
    training objects*. Strikingly, the policy trained *only* on the 50 objects
    from GRAB achieved a high success rate on OakInk, demonstrating powerful
    generalization. The policy trained on both datasets performed best, as it
    learned bi-manual manipulation from GRAB which was useful for larger objects
    in OakInk.

- *Significance and Limitations*:
  - *Significance*: The framework is highly scalable and learns a robust
    grasping policy that generalizes well to new object shapes, rather than just
    memorizing grasps for specific objects.
  - *Limitations*: The primary failure cases were objects that were physically
    too large or too small for the humanoid to establish a stable grasp. The
    hard-negative mining process was also strained by the large number of
    objects.

=== Which components of the framework are most critical for success?
This series of ablation studies dissects the method to identify the contribution
of each individual component.

- *Experiment Design*:
  - *Task*: The GRAB cross-object test was used as the benchmark. The full
    Omnigrasp model was compared against several degraded versions, each with a
    single key component removed:
    - *Without PULSE-X*: Training directly in the high-dimensional joint
      actuation space.
    - *Without pre-grasp reward*: Removing the pre-grasp guidance from the
      reward function.
    - *Without Dexterous AMASS data*: Training the PULSE-X motion prior on a
      dataset without augmented hand motions.
    - *Without object randomization and hard-negative mining*: Removing
      curriculum strategies that aid robustness.

- *Results and Metrics*:
  - *Metrics*: Primarily trajectory success rate ($"Succ"_"traj"$).
  - *Results*: Removing *PULSE-X* was catastrophic, causing a drop in success
    from 94.1% to 33.6% and resulting in unnatural, jerky motions. Removing the
    *pre-grasp reward* also significantly harmed performance. Using a PULSE-X
    model without dexterous hand data allowed the policy to grasp objects but it
    failed during trajectory following, showing the prior lacked the skill of
    "moving while holding".

- *Significance and Limitations*:
  - *Significance*: The ablations scientifically validate the paper's core
    claims. The universal motion prior is the single most important component,
    and the pre-grasp reward and dexterous training data are also crucial for
    achieving high performance.
  - *Limitations*: The study did not explore potential interactions between
    components (e.g., if a different reward scheme could compensate for a less
    powerful motion prior).

=== What kind of behaviors are learned and how robust are they?
This question provides a qualitative and quantitative analysis of the learned
policy's emergent behaviors and its resilience to noise.

- *Experiment Design*:
  - *Behavior Analysis*: The authors visualized the grasping strategies the
    policy used for various objects to qualitatively assess the diversity and
    plausibility of the learned motions.
  - *Robustness Test*: Uniform random noise was added to the policy's inputs
    during inference to simulate the noisy sensor data that a real-world robot
    might encounter.

- *Results and Metrics*:
  - *Metrics*: Qualitative observation for behavior; success rates and error
    metrics for the robustness test.
  - *Results*: The policy was observed to learn a *diverse set of grasping
    strategies*, intelligently switching between one-handed, two-handed, and
    even non-prehensile (scooping) motions based on the object's shape. Under
    noisy conditions, the policy's performance degraded only slightly (e.g.,
    trajectory success dropped from 94.1% to 91.4%), demonstrating strong
    robustness.

- *Significance and Limitations*:
  - *Significance*: The policy learns flexible and physically realistic
    behaviors, not just a single memorized motion. Its robustness to noise is a
    promising first step towards potential sim-to-real transfer.
  - *Limitations*: The authors explicitly state the system is not ready for
    real-world deployment and that this was a preliminary test. True sim-to-real
    transfer would require additional techniques like domain randomization.

=== What are the remaining limitations and directions for future work?
This question reflects on the project's unsolved challenges.

- *Identified Limitations*:
  - *In-hand Manipulation*: The system cannot perform precise manipulations like
    reorienting an object within the hand.
  - *Rotation Control*: Control of object rotation is less accurate than
    position control.
  - *Grasp Specificity*: It's not possible to instruct the humanoid to use a
    *specific type* of grasp (e.g., "use a power grip").
  - *Success Rate*: While high, the trajectory following success rate can still
    be improved to prevent dropping objects.

- *Future Work*:
  - *Improve Motion Representation*: Explore separating the hand and body
    representations for more nuanced control.
  - *Improve Object Representation*: Move towards vision-based systems that do
    not require a ground-truth object mesh.
  - *Improve Grasping Diversity*: Address the limitations above to enable a
    wider and more specific range of manipulation skills.

