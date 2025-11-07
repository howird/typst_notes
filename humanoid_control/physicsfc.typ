= PhysicsFC

== Overview

This paper introduces *PhysicsFC*, a hierarchical control framework for
physically simulated football characters. It enables a user to control an agent
to perform a variety of skills—moving, trapping, dribbling, and kicking—and
transition between them seamlessly in a physics-based environment.

=== Challenges and Approaches

- *Challenge: Realistic and Agile Motion*
  - Existing football games often rely on kinematic animation, which can produce
    artifacts like foot-sliding and unrealistic movements. The goal is to create
    more realistic motions by controlling a character where both the agent and
    the ball are fully physics-simulated.
  - *Approach:* The paper proposes a hierarchical framework. A low-level policy,
    based on a pre-existing motion embedding model, is trained to reproduce a
    wide range of motions from a motion capture dataset. High-level,
    skill-specific policies are then trained to generate latent variables that
    command this low-level policy to perform specific football skills based on
    user input.
  - *Hypothesis:* By building on a foundation of a physics-based motion
    embedding model, the controller can generate a wide variety of natural and
    physically plausible football movements, avoiding the artifacts of kinematic
    systems.
  - *Alternative Solutions:* Continuing to use kinematic animation for character
    movements, as is common in popular football games.

- *Challenge: Smooth and Responsive Skill Transitions*
  - In football, players must switch between actions instantly (e.g., trapping a
    pass and immediately starting to dribble) to be effective.
  - *Approach:* The paper introduces *Skill Transition-Based Initialization
    (STI)*. When training a skill policy (e.g., Dribble), episodes are
    initialized using states sampled from simulations of preceding skill
    policies (e.g., Trap or Move), as defined by a finite state machine.
  - *Hypothesis:* By training a policy to start from states where a transition
    would realistically occur, it learns to handle the specific dynamics of that
    transition, resulting in smoother and more agile gameplay.
  - *Alternative Solutions:* Initializing training from a standard rest pose or
    only using the terminal states of a previous policy's execution, which is
    less suitable for the dynamic, user-driven transitions required in football.

- *Challenge: Diverse and Natural Non-Ball-Related Movement*
  - When trained only on a task-based reward (e.g., reach a target velocity), a
    policy might learn unnatural movements (e.g., an odd-looking sideways run)
    because it is the most direct way to maximize the reward, even if more
    natural motions exist in the training data.
  - *Approach:* The paper proposes *Data-Embedded Goal-Conditioned Latent
    Guidance (DEGCL)* for the Move policy. This method supplements the standard
    task reward with a "latent similarity" reward. It encourages the policy's
    output to be similar to reference latent variables extracted from the motion
    capture data that correspond to the desired goal (e.g., making a "sideways
    walk" goal produce a latent similar to an actual "sideways walk" motion
    clip).
  - *Hypothesis:* By explicitly guiding the policy to use motions from the
    dataset that are appropriate for a given goal, the character will learn to
    perform a wide range of movements that are not only effective but also
    natural and diverse.
  - *Alternative Solutions:* Training the Move policy using only a task reward
    that measures the difference between the character's current and target
    state, which can lead to less natural motions.

- *Challenge: Mastering Skill-Specific Nuances*
  - Each football skill has unique objectives. Dribbling requires keeping the
    ball close while running , trapping requires absorbing a ball's momentum ,
    and kicking requires precise velocity control.
  - *Approach:* The paper designs tailored reward functions and initialization
    strategies for each skill policy.
    - *Dribble:* The reward function encourages matching a target velocity,
      keeping the ball close to the character's root, and moving the character
      towards the ball.
    - *Trap:* A two-phase reward structure is used (pre- and post-collision) to
      first guide a body part to the ball and then to absorb the ball's
      momentum. Initialization uses projectile dynamics to ensure the ball is
      catchable during training.
    - *Kick:* The reward is focused on matching the ball's velocity to the
      target kick velocity immediately after impact.
  - *Hypothesis:* Custom-designing the training process (rewards,
    initialization) for each specific skill is critical for the policy to learn
    the fine-grained control necessary to perform that skill effectively.

=== Proposed System: PhysicsFC

- *Component:* A hierarchical controller featuring a finite state machine
  (*PhysicsFC FSM*) that manages four distinct, high-level skill policies
  (Dribble, Trap, Move, Kick). These policies operate on top of a shared,
  pre-trained low-level policy.
- *Inputs:*
  - *User Commands:* Gamepad inputs that either define the goal for the current
    skill policy (e.g., target movement velocity for the Move policy, target
    kick velocity for the Kick policy) or trigger a state transition in the FSM
    (e.g., a "kick start command").
  - *Character and Ball State:* The physical state of the character and ball in
    the simulation (e.g., positions, velocities), which are used as input to the
    policies and can also trigger automatic FSM transitions.
- *Outputs:*
  - *Low-Level:* Each high-level policy outputs a latent vector `z`.
  - *System-Level:* This latent vector is fed to the low-level policy, which
    generates physics-based character actions, resulting in a character
    performing football skills in the simulated environment.

=== Dependencies

- *Pre-trained Model:* *CALM (Conditional Adversarial Latent Models)* is used as
  the foundational physics-based motion embedding model, providing the
  pre-trained low-level policy and encoder.
- *Dataset:* A commercial football motion capture dataset from the Unity Asset
  Store, containing 90 clips of various football movements, was used to train
  the low-level policy.
- *Environment:* The *Isaac Gym* physics simulation engine is used for all
  training and evaluation.

=== Assumptions

- The physics simulation is simplified and does not model the *Magnus effect*,
  which influences the curved trajectory of a spinning ball in flight. This
  limitation is acknowledged as affecting the realism of kicks and long passes.
- The analytical calculation for the ball's trajectory during the initialization
  of the Trap policy assumes motion is only influenced by gravity, ignoring
  factors like air resistance (beyond simple damping) and spin.

=== Prerequisite Knowledge

- *CALM [Tessler et al. 2023]:* Understanding this paper is crucial, as
  PhysicsFC directly uses it as its low-level controller and motion embedding
  model. The concept of using a latent vector `z` to control a physics-based
  character originates from this line of work.
- *DeepMimic [Peng et al. 2018]:* This paper is a foundational work for learning
  physics-based character skills from motion capture data using deep
  reinforcement learning. It provides essential context for the general approach
  taken by CALM and, by extension, PhysicsFC.

== Problem Formulation

The project's objective is to train a set of high-level, skill-specific
policies, denoted as $pi_"skill"(z | s, g)$, using reinforcement learning. These
policies take a character and ball state $s$ and a goal vector $g$ as input to
produce a latent vector $z in RR^64$. This latent vector $z$ serves as a command
for a shared, pre-trained low-level policy, $pi_"low"(a | s, z)$, which in turn
generates the physical actions $a$ for the character in the simulation. The
formulation for each skill is defined by its unique goal space and reward
function.

==== Dribble Policy ($pi_"dribble"$)
The goal is to move the ball at a user-specified target velocity while keeping
it close to the character's feet.

- *Goal Input ($g$):* The target dribble velocity on the horizontal plane,
  $hat(v)_t^"drib" in RR^2$.
- *Reward Function ($r_t^"drib"$):* A weighted sum of three components.

$
  r_t^"drib" = 0 . 6 r_t^"ball_vel" + 0 . 2 r_t^"ball_root_pos" + 0 . 2 r_t^"root_vel" quad(E q . 1)
$

- *Ball Velocity Reward ($r_t^"ball_vel"$):* Encourages the ball's horizontal
  velocity ($v_t^( "ball"(2) )$) to match the target dribble velocity
  ($hat(v)_t^"drib"$).

$
  r_t^"ball_vel" = exp(
    -10(((norm(hat(v)_t^"drib" - v_t^("ball"(2))))/(norm(hat(v)_t^"drib") + epsilon.alt))^2 + 0 . 1((norm(hat(v)_t^"drib") -norm(v_t^("ball"(2))))/(norm(hat(v)_t^"drib") + epsilon.alt))^2)
  ) quad(E q . 2)
$

- *Ball-Root Position Reward ($r_t^"ball_root_pos"$):* Encourages the character
  to keep the ball close by minimizing the horizontal distance between the
  character's root position ($x_t^""root"(2)"$) and the ball's position
  ($x_t^( "ball"(2) )$).

$
  r_t^"ball_root_pos" = exp(-10norm(x_t^("ball"(2)) - x_t^("root"(2)))^2) quad(E q . 3)
$

- *Root Velocity Reward ($r_t^"root_vel"$):* Guides the character's root
  ($v_t^( "root"(2) )$) to move towards the ball's current position at the
  target speed.

$
  r_t^"root_vel" = exp(
    -10 (norm(norm(hat(v)_t^"drib")d_t^(r 2 b) - v_t^("root"(2)))/( norm(hat(v)_t^"drib") + epsilon.alt ))^2
    + 0.1 ((norm(hat(v)_t^"drib") - norm(v_t^("root"(2))))/( norm(hat(v)_t^"drib") + epsilon.alt))^2
  )
  ) quad(E q . 4)
$

where $d_t^"r2b"$ is the unit vector from the character's root to the ball.

==== Trap Policy ($pi_"trap"$)
The goal is to control an incoming ball by making contact with a specified body
part and absorbing its momentum.

- *Goal Input ($g$):* A one-hot vector representing the target body part (e.g.,
  head, foot) to be used for the trap.
- *Reward Function ($r_t^"trap"$):* A two-phase reward that changes based on the
  time of the first collision, $t_c$.

$
  r_t^"trap" = cases(r_t^"before" & "if " t < t_c, r_t^"after" & "otherwise") quad(E q . 5)
$

- *Pre-Collision Reward ($r_t^"before"$):* Guides the specified body part
  ($x_t^"body"$) to the ball's 3D position ($x_t^"ball"(3)$).

$
  r_t^"before" = exp(-10norm(x_t^("ball"(3)) - x_t^"body")^2) quad(E q . 6)
$

- *Post-Collision Reward ($r_t^"after"$):* Encourages the ball's 3D velocity
  ($v_t^"ball"(3)$) to match the character's root velocity ($v_t^"root"(3)$) to
  stabilize the ball.

$
  r_t^"after" = exp(-10norm(v_t^("ball"(3)) - v_t^("root"(3)))^2) quad(E q . 7)
$

==== Move Policy ($pi_"move"$)
The goal is to enable the character to move at a target velocity and face a
target direction, using natural motions from the training data.

- *Goal Input ($g$):* The target movement velocity $hat(v)_t^"move" in RR^2$ and
  the target facing direction unit vector $hat(d)_t^"face" in RR^2$.
- *Reward Function ($r_t^"move"$):* A conditional reward for implementing
  Data-Embedded Goal-Conditioned Latent Guidance (DEGCL).

$
  r_t^"move" = cases(
    0 . 5 r_t^"mv_task" + 0 . 5 r_t^"lt_sim" & "for DEGCL episodes", r_t^"mv_task" & "for General episodes",
  ) quad(E q . 8)
$

- *Task Reward ($r_t^"mv_task"$):* A weighted sum of velocity and direction
  rewards.

$
  r_t^"mv_task" = 0.7 r_t^"vel" + 0.3 r_t^"dir" quad (E q. 9)
$

- $r_t^"vel"$ encourages matching the target velocity:

$
  r_t^"vel" = exp(
    -0 . 25(
      ( (norm(v_t^"target" - v_t^("root"(2))))/(norm(v_t^"target") + epsilon.alt))^2 +
      0.1(
        (norm(v_t^"target") -norm(v_t^("root"(2))))/(norm(v_t^"target") + epsilon.alt))^2
    )
  ) quad(E q . 10)
$

- $r_t^"dir"$ encourages matching the target facing direction:

$
  r_t^"dir" = d_t^"target" dot.op d_t^"root" quad(E q . 11)
$

- *Latent Similarity Reward ($r_t^"lt_sim"$):* The cosine similarity between the
  policy's output latent $z_t$ and a reference latent $overline(z)_t$ from the
  DEGCL buffer.

$
  r_t^"lt_sim" = (z_t dot.op macron(z)_t)/(norm(z_t)norm(macron(z)_t)) quad(E q . 12)
$

==== Kick Policy ($pi_"kick"$)
The goal is to kick the ball such that its initial velocity matches a
user-specified target velocity.

- *Goal Input ($g$):* The target kick velocity $hat(v)_t^"kick" in RR^3$.
- *Reward Function ($r_t^"kick"$):* Evaluated only for a short duration after
  collision, this reward encourages the ball's 3D velocity ($v_t^"ball"(3)$) to
  match the target kick velocity ($hat(v)_t^"kick"$).

$
  r_t^"kick" = exp(
    -10(
      (norm(hat(v)_t^"kick" - v_t^("ball"(3))))/(norm(hat(v)_t^"kick") + epsilon.alt))^2
  ) quad(E q . 13)
$

== Pipeline

The project is implemented in a sequential, multi-stage pipeline where
dependencies from earlier stages are used to train later ones.

=== Low-Level Policy and Encoder Training

- *Description:* This initial stage involves training the foundational
  physics-based motion embedding model, CALM. This model learns a latent space
  of motions and a low-level policy capable of reproducing those motions from a
  dataset. This stage does not involve any of the high-level football skills.
- *Inputs:*
  - *Football Motion Dataset:* A collection of 90 motion capture clips.
  - *Character State:* The physical state of the character (e.g., joint
    positions, velocities).
- *Outputs:*
  - *Trained Low-Level Policy ($pi_"low"$):* A neural network that maps a
    character state and a latent vector to low-level physical actions.
  - *Trained Encoder:* A neural network that maps motion sequences from the
    dataset into their corresponding latent vectors.
  - *Latent Vector ($z$):* The output of the Encoder and input to the low-level
    policy.
    - *Shape:* `[Number of Environments, 64]`.

=== Training Data Buffer Construction

- *Description:* Before training the high-level policies, two types of data
  buffers are constructed by running simulations with already trained policies.
  This stage generates the necessary data for DEGCL and STI.
- *Sub-stage 2a: DEGCL Buffer Construction*
  - *Description:* This buffer stores reference goal-latent pairs extracted from
    the motion dataset and is used exclusively for training the Move policy.
  - *Input:* 16 selected motion clips and the trained Encoder from Stage 1.
  - *Process:* For each clip, the average movement goal (velocity and direction)
    is computed. The trained Encoder generates the corresponding reference
    latent vector for that motion.
  - *Output:* *DEGCL Buffer:* A collection of
    `(reference_goal, reference_latent)` pairs.
- *Sub-stage 2b: STI Buffer Construction*
  - *Description:* Skill Transition-Based Initialization (STI) buffers store
    character and ball states from simulated moments where a transition between
    skills is likely to occur. These are used to initialize training episodes
    for subsequent policies to ensure smooth transitions.
  - *Input:* A previously trained skill policy (e.g., `π_move`) and randomly
    sampled goals.
  - *Process:* The policy is run in the physics simulator, and states are
    recorded at specific moments (e.g., at collision for Trap, at any time for
    Dribble).
  - *Output:* *STI Buffers:* Collections of saved simulation states. The paper
    creates a Move STI Buffer, Trap STI Buffer, and Dribble STI Buffer, each
    containing 50,000 states.

=== High-Level Skill Policy Training

- *Description:* The four high-level skill policies (Move, Trap, Dribble, Kick)
  are trained sequentially using the Proximal Policy Optimization (PPO)
  algorithm. The low-level policy from Stage 1 is frozen and used to execute
  actions in the simulation.
- *Training Order:* Move $->$ Trap $->$ Dribble $->$ Kick.
- *Inputs (for each policy):*
  - *Character State:* The character's physical state.
    - *Shape:* `[Number of Environments, 223]`.
  - *Ball State:* The ball's physical state.
    - *Shape:* `[Number of Environments, 13]`.
  - *Goal Input:* The specific goal for the policy being trained (e.g., target
    velocity).
  - *STI / DEGCL Buffers:* As required. For example, Dribble policy training
    uses the Move and Trap STI buffers for episode initialization.
- *Process (Training Loop):*
  1. *Initialize Episode:* An episode's initial state is sampled from the
    relevant STI buffer (or a rest pose for the Move policy).
  2. *Generate Latent:* The current high-level policy takes the state and goal,
    and outputs a latent vector $z$.
  3. *Simulate Step:* The frozen low-level policy uses $z$ to generate a
    physical action, and the simulator advances one step.
  4. *Calculate Reward:* The reward is calculated using the policy-specific
    formulation (e.g., *Eq. 1* for the Dribble policy).
  5. *Update Policy:* The PPO algorithm updates the high-level policy's weights
    based on the collected experience (state, action, reward).
- *Output (for each policy):* A trained high-level skill policy (e.g.,
  $pi_"dribble"$).

=== Runtime Control via PhysicsFC FSM

- *Description:* The four trained high-level policies are integrated into a
  *PhysicsFC Finite State Machine (FSM)*. This FSM manages the active policy and
  transitions between them based on user input and game context to create
  interactive gameplay.
- *Inputs:*
  - The four trained skill policies.
  - *User Gamepad Input:* Dynamically provides the goal for the currently active
    policy (e.g., left stick sets target dribble velocity) or triggers a state
    transition (e.g., a button press initiates a kick).
  - *Character and Ball State:* Used to evaluate automatic transition conditions
    (e.g., transitioning from Move to Dribble when the ball is close and
    approaching).
- *Process:*
  1. The FSM is in a current state, corresponding to an active policy (e.g.,
    `Dribble`).
  2. User input is parsed to provide the goal for that policy.
  3. The active policy generates a latent vector $z$, which the low-level policy
    uses to control the character.
  4. FSM transition conditions are checked every frame. If a condition is met
    (e.g., user presses kick button), the FSM switches to the new state
    (`Kick`), and its corresponding policy becomes active.
- *Output:* A real-time, interactive character controlled by the user,
  performing a variety of physics-based football skills.

== Discussion

The paper's evaluation sections are designed to validate the effectiveness of
each proposed component and skill policy through a series of quantitative
experiments and ablation studies.

=== How effective is the Dribble policy and its core components?

This question assesses whether the proposed reward structure and training setup
can produce a policy that keeps the ball close to the character's feet while
moving at a target velocity.

- *Experiments and Ablations:*
  - An ablation study was conducted comparing the full Dribble policy (`Ours`)
    to several versions, each with a key component removed: one of the three
    reward terms (`w/o rball_vel`, `w/o rball_root_pos`, `w/o rroot_vel`), the
    early termination condition (`w/o DistanceET`), Normalization by Target
    Speed (`w/o NTS`), or the custom football boot mesh (`w/BoxFoot`) .
  - A second experiment evaluated the policy's performance across a range of
    fixed target speeds from 1 m/s to 7 m/s .

- *Metrics Used:*
  - *Character-Ball Distance (CBD):* The average horizontal distance between the
    character's root and the ball.
  - *Foot-Ball Distance (FBD):* The average distance between the foot and ball
    at the moment of ground contact.
  - *Dribbling Goal Achievement Rate (DGAR):* The percentage of times the ball's
    velocity successfully matched the target velocity within a time limit.
  - *Character Speed (CS):* The average horizontal speed of the character.

- *Results and Significance:*
  - The full policy (`Ours`) performed best across all metrics, achieving the
    highest goal achievement rate (90.3%) and the smallest distances to the
    ball.
  - Removing any component resulted in significant performance degradation. Most
    notably, removing the early termination condition (`w/o DistanceET`) caused
    the policy to fail completely, demonstrating that preventing the character
    from losing the ball is critical for learning.
  - The `w/BoxFoot` model struggled to change directions, highlighting that the
    collision mesh shape is important for fine-grained control.
  - The policy was effective at matching speeds up to 5 m/s, after which its
    performance declined, reflecting the real-world difficulty of high-speed
    dribbling.

- *Limitations:*
  - The trained policy consistently learns to use only a single foot (e.g., the
    left foot) for dribbling, which reduces realism.

=== Can the Trap policy reliably control incoming balls?

This question evaluates the effectiveness of the two-phase reward structure and
the projectile-based initialization for training a character to intercept both
aerial (lob) and ground passes.

- *Experiments and Ablations:*
  - An ablation study compared the full Trap policy (`Ours`) to versions without
    the pre-collision reward (`w/o rbefore`), without the post-collision reward
    (`w/o rafter`), without the projectile-based initialization
    (`w/o ProjectileInit`), and without early termination for handball fouls
    (`w/o HandArmET`) .

- *Metrics Used:*
  - *Trapping Success Rate (TSR):* The percentage of lob passes successfully
    touched by the character before hitting the ground.
  - *Handball Ratio in Trapping Success (HRTS):* The percentage of successful
    traps that involved an illegal hand/arm touch.
  - *Relative Ball Speed Post-Trap (RBSPT):* The speed of the ball relative to
    the character immediately after contact; lower is better, indicating greater
    momentum absorption.

- *Results and Significance:*
  - `Ours` achieved the highest success rate (78.3%) and the lowest post-trap
    ball speed, indicating it was the most effective at both reaching and
    controlling the ball.
  - The `w/o ProjectileInit` model performed the worst, with a TSR of only
    21.1%. This shows that the proposed method of calculating the ball's initial
    state to ensure it lands near the character is crucial for the policy to
    learn successfully.
  - The `w/o HandArmET` model had a very high handball rate (20.7%), validating
    that this termination condition is necessary to enforce the rules of the
    game.

- *Limitations:*
  - The physics simulation does not account for the *Magnus effect*, meaning it
    cannot replicate the curved trajectories of spinning balls, which is a key
    factor in realistic passes and shots.

=== Does DEGCL enable the Move policy to learn natural motions?

This question investigates whether the proposed Data-Embedded Goal-Conditioned
Latent Guidance (DEGCL) method helps the Move policy learn to use diverse and
realistic motions from the training data, rather than unnatural but effective
ones.

- *Experiments and Ablations:*
  - An ablation study was performed on the Move policy, comparing the full
    version (`Ours`) against versions without DEGCL (`w/o DEGCL`), without each
    task reward term (`w/o rdir`, `w/o rvel`), and without NTS (`w/o NTS`) .

- *Metrics Used:*
  - *Moving Goal Achievement Rate (MGAR):* The percentage of times the character
    successfully matched the target velocity and facing direction.
  - *Goal Matching Latent Similarity (GMLS):* A measure of cosine similarity
    between the policy's output latent vector and the reference latent vector
    from the motion data for a given goal. A higher score means the generated
    motion is more stylistically similar to the reference data.

- *Results and Significance:*
  - The `w/o DEGCL` model achieved a slightly higher MGAR but had a much lower
    GMLS score. This demonstrates that while it could achieve its goals, it did
    so using unnatural motions that deviated from the training data.
  - `Ours`, with DEGCL, achieved the highest GMLS score while maintaining a high
    MGAR, indicating it successfully produced natural-looking motions (like
    proper backward and sideways walking) while still being effective at its
    task. This validates DEGCL as an effective method for improving motion
    quality.

- *Limitations:*
  - The paper's method for learning fall recovery, adopted from prior work,
    often leads to abrupt standing motions with high joint torques rather than
    realistic, human-like recovery motions.

=== How accurate is the Kick policy?

This question assesses the policy's ability to strike the ball with a
user-specified velocity, and it evaluates the importance of NTS and the foot
collision mesh.

- *Experiments and Ablations:*
  - The full Kick policy (`Ours`) was compared to a version without
    Normalization by Target Speed (`w/o NTS`) and a version using a simple
    box-shaped foot (`w/BoxFoot`) .

- *Metrics Used:*
  - *Kick Success Rate (KSR):* The percentage of attempts where the character
    successfully collides with the ball.
  - *Kick Direction Deviation (KDD):* The average angular difference between the
    actual and target kick direction.
  - *Kick Speed Deviation (KSD):* The average difference between the actual and
    target kick speed.

- *Results and Significance:*
  - Normalization by Target Speed (NTS) was found to be critical. The `w/o NTS`
    model failed on every single attempt (0% KSR), likely because the wide range
    of target kick speeds ($[5, 35~m/s]$) made it impossible to learn without
    this normalization.
  - `Ours` achieved a near-perfect KSR (99.9%) and was significantly more
    accurate in both direction and speed compared to the `w/BoxFoot` model. This
    confirms the importance of both NTS and a realistic collision mesh for
    precise skills like kicking.

- *Limitations:*
  - Similar to the Dribble policy, the Kick policy learns to rely on a single
    foot.
  - The lack of the Magnus effect in the simulation is a significant limitation
    for producing realistic curved shots.

=== How effective is Skill Transition-Based Initialization (STI)?

This is a central question of the paper, evaluating whether the proposed STI
method enables smoother, faster, and more reliable transitions between different
skills during gameplay.

- *Experiments and Ablations:*
  - Four key transitions from the FSM were evaluated: Trap-to-Dribble,
    Move-to-Dribble, Move-to-Trap, and Dribble-to-Kick. For each, the
    performance of the post-transition policy trained *with* STI (`Ours`) was
    compared to one trained *without* STI (`w/o STI`).

- *Metrics Used:*
  - A combination of the metrics from the individual skill evaluations, plus:
  - *Time to Achieve Dribbling Goal (TADG):* Time taken to achieve the dribble
    goal after a transition.
  - *Time to Kick (TTK):* Time taken from the kick command to the actual
    foot-ball collision.

- *Results and Significance:*
  - STI provided a dramatic improvement across all evaluated transitions.
    Policies trained with STI were able to achieve their goals significantly
    faster and more reliably.
  - The most striking result was in the *Dribble-to-Kick* transition. The `Ours`
    policy succeeded in 100% of attempts, while the `w/o STI` policy failed over
    83% of the time, as it had never been trained to kick from a dynamic,
    dribbling state.
  - These results strongly validate STI as a crucial method for creating a
    cohesive and responsive controller. It proves that training policies on
    states that are representative of actual gameplay transitions is essential
    for chaining skills together effectively.

- *Limitations:*
  - The paper does not identify limitations specific to the STI method itself;
    its effectiveness is bound by the quality of the individual skill policies
    it connects.
