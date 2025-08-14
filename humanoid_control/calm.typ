= CALM

Here is an overview of the paper "CALM: Conditional Adversarial Latent Models
for Directable Virtual Characters."

== Overview

=== Challenges and Solutions

This paper addresses three primary challenges in creating directable virtual
characters.

- *Challenge 1: Lack of Direct Control in Generative Models*
  - Existing unsupervised methods, like ASE, can generate a diverse set of motions
    but lack a straightforward way for a user to specify *which* particular motion
    to perform at a given time.
  - *CALM's Approach:* The core of the solution is the joint, end-to-end training of
    a motion encoder and a low-level policy using a *conditional discriminator*. The
    discriminator is conditioned on the latent code $z$ of a specific reference
    motion $M$, forcing the policy to generate a motion that matches the
    characteristics of $M$ to successfully fool the discriminator.
  - *Hypothesis:* By making the discriminator's task motion-specific (i.e., "is this
    a good example of motion $M$?") rather than general (i.e., "is this a human-like
    motion?"), the model will learn a meaningful and directable mapping between a
    motion and its latent representation.
  - *Alternatives Mentioned:*
    - *ASE (Adversarial Skill Embeddings):* Uses an information maximization objective
      to encourage diversity, but this results in a latent space that is difficult to
      control.
    - *VAE-based Models:* Rely on a reconstruction loss, which limits their ability to
      generalize or create novel transitions not explicitly present in the training
      data.
    - *Language-Supervised Models (PADL):* Require labeled data (text descriptions of
      motions) to learn control, whereas CALM is unsupervised and works with raw
      motion capture data.

- *Challenge 2: Controlling Motion Directionality*
  - The base generative policy can be instructed to perform an action like "walk,"
    but cannot be intuitively controlled to walk in a specific direction.
  - *CALM's Approach:* A *high-level policy* is trained on top of the frozen
    low-level policy in a phase called "Precision Training". This policy learns to
    select and provide latent codes $z_t$ to the low-level policy to steer the
    character in a desired direction while maintaining a specific motion style
    (e.g., crouch-walking). This is guided by a task reward that encourages moving
    in the correct direction while staying stylistically similar to a reference
    motion.
  - *Hypothesis:* A hierarchical approach, where a high-level controller learns to
    modulate a pre-trained, versatile low-level policy, can achieve nuanced control
    like directionality without having to retrain the fundamental motion generation
    model.

- *Challenge 3: Solving Complex, Sequential Tasks without Re-training*
  - Traditionally, solving complex tasks requires designing intricate, task-specific
    reward functions and retraining a model for every new task, which is inefficient
    and brittle.
  - *CALM's Approach:* At inference time, the system uses a simple, rule-based
    *Finite-State Machine (FSM)* to sequence skills and solve tasks without any
    further training. The FSM can either call the high-level policy for directional
    movement (e.g., "run towards the target") or directly command the low-level
    policy with a specific motion encoding (e.g., "kick").
  - *Hypothesis:* If the underlying policies provide a sufficiently robust and
    directable set of skills, complex behaviors can be composed using simple,
    classic programming tools like FSMs, completely bypassing the need for further
    reinforcement learning or reward design.

=== Proposed Component: CALM Framework

- *High-Level Description:* CALM is a three-phase framework for training a
  directable controller for a physically simulated character using unlabeled
  motion data. It learns a semantic representation of motions that allows a user
  to direct the character's behavior through high-level commands.
- *Inputs:*
  - *For Training:* An unlabeled dataset of diverse motion capture recordings.
  - *For Inference:* A sequence of high-level commands managed by an FSM, which
    provides either a target direction to the high-level policy or a specific motion
    encoding directly to the low-level policy.
- *Outputs:*
  - *From Training:* Three trained components: a motion encoder, a low-level policy,
    and a high-level policy.
  - *From Inference:* A sequence of low-level motor actions that drive the
    physically simulated character to perform the commanded, multi-step behaviors.

=== External Dependencies

To reproduce the method, the following non-novel components are required:

- *Dataset:* The "Reallusion [2022]" motion dataset, which contains 160 motion
  clips.
- *Simulator:* *Isaac Gym* is used for the high-performance, GPU-based physics
  simulation during training.
- *RL Algorithm:* *Proximal Policy Optimization (PPO)* is the specific RL
  algorithm used for policy training.
- *Underlying Concepts:* The framework is built directly upon the ideas and
  architectures of:
  - *ASE (Peng et al. 2022):* CALM is explicitly designed to fix the controllability
    issues of ASE.
  - *AMP (Peng et al. 2021):* This work provides the foundational adversarial motion
    priors approach.
  - *GAIL (Ho & Ermon 2016):* The fundamental adversarial imitation learning
    algorithm that underpins the entire approach.

=== Additional Context

- *Perspectives Missing from Abstract:*
  - The abstract does not mention the *three-phase structure* (Low-level Training,
    Precision Training, Inference), which is a core part of the CALM methodology.
  - It highlights control via "intuitive interfaces" but omits the key
    implementation detail that this is achieved with a non-learning, rule-based
    *Finite-State Machine (FSM)* at inference time.

- *Key Assumptions:*
  - *Data Sufficiency:* The model assumes the motion dataset is diverse and contains
    enough transitional information for the policy to learn to smoothly blend
    between different skills, even if direct transitions are not explicitly
    demonstrated.
  - *Task Decomposability:* The FSM-based inference assumes that complex tasks can
    be successfully broken down and scripted into a sequence of predefined states
    and transitions.
  - *Stable Physics:* The method assumes a consistent character morphology and
    environment. The authors note that the policy may fail on different terrains
    (e.g., stairs) and that visual artifacts can occur when the simulation
    character's geometry differs from the rendering character's geometry.

- *Recommended Prerequisite Reading:*
  - *Peng, X. B. et al. (2022).*ASE: Large-scale Reusable Adversarial Skill
    Embeddings for Physically Simulated Characters*.* Understanding ASE is critical,
    as CALM is a direct successor that aims to solve its primary limitations
    regarding control.
  - *Peng, X. B. et al. (2021).*AMP: Adversarial Motion Priors for Stylized
    Physics-Based Character Control*.* This paper introduces the adversarial
    imitation learning framework for physics-based characters that both ASE and CALM
    are built on.

== Problem Formulation

=== Overall Goal: Learning a Directable Motion Manifold

The primary objective is to learn a mapping from a large, unlabeled dataset of
motion capture clips, $cal(M)$, to a controllable, physics-based character. This
is achieved by jointly learning two key components:

- *A Motion Encoder ($E$):* This network maps a given motion clip $M in cal(M)$ to
  a low-dimensional latent vector $z$ in a latent space $cal(Z)$. The mapping is
  denoted as $z = E(M)$. This vector $z$ should capture the semantic essence of
  the motion (e.g., "walking," "kicking").

- *A Low-Level Control Policy ($pi$):* This is a stochastic policy, $pi(a_t | s_t, z)$,
  that takes the current character state $s_t$ and a latent code $z$ as input, and
  outputs an action $a_t$ for the physics simulator. The policy's goal is to
  generate a motion that matches the style encoded by $z$.

The overarching problem is to train $E$ and $pi$ such that for any motion $M$ from
the dataset, the policy $pi$ conditioned on $z=E(M)$ produces a behavior that is
perceptually similar to $M$.

=== Low-Level Training: Conditional Imitation Learning

The core of CALM is formulated as a conditional imitation learning problem,
optimized using an adversarial framework. The goal is to make the
state-transition distribution of the policy match that of the reference motion
data.

==== High-Level Objective

The objective is to minimize the Jensen-Shannon (JS) divergence between the
state-transition distribution of the policy, $d^{pi}(s,s'|z)$, and the
state-transition distribution of the reference motion, $d^{M}(hat(s),hat(s)')$.

$
  max_(pi, E) - EE_(M in cal(M)) [D_(J S)(d^pi (s, s'|z)|_(z = E(M))||d^M (hat(s), hat(s)'))]
$

where:
- $M in cal(M)$ is a motion clip sampled from the dataset.
- $z = E(M)$ is the latent code for that motion.
- $d^{pi}(s,s'|z)$ is the distribution of state transitions $(s, s')$ produced by
  the policy $pi$ when conditioned on $z$.
- $d^{M}(hat(s),hat(s)')$ is the distribution of state transitions from the
  ground-truth motion clip $M$.

This objective is intractable to optimize directly, so it's operationalized via
an adversarial setup.

==== Adversarial Formulation

A conditional discriminator, $D(s, s' | z)$, is introduced to distinguish
between policy-generated transitions and real transitions, given a latent code $z$.

- *Discriminator Objective:* The discriminator $D$ is trained to output 1 for real
  transitions and 0 for fake (policy-generated) ones. Its loss function, $cal(L)_D$,
  is:

  $
    cal(L)_D = - EE_(M in cal(M))(
    EE_(d^M (hat(s), hat(s)')) [log D(hat(s), hat(s)'|z)] + EE_(d^pi (s, s'|z)) [log(1 - D(s, s'|z))])
  $

The paper uses a more practical version (Equation 7) with a gradient penalty ($w_"gp"$)
and negative sampling to improve training stability.

- *Policy Objective:* The policy $pi$ is trained to fool the discriminator. It
  receives a reward based on the discriminator's output. The policy's objective $J$ is
  to maximize the expected discounted reward, where the reward is defined as:

  $
    r(s_t, s_(t + 1), z) = log(D(s_t, s_(t + 1)|z))
  $

This encourages the policy to produce state transitions that the discriminator
classifies as real. The full RL objective is:

$
  J = EE_(M in cal(M)) [EE_(p(tau|pi, z))(sum_t gamma^t r(s_t, s_(t + 1), z)) |_(z = E(M))]
$

==== Encoder Regularization

To ensure the latent space $cal(Z)$ is well-structured, two auxiliary losses are
applied to the encoder $E$:

- *Alignment Loss ($cal(L)_"align"$):* Encourages temporally adjacent
  (overlapping) sub-motions, $M$ and $M'$, to have similar latent codes.

  $
    cal(L)_"align" = EE_((M, M') ~ "overlapping") [||E(M) - E(M')||_2^2 ]
  $

- *Uniformity Loss ($cal(L)_"uniform"$):* Encourages the latent codes of all
  motions to be uniformly distributed on the unit hypersphere, preventing
  representational collapse.

  $
    cal(L)_"uniform" = log EE_((M, M') ~ "i.i.d.") [exp(-2||E(M) - E(M')||_2^2)]
  $

=== High-Level Control: Precision Training

After pre-training, the low-level policy $pi$ can generate motions of a certain
*style* (e.g., "run") but cannot control its *direction*. A high-level policy is
trained to provide this control.

- *Problem:* Train a high-level policy to select a sequence of latent codes $z_t$ that
  are fed to the frozen low-level policy $pi(a_t | s_t, z_t)$. This should guide
  the character to move in a desired direction $d_t^*$ while maintaining a desired
  motion style, represented by a reference encoding $hat(z)$.

- *Objective:* The high-level policy is trained with a reward function $r_"locomotion"$ that
  balances task completion and style adherence.

  $
    r_"locomotion" = exp(-0 . 25 lr(|lr(|d_t^* - dot(x)_t^"root"/(||dot(x)_t^"root"||)|)|)^2) + exp(-4||z_t - hat(z)||^2)
  $

where:
- The first term rewards the character for moving in the target direction $d_t^*$,
  where $dot(x)_t^"root"$ is the character's root velocity.
- The second term rewards the high-level policy for selecting latent codes $z_t$ that
  are close to the target style's encoding $hat(z)$.

== Pipeline

=== Low-Level Training (Generative Skill Learning)
The goal of this stage is to jointly train a motion *Encoder* and a *Low-Level
Policy* to generate a wide variety of motions from an unlabeled dataset. This
phase creates a generative model of behavior.

- *Inputs:*
  - *Motion Capture Dataset ($cal(M)$):* A large collection of unlabeled MoCap
    clips. These are processed into overlapping 2-second sub-motions. Each
    sub-motion consists of 60 frames of joint data, as the controller operates at
    30Hz.
  - *Physics Simulator:* An environment like Isaac Gym where the character
    interacts.

- *Components:*
  - *Encoder ($E$):* An MLP that maps a motion clip to a latent vector.
    - Input: Motion Clip $M in RR^( 60 times D_"pose" )$, where $D_"pose"$ is the
      dimension of a single pose.
    - Output: Latent Vector $z in RR^( 64 )$, projected onto a unit hypersphere.
  - *Low-Level Policy ($pi$):* An MLP that acts as the decoder, conditioned on a
    latent vector.
    - Input: State $s_t in RR^( 120 )$ and Latent Vector $z in RR^( 64 )$.
    - Output: Action $a_t in RR^{31}$, representing target joint rotations.
  - *Conditional Discriminator ($D$):* An MLP that distinguishes real from fake
    state transitions, conditioned on a latent vector.
    - Input: A sequence of 10 state transitions and a Latent Vector $z in RR^( 64 )$.
    - Output: A scalar value indicating realism (probability).

- *Process:*
  1. A random motion clip $M$ is sampled from the dataset $cal(M)$.
  2. The *Encoder* computes the corresponding latent vector: $z = E(M)$.
  3. The *Low-Level Policy*, conditioned on $z$, interacts with the physics simulator
    for a rollout of K steps to generate a sequence of states and actions ($s_0, a_0, s_1, ...$).
    To improve transition ability, the conditional motion $M$ (and thus $z$) is
    re-sampled at random timesteps during the rollout.
  4. The *Conditional Discriminator* is trained to distinguish between state
    transitions sampled from the real motion clip $M$ and transitions generated by
    the policy. Gradients are stopped from flowing from the discriminator to the
    encoder.
  5. The *Low-Level Policy* and the *Encoder* are updated together using PPO. The
    policy's reward is derived from the discriminator's output, encouraging it to
    fool the discriminator. The gradients from the policy update flow back to the
    encoder, optimizing the latent representation for the control task. The encoder
    is also updated using alignment and uniformity losses to structure the latent
    space.

- *Outputs:*
  - *Trained Encoder ($E$):* A model capable of mapping any motion clip from the
    dataset into a meaningful 64D latent code.
  - *Trained Low-Level Policy ($pi$):* A versatile motion generator that can execute
    a skill corresponding to an input latent code $z$.

=== Precision Training (High-Level Control)
The goal of this stage is to train a *High-Level Policy* that can steer the
character, enabling directional control over the skills learned in Stage 1.

- *Inputs:*
  - *Trained Encoder ($E$):* Frozen from Stage 1.
  - *Trained Low-Level Policy ($pi$):* Frozen from Stage 1.
  - *Task-Specific Inputs:*
    - Target Direction $d_t^*$: A vector indicating the desired direction of movement.
    - Target Motion Style ($hat(z)$): The latent code for a reference motion (e.g., "run"
      or "crouch-walk"), obtained by passing the reference motion through the frozen
      *Encoder*.

- *Components:*
  - *High-Level Policy ($pi_"HL"$):* An MLP that operates at a lower frequency (6Hz)
    than the low-level policy.
    - Input: Character State $s_t$, Target Direction $d_t^*$, and Target Motion Style
      encoding $hat(z)$ (or a one-hot encoding of the style).
    - Output: A latent vector $z_t in RR^( 64 )$ for the low-level policy to execute.

- *Process:*
  1. At each high-level step (every 6Hz), the *High-Level Policy* observes the
    character state and task goal (target direction and style).
  2. It outputs a latent code $z_t$, which is passed to the frozen *Low-Level
    Policy*.
  3. The *Low-Level Policy* executes for several steps (at 30Hz), controlling the
    character's joints.
  4. The *High-Level Policy* is updated using an RL algorithm. Its reward function, $r_"locomotion"$,
    encourages it to select latent codes $z_t$ that both move the character in the
    target direction and adhere to the desired motion style (i.e., keep $z_t$ close
    to $hat(z)$).

- *Output:*
  - *Trained High-Level Policy ($pi_"HL"$):* A model capable of directing the
    character to perform a specific style of locomotion in any specified direction.

=== Inference (Zero-Shot Task Solving)
The goal of this final stage is to solve complex, multi-step tasks by composing
the learned skills using a simple, rule-based system without any further
training.

- *Inputs:*
  - *Trained & Frozen Models:* The *Encoder*, *Low-Level Policy*, and *High-Level
    Policy* from the previous stages.
  - *Task Command:* A high-level instruction, like "crouch-walk to the target, then
    kick".

- *Components:*
  - *Finite State Machine (FSM):* A rule-based system designed by the user that
    dictates the sequence of behaviors required to solve a task. It does not require
    training.

- *Process:*
  1. The FSM tracks the overall task state (e.g., character's distance to a target).
  2. Based on the current state, the FSM selects a behavior. There are two types of
    commands it can issue:
    - *Directional Command:* For tasks like locomotion, the FSM provides the
      *High-Level Policy* with a target direction (e.g., "towards the enemy") and a
      desired motion style (e.g., "run"). The *High-Level Policy* then generates the
      appropriate latent codes for the *Low-Level Policy*.
    - *Direct Skill Command:* For isolated actions like "kick" or "roar," the FSM
      retrieves the corresponding pre-computed latent vector $hat(z)$ (using the
      frozen *Encoder*) and provides it directly to the *Low-Level Policy*, bypassing
      the high-level controller.
  3. The FSM transitions between states based on simple rules (e.g., "if distance to
    target < 1m, switch from 'run' to 'kick'").

- *Output:*
  - *Character Behavior:* A sequence of actions $a_t$ from the *Low-Level Policy*
    that results in the character executing the complex task defined by the FSM.

== Discussion

=== How effectively does the core generative model learn to represent and create controllable motions?

This overarching question assesses the quality of the foundational components
trained in the low-level phase: the *Encoder* and the *Low-Level Policy*. The
authors break this down into three sub-questions concerning the latent space
structure, motion diversity, and directability.

==== Does the encoder learn a semantically meaningful and well-structured latent space?

- *Experiment:*
  - *Quantitative:* The authors measured the class separability of the latent space
    embeddings produced by the *Encoder*. A "motion class" was defined as all the
    2-second sub-motions originating from a single motion capture file. They
    compared the *Encoder* from *CALM* against the one from *ASE*.
  - *Qualitative:* They visualized the latent space structure using a pairwise
    distance matrix to show the proximity between different motion categories (e.g.,
    Walk, Sword Attack). They also performed an interpolation between the latent
    codes for "sprint" and "crouching idle" to see if the character transitioned
    smoothly and logically between the two behaviors.

- *Metrics:*
  - *Fisher's Concentration Coefficient:* A metric for measuring class separability,
    where a lower value indicates that motions within the same class are clustered
    more tightly and are more distinct from other classes.
  - *Visual Inspection:* Qualitative assessment of the distance matrix heatmap and
    the interpolation video sequence.

- *Results & Significance:*
  - *CALM*'s encoder achieved a concentration coefficient of *0.23*, which was
    significantly better (lower) than *ASE*'s *0.68*.
  - The interpolation experiment showed a smooth and natural transition from a fast
    sprint to a slow crouch, indicating the space between these points is
    semantically meaningful.
  - *Significance:* This demonstrates that *CALM* learns a superior motion
    representation where similar motions are grouped together. A well-structured
    latent space is critical for enabling reliable control, smooth transitions, and
    effective learning for the high-level policy.

- *Limitations:*
  - The model's ability to generalize is not infinite. The authors anticipate that
    when conditioned on encodings of unseen motions that are very different from the
    training data (i.e., out-of-distribution), the model may fail to produce
    high-quality motions.

==== Does the low-level policy avoid mode collapse and generate a diverse set of behaviors?

- *Experiment:* The authors trained a classifier on the reference motion data to
  identify different motion types. They then used this classifier to evaluate the
  diversity of motions produced by the *Low-Level Policy* when it was fed randomly
  sampled latent codes.

- *Metric:*
  - *Inception Score:* A common metric for evaluating the quality and diversity of
    generative models. A higher score indicates that the model generates diverse and
    high-quality samples.

- *Results & Significance:*
  - *CALM* achieved a significantly higher Inception Score ($19.8 plus.minus 0.1$)
    compared to *ASE* ($18.6 plus.minus 0.4$).
  - *Significance:* This result shows that *CALM*'s training procedure, particularly
    the conditional discriminator, is more effective at preventing mode collapse.
    The model learns to cover the broad distribution of the training data, giving it
    a rich repertoire of skills to draw from.

- *Limitations:*
  - The authors state that mode collapse remains an open challenge. They observed
    that for some motions, like "idle," the policy produces unrealistic
    micro-motions instead of standing still, indicating a form of minor mode
    collapse.

==== Can the system be reliably controlled to produce a *specific* requested motion on demand?

- *Experiment:* This was the main "Controllability" test. The authors conducted a
  user study where human participants were shown a reference motion clip (e.g., a
  specific kick) and then a motion generated by either *CALM* or *ASE* when
  conditioned on that reference clip's encoding. Participants were asked to
  classify whether the generated motion was similar to the reference.

- *Metric:*
  - *Generation Accuracy:* The percentage of generated motions that users classified
    as accurately matching the reference motion.

- *Results & Significance:*
  - *CALM* achieved a generation accuracy of *78%*, which was a dramatic improvement
    over *ASE*'s *35%*.
  - *Significance:* This is arguably the most important result for the low-level
    system. It proves that *CALM* is not just a generative model but a *directable*
    one. The ability to reliably request a specific skill by providing its encoding
    is the core enabler for all the downstream applications, like the FSM control.

- *Limitations:*
  - While a significant improvement, 78% is not perfect, meaning there is still a
    ~22% failure rate in producing the requested motion to a human's satisfaction.

=== Can the learned skills be composed and controlled to solve complex, unseen tasks without task-specific training?

This question evaluates the practical utility of the framework, focusing on the
*High-Level Policy* and the *Finite State Machine (FSM)* inference method.

==== Can a high-level policy learn to add precise directional control to the generated motions?

- *Experiment:* This is the "Precision Training" or "Heading" task. A *High-Level
  Policy* was trained to control the frozen *Low-Level Policy*. The goal was to
  make the character move in a specified compass direction while performing a
  specific style of locomotion (run, walk, or crouch-walk).

- *Metrics:*
  - *Style Score:* A score from human raters assessing how well the generated motion
    matched the requested style (e.g., "was the character actually
    crouch-walking?").
  - *Heading Score:* A quantitative score based on the cosine distance between the
    character's actual velocity vector and the requested direction vector.

- *Results & Significance:*
  - The model achieved very high scores across the board (e.g., for run, style was
    1.0 and heading was 0.92; for crouch-walk, style was 0.94 and heading was 0.91).
  - *Significance:* This demonstrates that the learned skills are not just canned
    animations; they can be modulated in real-time by a higher-level controller to
    achieve task-oriented goals. This is the crucial step that connects the
    generative model to interactive control.

- *Limitations:*
  - The authors note that learning this control required "delicate tuning of the
    reward parameters," suggesting the process is not fully robust or automatic.
  - They also suggest that this method might be insufficient for controlling more
    intricate movements, like the precise arc of a sword swing, which would likely
    require further innovation.

==== Can the complete system solve multi-step tasks in a zero-shot manner using rule-based control?

- *Experiment:* The authors designed two downstream tasks: "Location" (reach a
  target) and "Strike" (reach a target and knock it down). Crucially, they did not
  train new policies for these tasks. Instead, they wrote simple, rule-based
  *Finite State Machines (FSMs)* that issued commands to the already-trained
  policies. For example, an FSM rule would be: "run towards the target until the
  distance is less than 1m, then perform a kick".

- *Metrics:*
  - *Success Rate:* For various combinations of locomotion styles and finishing
    moves (e.g., run then kick, crouch-walk then sword swipe), they measured the
    success rate of completing the task . The results are shown in Table 2.

- *Results & Significance:*
  - The system achieved near-perfect success rates (scores of 0.96 to 1.0) across
    all task variations .
  - *Significance:* This is the ultimate payoff of the *CALM* framework. It shows
    that by investing in a high-quality, directable generative model upfront, the
    burden of solving new tasks is dramatically reduced. Instead of complex and
    fragile reward engineering for every new task, a designer can use intuitive,
    rule-based systems (like FSMs or behavior trees) to compose behaviors, which is
    much closer to existing workflows in the animation and gaming industries.

- *Limitations:*
  - The authors explicitly state that the robustness of the FSM approach is limited
    to environments and dynamics similar to those seen during training. The policy's "robustness
    envelope" would likely not cover tasks like walking on very uneven terrain or
    climbing stairs without further, more specific training.

Of course. Here is a detailed comparison of Adversarial Skill Embeddings (ASE)
and Conditional Adversarial Latent Models (CALM), structured as a pipeline of
their common high-level stages.

== Comparison with ASE

The shared pipeline can be broken down into two main stages: a pre-training
phase to learn a general skill repertoire and a subsequent phase where those
skills are used to solve specific tasks.

=== Stage 1: Low-Level Skill Pre-Training
The goal of this stage is to learn a versatile, low-level policy from a large,
unstructured dataset of motion clips.

==== ASE's Approach

- *Imitation Objective:* ASE trains its low-level policy, $pi(a|s, z)$, to match
  the *overall marginal distribution* of state-transitions from the entire motion
  dataset $cal(M)$. This is achieved with a standard, *unconditional
  discriminator* $D(s,s')$ that only determines if a transition looks like it came
  from the dataset in general.
- *Skill Discovery Objective:* To prevent the policy from ignoring the latent code $z$ and
  to encourage diversity, ASE adds a separate, explicit objective to maximize the
  *mutual information* $I(s,s';z|pi)$ between the latent code and the resulting
  state transitions. This is implemented with a variational lower bound using a
  separate *encoder* network $q(z|s,s')$.
- *Diversity Objective:* To further combat mode-collapse, ASE introduces an
  additional diversity loss term that explicitly encourages different latent codes
  to produce different action distributions .

==== CALM's Approach
- *Imitation Objective:* CALM fundamentally changes the imitation objective. It
  uses a *conditional discriminator*, $D(s,s'|z)$, which is tasked with
  determining if a state transition is realistic *given the specific latent code
  z* that is conditioning the policy .
- *Unified Objective:* CALM does *not* use a separate mutual information or
  diversity objective in its reward function. The conditional nature of the
  discriminator implicitly forces the policy to be diverse and controllable. If
  the policy produces the wrong motion for a given $z$, the discriminator will
  identify it as "fake," thus collapsing the imitation and skill discovery goals
  into a single, more powerful objective.
- *Encoder Training:* The encoder in CALM is trained end-to-end using gradients
  that flow back from the policy's objective, directly optimizing the
  representation for the control task. It also adds alignment and uniformity
  losses to better structure the latent space .

==== Why CALM Changed the Design
The authors of CALM identified significant limitations in ASE's approach that
they aimed to solve:
1. *Lack of Directability:* ASE's method of matching the *marginal* data
  distribution and then adding diversity objectives resulted in an "imprecise
  mapping" between motions and latent codes. It was difficult to request a
  *specific* motion and have the policy reliably produce it, with ASE achieving
  only 35% accuracy in a user study on this task. *CALM's conditional
  discriminator directly enforces this mapping*, leading to a much higher 78%
  controllability accuracy.
2. *Prone to Mode-Collapse:* Despite having explicit diversity objectives, ASE was
  still susceptible to mode-collapse, where it fails to model the full variety of
  motions in the dataset. CALM's conditional approach forces the policy to learn
  how to produce *every* motion in the dataset to fool the discriminator, which
  naturally mitigates mode collapse and results in a more diverse skill set.

=== Stage 2: High-Level Control and Task Solving
The goal of this stage is to use the pre-trained, low-level policy to perform
new, downstream tasks.

==== ASE's Approach
- *Task-Training:* For each new task, a *high-level policy* $omega(z|s, g)$ is
  trained from scratch. This policy learns to select a sequence of latent codes $z_t$ to
  command the frozen low-level policy to solve the task.
- *Reward Function:* The high-level policy's reward is a weighted sum of a
  task-specific reward (e.g., for reaching a location) and a "style reward" from
  the frozen, pre-trained discriminator. This style reward encourages the
  character to maintain natural-looking motions while pursuing the task goal.

==== CALM's Approach
- *Precision Training:* CALM also trains a high-level policy, but its reward
  function is designed for more direct style control. It includes a task term
  (e.g., move in direction $d^*$) and a *latent similarity loss* that explicitly
  encourages the high-level policy's output latents $z_t$ to stay close to a
  target style's encoding $hat(z)$.
- *Zero-Shot Inference via FSM:* This is a key innovation not present in ASE. CALM
  introduces a *Finite State Machine (FSM)* for solving complex, multi-step tasks
  *without any new training*. The FSM acts as a master controller, directing the
  high-level policy for continuous control (like locomotion) or directly
  commanding the low-level policy with specific latent codes for discrete actions
  (like "kick").

==== Why CALM Changed the Design
The changes in this stage were motivated by practicality and the desire to
create a more user-friendly and flexible system:
1. *Avoiding Reward Engineering:* ASE's method requires training a new high-level
  policy with a potentially complex, hand-tuned reward function for each new task.
  CALM's *FSM approach completely removes the need for task-specific training or
  reward
  engineering for many tasks*. This is more practical and aligns better with
  existing workflows in the gaming and animation industries, where designers use
  rule-based systems like behavior trees.
2. *Enabling True Compositionality:* Because ASE's mapping was imprecise, composing
  skills like "run, then strike" would require careful reward design. CALM's
  superior directability allows an FSM to simply command these skills sequentially
  ("play the 'run' skill, then play the 'strike' skill") to achieve a complex goal
  with no new learning, enabling true zero-shot composition of behaviors .
