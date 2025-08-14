= Adversarial Skill Embeddings

== Overview

=== Challenges

==== Challenge 1: Inefficiency of Learning from Scratch

- *Problem:* The standard approach in character animation is to train a new
  control policy from scratch for every new task (`tabula-rasa` learning). This is
  inefficient as it forces the agent to relearn common skills like walking and
  balancing repeatedly .
- *Hypothesis:* A hierarchical model that first learns a broad repertoire of
  reusable skills and then leverages those skills for new tasks will be more
  efficient and capable, mirroring how humans learn.
- *Approach:* The paper proposes a two-stage framework:
  1. *Pre-training:* A *low-level policy* is trained on a large, unstructured motion
    dataset to learn a general "skill embedding".
  2. *Task-training:* A *high-level policy* is then trained to solve new tasks by
    commanding the pre-trained low-level policy, effectively reusing the learned
    skills.
- *Alternatives Discussed:* The primary alternative is the conventional method of
  training specialized, non-reusable policies from scratch for each task.

==== Challenge 2: Difficulty of Designing Realistic Behaviors
- *Problem:* Manually designing reward functions that produce natural, life-like
  motions is a tedious and labor-intensive process. Unsupervised RL methods, while
  promising, are unlikely to discover human-like behaviors on their own in
  complex, high-dimensional spaces.
- *Hypothesis:* A large dataset of human motion implicitly contains the priors for
  natural behavior. This data can be used to train a model to imitate these
  behaviors without needing complex, hand-engineered reward functions.
- *Approach:* The framework uses *adversarial imitation learning*. A discriminator
  is trained to distinguish between motions from a real dataset and those produced
  by the character's policy . The policy is then rewarded for fooling the
  discriminator, pushing it to produce motions that are statistically similar to
  the reference data . This allows complex behaviors to be learned from simple
  task rewards later on, as the "naturalness" is provided by the pre-trained
  model.
- *Alternatives Discussed:* Manually designing objectives with heuristics for
  energy efficiency, stability, or symmetry, which are often not broadly
  applicable across different skills .

==== Challenge 3: Limitations of Existing Imitation and Skill Discovery Methods
- *Problem:*
  1. Simple motion tracking is too rigid and limits a model's ability to generalize
    or produce novel behaviors not explicitly seen in the data.
  2. Standard adversarial imitation learning is prone to *mode-collapse*, where the
    policy learns only a narrow subset of the behaviors from the dataset.
- *Hypothesis:* Combining a flexible, distributional imitation objective with an
  unsupervised skill discovery objective can solve both problems. The imitation
  objective provides naturalness, while the skill discovery objective ensures a
  diverse and controllable set of skills is learned.
- *Approach:* The core of ASE is a novel pre-training objective that combines two
  main terms:
  1. *Imitation Objective:* Minimizes the Jensen-Shannon divergence between the
    policy's state-transition distribution and the motion dataset's distribution,
    encouraging realism without rigid tracking.
  2. *Skill Discovery Objective:* Maximizes the mutual information between a latent
    skill variable $z$ and the resulting behavior, encouraging the model to learn a
    diverse and directable set of skills . An additional diversity loss is also used
    to further mitigate mode-collapse .
- *Alternatives Discussed:*
  - *Motion Tracking:* Explicitly tracking a sequence of poses, which is less
    flexible.
  - *Pure Unsupervised RL:* Fails to discover naturalistic behaviors in
    high-dimensional domains .

=== Proposed Solution: Adversarial Skill Embeddings (ASE)
- *Description:* ASE is a hierarchical, data-driven framework for learning
  reusable motor skills. It consists of a *low-level, skill-conditioned policy* ($pi(a|s, z)$)
  that translates latent codes into behaviors, and a *high-level policy* ($omega(z|s, g)$)
  that learns to select these latent codes to solve downstream tasks.
- *Inputs:*
  - *Pre-training:* A large, unstructured dataset of motion clips ($cal(M)$).
  - *Task-training:* A task-specific goal ($g$) and a simple reward function.
- *Outputs:*
  - *From Pre-training:* A versatile, reusable *low-level policy* that forms a rich
    skill repertoire.
  - *From Task-training:* A physically simulated character that can perform a
    specific task (e.g., "Strike," "Reach") using complex, naturalistic motions that
    emerge automatically.

=== Dependencies

- *Simulator:* *Isaac Gym*, a high-performance GPU-based physics simulator from
  NVIDIA, is used for all experiments.
- *Dataset:* A motion dataset of 187 clips (~30 minutes) provided by *Reallusion*.
- *RL Algorithm:* The policies are trained using *Proximal Policy Optimization
  (PPO)*.

=== Perspectives Missing from the Abstract

- *The Core Objective Function:* The abstract mentions combining adversarial
  imitation and unsupervised RL, but doesn't specify the concrete objective
  function, which is a sum of a discriminator-based reward and a
  mutual-information-based skill discovery reward .
- *Specific Techniques for Stability and Diversity:* The abstract does not mention
  several crucial design decisions detailed in the paper, such as modeling the
  latent space as a *hypersphere* to prevent out-of-distribution samples , adding
  an explicit *diversity loss objective* to prevent mode-collapse , and training
  the policy to *recover from fallen states* to improve robustness.
- *Hierarchical Action Space:* It omits the important detail that the high-level
  policy operates in the *unnormalized* latent space ($tilde(z)$) to provide a
  natural trade-off between exploration and exploitation, which is a key part of
  the high-level controller design.

=== Glaring Assumptions
- *Data Sufficiency:* The framework's success hinges on the assumption that the
  provided motion dataset is large and diverse enough to contain all the
  characteristics of behavior required for downstream tasks. The model can only
  produce motions that are statistically similar to what it has seen.
- *Simulation Fidelity:* The work assumes that the physics simulation in Isaac Gym
  is a sufficient proxy for real-world physics, and that the skills learned within
  it are meaningful.
- *Task Decomposability:* It assumes that complex tasks can be solved by a
  high-level policy sequencing skills from the pre-trained low-level policy. This
  may not hold for all tasks, especially those requiring novel skills not
  representable by the pre-trained model.

=== Prerequisite Knowledge
To fully grasp the paper's contributions, familiarity with these concepts is
recommended:
- *Adversarial Imitation Learning (AIL):* Specifically, understanding how a
  GAN-like framework can be used for imitation, where a discriminator's output
  serves as a reward signal. The paper builds directly on methods like *AMP*.
- *Information Maximization (Mutual Information):* The use of mutual information
  as an intrinsic reward for skill discovery is central. Familiarity with works
  like *InfoGAN* is helpful context.
- *Hierarchical Reinforcement Learning (HRL):* A general understanding of the HRL
  paradigm, involving low-level policies executing skills and a high-level policy
  managing them, is essential to understand the overall architecture .

== Problem Formulation

=== Overall Objective

The primary goal is to learn a versatile and reusable motor skill model for a
physically simulated character. This is framed as a two-stage learning problem:
1. *Pre-training*: Learn a low-level, skill-conditioned policy $pi(a|s, z)$ from a
  large, unstructured motion dataset $cal(M)$.
2. *Task-training*: Use the pre-trained low-level policy to efficiently learn a
  high-level policy $omega(z|s, g)$ for new downstream tasks.

The entire framework is built on the foundation of *Reinforcement Learning
(RL)*, where the objective is to learn a policy $pi$ that maximizes the expected
discounted return:

$
  J(pi) = EE_(tau ~ p(tau|pi)) [sum_(t = 0)^(T - 1) gamma^t r_t ]
$

where $tau$ is a trajectory, $gamma$ is the discount factor, and $r_t$ is the
reward at time $t$.

=== Stage 1: Pre-training the Low-Level Policy

In this stage, the goal is to train a skill-conditioned policy $pi(a|s, z)$ that
maps a state $s$ and a latent skill variable $z$ to an action $a$. The latent
variable $z$ is sampled from a prior distribution $p(z)$.

The core objective is to learn skills that are both *realistic* (imitating the
motion data) and *diverse/controllable*. This is captured by the following
objective function:

$
  max_pi - D_(J S)(d^pi (s, s')||d^cal(M) (s, s')) + beta I(s, s'; z|pi)
$

where:
- $D_"JS"$ is the *Jensen-Shannon divergence*, which measures the similarity
  between the state-transition distribution of the policy, $d^( pi )(s,s')$, and
  that of the motion dataset, $d^( cal(M) )(s,s')$. This is the *imitation
  objective*.
- $I(s,s';z|pi)$ is the *mutual information* between state transitions $(s,s')$ and
  the latent skill $z$. This is the *skill discovery objective*.
- $beta$ is a coefficient balancing the two objectives.

Since this objective is intractable to compute directly, it is approximated as
follows:

==== Imitation Objective (Adversarial Formulation)

The Jensen-Shannon divergence is minimized using a GAN-like adversarial setup. A
*discriminator* $D(s,s')$ is trained to distinguish between state transitions
from the motion dataset (`real`) and those generated by the policy (`fake`).

- *Discriminator's Objective* (to be minimized):

$
  min_D - EE_(d^cal(M) (s, s')) [log(D(s, s'))] - EE_(d^pi (s, s')) [log(1 - D(s, s'))]
$

To improve stability, a gradient penalty is added, making the full objective:

$
  min_D - EE_(d^cal(M) (s, s')) [log(D(s, s'))] - EE_(d^pi (s, s')) [log(1 - D(s, s'))] + w_(g p) EE_(d^cal(M) (s, s')) [||nabla_phi.alt D(phi.alt)|_(phi.alt = (s, s'))||^2 ]
$

- *Policy's Imitation Reward*: The policy is then trained with a reward signal
  from the fixed discriminator to make its transitions look more `real`:

$
  r_t^"imitate" = - log(1 - D(s_t, s_(t + 1)))
$

==== Skill Discovery Objective (Mutual Information Maximization)

The goal is to make skills distinct and predictable. Maximizing mutual
information $I(s,s';z|pi)$ encourages this.

- *Mutual Information*: $I(s,s';z|pi) = cal(H)(z) - cal(H)(z|s,s',pi)$
  - $cal(H)(z)$ is the entropy of the prior over skills (a constant).
  - $cal(H)(z|s,s',pi)$ is the conditional entropy of the skill given the resulting
    transition. Minimizing this term means that given a transition, the skill that
    produced it should be unambiguous.

- *Variational Approximation*: To make this tractable, the conditional entropy is
  approximated using a variational distribution $q(z|s,s')$, called the *encoder*.
  This results in a variational lower bound on the mutual information:

$
  I(s, s'; z|pi) >= cal(H)(z) + EE_(p(z)) EE_(p(s, s'|pi, z)) [log q(z|s, s')]
$

- *Policy's Skill Discovery Reward*: The policy receives a reward based on how
  well the encoder can predict the skill $z$ from the resulting transition $(s_t, s_{t+1})$:

$
  r_t^"discover" = log q(z_t |s_t, s_(t + 1))
$

==== Full Pre-training Objective

Combining the components, the low-level policy $pi$ is trained via RL using a
reward function that is the sum of the imitation and skill discovery rewards. To
further improve responsiveness and prevent mode-collapse, a *diversity
objective* is added directly to the policy's loss function.

- *Final Policy Reward* at each timestep $t$:

$
  r_t = - log(1 - D(s_t, s_(t + 1))) + beta log q(z_t |s_t, s_(t + 1))
$

- *Full Policy Objective* (to be maximized, including diversity term):

$
  arg max_pi EE_(p(Z)) EE_(p(tau|pi, Z)) [sum_(t = 0)^(T - 1) gamma^t r_t ] - w_"div" EE_(d^pi (s)) EE_(z_1, z_2 ~ p(z)) [((D_(K L)(pi(dot.op|s, z_1), pi(dot.op|s, z_2)))/(D_z (z_1, z_2)) - 1)^2 ]
$

The diversity term encourages the action distributions produced by two different
skills ($z_1, z_2$) to be as dissimilar as the skills themselves.

=== Stage 2: Task-Training the High-Level Policy

After pre-training, the low-level policy $pi(a|s, z)$ and discriminator $D(s,s')$ are
frozen. A new, task-specific *high-level policy* $omega(z|s, g)$ is trained to
solve a downstream task defined by a goal $g$. This policy outputs a latent
skill vector $z$ to command the low-level policy.

- *Hierarchical Action*: The high-level policy $omega$ selects $z_t$, and the
  low-level policy $pi$ executes action $a_t ~ pi(dot|s_t, z_t)$.

- *High-Level Policy Reward*: The reward function for $omega$ is a weighted sum of
  a task-specific reward and a style reward from the pre-trained discriminator.

$
  r_t = w_G r_t^G (s_t, a_t, s_(t + 1), g) - w_S log(1 - D(s_t, s_(t + 1)))
$

- $r_t^G$ is the extrinsic reward for the specific task (e.g., reaching a target).
- The second term is a *motion prior* that encourages the high-level policy to
  select sequences of skills that result in natural, human-like motions,
  preventing jittery or strange behaviors.

Here is a detailed outline of the research questions, experiments, results, and
limitations discussed in the paper.

== Discussion

=== Quality and Diversity of Learned Skills

- *Question*: Can the ASE pre-training process learn a rich and diverse set of
  skills from a large, unstructured motion dataset without any explicit skill
  labels?

- *Experiment Design*:
  - The fully pre-trained low-level policy, $pi(a|s, z)$, was qualitatively
    evaluated.
  - Researchers conditioned the policy on random latent vectors, $z$, and observed
    the resulting character behaviors in the simulation.

- *Results & Metrics*:
  - *Metric*: The primary metric was *qualitative observation* of the variety and
    realism of the generated motions.
  - *Finding*: A single low-level policy was able to generate a wide array of
    complex and naturalistic skills, ranging from standard locomotion (walking,
    running) to dynamic, specialized actions (sword swings, shield bashes, kicks) .
    The model successfully organized the unstructured data into a coherent skill
    embedding, where different latent vectors produced semantically distinct
    behaviors .

- *Significance*: This result validates the core premise of the pre-training
  stage. It shows that the combination of adversarial imitation and an
  unsupervised skill discovery objective is powerful enough to learn a versatile
  repertoire of skills from a "soup" of unorganized motion clips, a key step
  towards creating general-purpose character controllers .

- *Limitations*: This initial evaluation was purely qualitative.

=== Reusability for Downstream Tasks

- *Question*: Can the pre-trained low-level policy be effectively reused as a
  foundation to solve a variety of new, downstream tasks with minimal, simple
  reward functions?

- *Experiment Design*:
  - The *same* pre-trained low-level policy was used for all experiments in this
    section.
  - Separate high-level policies were trained for five distinct tasks designed to
    test different capabilities:
    1. *Reach*: Precision control (move sword tip to target) .
    2. *Speed*: Locomotion control (match a target speed and direction).
    3. *Steering*: Decoupled control (walk in one direction while facing another).
    4. *Location*: Navigation (move to a target location).
    5. *Strike*: Skill composition (run to a target and knock it over with the sword).

- *Results & Metrics*:
  - *Metrics*: Performance was measured both *quantitatively* (task success,
    normalized return) and *qualitatively* (naturalness of the motion). For the
    Reach task, the average tracking error was $0.088 plus.minus 0.046$ meters.
  - *Finding*: The character learned to solve all tasks by automatically composing
    skills from the low-level policy in natural ways. For example, in the Strike
    task, it learned to sequence running behaviors with a sword swing to achieve the
    goal, despite only being rewarded for knocking the target over .

- *Significance*: This is the central result of the paper. It demonstrates that
  the ASE framework successfully creates a *reusable skill embedding*. This
  embedding serves as a powerful motion prior, allowing a character to learn new
  tasks efficiently while producing high-quality, life-like motions without the
  need for complex, hand-engineered reward functions.

- *Limitations*: The paper notes that policies trained entirely from scratch
  sometimes achieved higher numerical scores (normalized return). However, this
  was because the "scratch" policies learned to exploit the simple reward
  functions with unnatural, sporadic, and jittery movements, whereas the ASE-based
  policies maintained more realistic behaviors.

=== Importance of Pre-Training Objectives

- *Question*: How critical are the *skill discovery (SD)* and *diversity (Div.)*
  objectives for learning a high-quality, non-collapsed skill embedding and for
  downstream performance?

- *Experiment Design (Ablation Study)*:
  1. *Dataset Coverage*: Compared the full ASE model against versions trained without
    the SD objective, without the Div. objective, and without both. They measured
    how well each model could reproduce the variety of motions in the original
    dataset to test for mode-collapse.
  2. *Skill Transitions*: Measured the ability of each ablated model to transition
    between different skills by feeding it a sequence of two different latent codes
    and observing if the behavior changed accordingly .
  3. *Task Performance*: Re-ran the downstream task experiments using the ablated
    low-level policies to measure the impact on final performance.

- *Results & Metrics*:
  - *Metrics*:
    - *Dataset Coverage*: A frequency histogram showing which motion clips the policy
      most often imitated. A flatter distribution indicates less mode-collapse.
    - *Transition Coverage*: A matrix showing the observed transitions between skills.
      A denser matrix indicates a more responsive model. The metric was the portion of
      all possible transitions the model could perform.
    - *Normalized Return*: Learning curves and final scores on downstream tasks.
  - *Finding*: The *skill discovery objective was crucial*. Without it, the model
    suffered from severe mode-collapse, predominantly learning a single "idle"
    behavior and failing to cover the dataset. It also resulted in far fewer
    transitions between skills and a significant drop in downstream task
    performance. The diversity objective also improved transitions but had a less
    dramatic impact on the final performance in this specific suite of tasks.

- *Significance*: This provides strong quantitative evidence that simply using an
  adversarial objective is not enough. The information-theoretic *skill discovery
  objective is a critical ingredient* that forces the model to learn a structured
  and diverse set of skills, which directly translates to better performance and
  responsiveness in new tasks.

=== Robustness and Recovery

- *Question*: Can the framework produce a policy that automatically and robustly
  recovers from falls, even if "get up" motions are not present in the training
  data?

- *Experiment Design*:
  - The pre-training process included a 10% chance of starting an episode in a
    random fallen state.
  - To test the final policy, the character was subjected to a large, random
    external force for 0.5s to knock it down. This test was repeated 500 times.

- *Results & Metrics*:
  - *Metrics*: The *success rate* of recovery and the *time required* to get back to
    a stable, upright posture .
  - *Finding*: The policy was *100% successful* in recovering from all 500 falls.
    The average recovery time was a quick 0.31 seconds. The character learned
    plausible recovery strategies, like using its hands to push off the ground, that
    emerged naturally from the training process.

- *Significance*: This shows that robust recovery behaviors can be "baked in"
  during pre-training and are then available for *free* on any downstream task,
  without needing to be re-learned. This makes the resulting controller
  significantly more practical and resilient to perturbations .

- *Limitations*: The authors admit that some of the learned recovery motions can
  appear unnatural or "overly energetic". They suggest that including motion
  capture data of actual human recovery strategies could improve the realism.
