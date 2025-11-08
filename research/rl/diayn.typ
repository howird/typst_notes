#import "../styles/things.typ": challenge, hypothesis, question

= Diversity is All You Need

== Overview

=== Challenges

#challenge[
  Learning without a Reward Function
][
  Standard reinforcement learning requires a reward function, which can be
  difficult to design or too sparse to provide a useful learning signal.

  #hypothesis[
    Skills that are maximally diverse and distinguishable from one another based
    on the states they visit will be inherently useful and cover a wide range of
    behaviors.
  ]

  The paper proposes an information-theoretic objective. It trains a
  latent-conditioned policy by maximizing the mutual information between the
  latent skill variable ($Z$) and the visited states ($S$), written as $I(S;Z)$.
  To encourage diversity and robust exploration, the objective also includes
  maximizing the policy's entropy ($cal(H)[A|S, Z]$). The final objective is:

  $
    cal(F)(theta) & eq.delta I(S;Z)+cal(H)[A|S]-I(A;Z|S) \
                  & = cal(H)[Z]-cal(H)[Z|S]+cal(H)[A|S,Z]
  $
]

#challenge[
  Mode Collapse in Skill Discovery
][
  Prior methods for unsupervised skill discovery, such as Variational Intrinsic
  Control (VIC), sometimes suffer from a "Matthew Effect" where the algorithm
  learns to favor a small subset of skills, causing a collapse in diversity and
  preventing other skills from improving.

  #hypothesis[
    Forcing the algorithm to sample all skills with equal probability throughout
    training will prevent it from ignoring certain skills and lead to a more
    diverse final set.
  ]

  Instead of learning the prior distribution over skills $p(z)$, DIAYN keeps it
  *fixed* as a uniform categorical distribution. This ensures all skills receive
  a continuous training signal. The VIC algorithm learns the prior $p(z)$, which
  the paper demonstrates leads to the effective number of skills collapsing
  during training.
]

#challenge[
  Leveraging Unsupervised Skills for Downstream Tasks
][
  Once a set of task-agnostic skills is learned, they must be effectively
  applied to solve specific, supervised tasks.

  #hypothesis[
    The learned skills provide a powerful set of behavioral primitives that can
    serve as building blocks for solving complex problems more efficiently than
    learning from scratch.
  ]

  The paper demonstrates three applications:
  1. *Policy Initialization:* The weights of the best-performing unsupervised
    skill are used to initialize a policy for a new task, which is then
    fine-tuned with the task-specific reward. This accelerates learning.
  2. *Hierarchical RL:* The learned skills are treated as a fixed set of
    low-level actions. A high-level "meta-controller" is then trained to select
    among these skills to solve complex, sparse-reward tasks like Ant
    Navigation.
  3. *Imitation Learning:* Given an expert's state-only trajectory, the learned
    discriminator is used to find the skill most likely to have generated that
    trajectory, enabling imitation without access to expert actions.
]

=== Proposed Component

DIAYN is a framework that learns a set of skills by optimizing an
information-theoretic objective. It uses a cooperative game between a policy and
a discriminator.
- The *policy* $pi_theta (a_t | s_t, z)$ is conditioned on a skill variable $z$
  and tries to visit states that make its skill easily identifiable.
- The *discriminator* $q_phi (z | s_t)$ is trained to predict the skill $z$ from
  the current state $s_t$.
- The discriminator's output provides an intrinsic reward for the policy:
  $r_t = log q_phi (z|s_t+1) - log p(z)$.

- *Inputs (for unsupervised training):*
  - An environment providing states and allowing actions, but with *no reward
    function*.
  - A pre-defined, fixed prior over skills, $p(z)$ (typically uniform).
- *Output:*
  - A single, latent-conditioned policy $pi_theta (a|s, z)$ that can execute a
    variety of distinct skills corresponding to each latent code $z$.

=== Dependencies

- *Underlying RL Algorithm:* *Soft Actor-Critic (SAC)* is used to perform the
  policy updates, as its maximum entropy framework naturally aligns with the
  DIAYN objective.
- *Environments:*
  - *Standard Benchmarks:* OpenAI Gym environments `HalfCheetah-v1`, `Ant-v1`,
    `Hopper-v1`, `MountainCarContinuous-v0`, and `InvertedPendulum-v1`.
  - *Custom Environments:*
    - A 2D point navigation task.
    - A `Half Cheetah Hurdle` task with added obstacles.
    - An `Ant Navigation` task with sparse waypoint rewards.
- *Comparison Baselines:*
  - Variational Information Maximizing Exploration (*VIME*)
  - Trust Region Policy Optimization (*TRPO*)
  - Variational Intrinsic Control (*VIC*)

=== Additional Perspectives & Assumptions

- *Significant Missing Perspective from Abstract:* The abstract does not
  explicitly mention the critical design choice of *using a fixed, uniform prior
  for the skills, $p(z)$*. This is a primary contribution that differentiates
  DIAYN from prior work like VIC and is key to preventing the mode collapse that
  plagues other methods.
- *Glaring Assumption:* The theoretical analysis for gridworlds (Appendix B)
  assumes that the empirical distribution of states visited within an episode
  exactly matches the policy's stationary distribution. This is justified by
  assuming infinite-length episodes, which is a strong assumption not met in
  practice but useful for deriving theoretical insights.

=== Recommended Prerequisite Knowledge

- *Soft Actor-Critic (SAC):* Understanding the SAC algorithm is crucial, as
  DIAYN is implemented directly on top of it and relies on its maximum entropy
  objective.
- *Information Theory Concepts:* A basic grasp of mutual information $I(X;Y)$
  and Shannon entropy $cal(H)[X]$ is necessary to understand the derivation and
  intuition behind the DIAYN objective.

== Problem Formulation

The goal of DIAYN is to learn a set of diverse skills, represented by policies
conditioned on a latent variable $z$, without an external reward signal. This is
framed as an information-theoretic optimization problem.

=== Core Objective

The central idea is to learn skills that are easily distinguishable from the
states they visit. This is captured by maximizing the *mutual information*
between the skill $Z$ and the state $S$, denoted as $I(S;Z)$. To encourage the
skills to be as different as possible and to explore widely, the objective also
maximizes the policy's entropy. The full objective function $cal(F)(theta)$ is
defined as:

$
  cal(F)(theta) eq.delta I(S; Z) + cal(H) [A|S] - I(A; Z|S)
$

Using the definitions of mutual information and entropy, this can be expanded
and simplified into a more intuitive form:

$
  cal(F)(theta) = underbrace(cal(H) [Z], "High-entropy prior") - underbrace(cal(H) [Z|S], "Low skill uncertainty given state") + underbrace(cal(H) [A|S comma Z], "High-entropy skills") quad "(1)"
$

This formulation encourages three properties:
- The prior distribution over skills $p(z)$ should be high-entropy (i.e.,
  uniform), which is achieved by fixing it.
- The skill $z$ should be easy to infer from a given state $s$, which means
  minimizing the conditional entropy $cal(H)[Z|S]$.
- Each individual skill should act as randomly as possible by maximizing its
  action entropy, $cal(H)[A|S,Z]$.

=== Variational Lower Bound

Directly optimizing Equation (1) is intractable because it requires computing
the true posterior $p(z|s)$. To address this, the paper introduces a trainable
*discriminator network*, $q_phi(z|s)$, to approximate this posterior. By
substituting $q_phi(z|s)$ for $p(z|s)$ and applying Jensen's Inequality, we get
a tractable variational lower bound, $cal(G)(theta, phi)$, on the original
objective:

$
  cal(F)(theta) >= cal(G)(theta, phi.alt) eq.delta EE_(z ~ p(z), s ~ pi(z)) [log q_phi.alt (z|s) - log p(z)] + cal(H) [A|S, Z] quad "(2)"
$

DIAYN works by maximizing this lower bound with respect to both the policy
parameters $theta$ and the discriminator parameters $phi$.

=== Intrinsic Reward

The optimization of the lower bound in Equation (2) is implemented within a
reinforcement learning framework. The expectation term is treated as an
*intrinsic pseudo-reward* given to the agent at each timestep. The entropy term
$cal(H)[A|S,Z]$ is maximized implicitly by the choice of RL algorithm, Soft
Actor-Critic (SAC).

The pseudo-reward $r_z(s,a)$ for a given skill $z$ is defined as:

$
  r_z (s, a) eq.delta log q_phi.alt (z|s) - log p(z) quad "(3)"
$

This reward encourages the policy to take actions that lead to states where the
discriminator can easily identify the active skill $z$. The $- log p(z)$ term
acts as a baseline that makes the reward non-negative and encourages the agent
to stay "alive" to accumulate more reward.

== Pipeline

The DIAYN algorithm is implemented as a cooperative game between the policy and
the discriminator, updating them iteratively using experiences stored in a
replay buffer. The pipeline proceeds in an episode-by-episode loop.

=== Episode Initialization & Skill Sampling

At the beginning of each training episode, a single skill is sampled and used
for the entire episode's duration.

- *Inputs:*
  - `p(z)`: A fixed, uniform categorical distribution over the skills.
- *Process:*
  - Sample a skill variable $z ~ p(z)$. This $z$ will remain constant for all
    steps in the current episode.
- *Outputs:*
  - `z`: The sampled skill. This is typically represented as a one-hot vector.
    - *Shape*: `(N_skills,)` where `N_skills` is the total number of skills.

=== Environment Interaction & Data Collection

The agent interacts with the environment for a sequence of steps, collecting
transition data.

- *Inputs:*
  - `s_t`: The current state from the environment.
    - *Shape*: `(state_dim,)`
  - `z`: The skill vector sampled in Stage 1.
    - *Shape*: `(N_skills,)`
- *Process (per timestep `t`):*
  1. Concatenate the state and skill `[s_t, z]`.
  2. Pass the concatenated vector to the policy network $pi_theta(a_t | s_t, z)$
    to get an action $a_t$.
  3. Execute action $a_t$ in the environment to receive the next state $s_t+1$.
  4. Store the transition tuple `(s_t, a_t, s_t+1, z)` in a replay buffer.
- *Outputs:*
  - A replay buffer populated with experience tuples.

=== Discriminator Update

The discriminator is trained to correctly classify the skill based on the state
visited. This is a standard supervised learning update.

- *Inputs:*
  - A mini-batch of `(s_t+1, z)` pairs sampled from the replay buffer.
    - *Shape*: `s_t+1` is `(batch_size, state_dim)`, `z` is
      `(batch_size, N_skills)`.
- *Process:*
  1. Feed the batch of states `s_t+1` into the discriminator network
    $q_phi(z|s_t+1)$.
  2. Calculate the cross-entropy loss between the discriminator's predicted
    skill distribution and the true skill `z`.
  3. Perform a gradient descent step on the discriminator's parameters $phi$ to
    maximize $log q_phi(z|s_t+1)$.
- *Outputs:*
  - Updated discriminator network parameters $phi$.

=== Policy & Value Function Update

The policy (and its associated value functions) is updated using the intrinsic
reward signal generated by the discriminator. This is where the skill learning
happens.

- *Inputs:*
  - A mini-batch of transitions `(s_t, a_t, s_t+1, z)` from the replay buffer.
    - *Shape*: `s_t`, `s_t+1` are `(batch_size, state_dim)`; `a_t` is
      `(batch_size, action_dim)`; `z` is `(batch_size, N_skills)`.
- *Process:*
  1. *Calculate Intrinsic Reward:* For the batch of next states `s_t+1` and
    skills `z`, compute the intrinsic reward using the current discriminator
    $q_phi$ as described in *Equation (3)*:
    $r_t = log q_phi (z|s_t+1) - log p(z)$.
  2. *Perform SAC Update:* Use the batch of transitions `(s_t, a_t, s_t+1, z)`
    and the calculated intrinsic rewards `r_t` to perform a standard Soft
    Actor-Critic (SAC) update step. This updates the parameters $theta$ of the
    policy $pi_theta$ and its associated Q-functions. This step maximizes the
    sum of intrinsic rewards and the policy's entropy, thereby optimizing the
    full objective from *Equation (2)*.
- *Outputs:*
  - Updated policy network parameters $theta$.
  - Updated Q-function and value function parameters used by SAC.

== Discussion

Here is a detailed outline of the questions the paper aimed to answer, the
experimental design for each, and their results.

=== Analysis of Learned Skills

#question[
  What skills does DIAYN learn?
][
  *Experiment Design:* The DIAYN algorithm was applied to a variety of standard
  reinforcement learning environments without providing any task-specific
  reward. The environments included 2D Navigation, Inverted Pendulum, Mountain
  Car, Half Cheetah, Hopper, and Ant. The goal was to observe the types of
  behaviors (skills) that emerged from optimizing the information-theoretic
  objective alone.
][
  *Metrics:* The primary metric was *qualitative observation* of the emergent
  behaviors, visualized as agent trajectories.

  *Results:* DIAYN successfully discovered a diverse set of complex skills. For
  example, it learned running forwards and backwards, flipping, and hopping for
  the Half Cheetah and Hopper agents. In the Ant environment, it learned to walk
  in various curved trajectories. Critically, on benchmark tasks like Mountain
  Car, it learned multiple distinct skills that could solve the task, despite
  never receiving the task reward.
][
  *Significance:* The results show that a simple, task-agnostic objective based
  on discriminability and diversity is sufficient to produce a rich repertoire
  of semantically meaningful skills. This validates the core hypothesis that
  maximizing information can lead to useful behaviors.

  *Limitations:* The discovered skills are not guaranteed to be optimal for any
  specific task. For instance, while the Ant agent learned many locomotion
  skills, no single skill learned to run in a straight line, which is the
  objective of the standard benchmark for that environment.
]

#question[
  How does the distribution of skills change during training?
][
  The authors tracked the performance of the learned skills throughout the
  training process on the Inverted Pendulum and Mountain Car tasks. At different
  training epochs, they evaluated every learned skill using the environment's
  true (but hidden from the agent) reward function.
][
  *Metrics:* The results were visualized as a *stacked bar chart* showing the
  number of skills that achieved a reward above or below a certain threshold
  over training epochs. The experiment was repeated across 5 random seeds to
  test for stability.

  *Results:* The plots show that the skills become *increasingly diverse* in
  their performance on the downstream task as training progresses. Initially,
  most skills perform poorly, but over time, more skills emerge that can solve
  the task and achieve a high reward.
][
  This demonstrates that the learning process is stable and progressive. The
  agent doesn't just discover one set of skills and stop; it continuously
  refines and diversifies them, improving the overall quality and range of the
  skill set over time.
]

#question[
  Does discriminating on single states restrict DIAYN to learn skills that visit
  disjoint sets of states?
][
  A simple environment was created with a narrow hallway leading to a large open
  room. The agent starts in the hallway. The question is whether skills can
  learn to pass through the same set of states (the hallway) to become
  distinguishable later.
][
  *Metrics:* The results were evaluated via *visual inspection* of the learned
  skill trajectories.

  *Results:* The agent learned skills that all successfully navigated the narrow
  hallway before branching out into different, distinguishable regions of the
  open room.
][
  This result is important because it shows DIAYN is not limited to simply
  partitioning the state space. It can learn skills that share common
  "bottleneck" states, a crucial capability for solving complex tasks that
  require navigating through specific points.

  *Limitations:* This was a simple, illustrative 2D experiment. While it
  demonstrates the principle, it doesn't guarantee the same behavior in much
  higher-dimensional or more complex environments.
]

#question[
  How does DIAYN differ from Variational Intrinsic Control (VIC)?
][
  This ablation study compared DIAYN against VIC, with the key difference being
  that DIAYN uses a *fixed, uniform prior over skills* ($p(z)$), while VIC
  learns this distribution. The comparison was run on the Half Cheetah
  environment.
][
  *Metrics:* The core metric was the *effective number of skills*, calculated as
  the exponent of the entropy of the skill distribution ($e^cal(H)[Z]$). A
  higher value means the agent is utilizing a more diverse set of skills.

  *Results:* The effective number of skills for VIC quickly collapsed to a very
  small number, indicating it was only sampling a handful of skills. In
  contrast, DIAYN maintained a high and constant number of effective skills
  throughout training because its prior was fixed.
][
  This highlights a critical and successful design choice in DIAYN. By not
  learning the prior, DIAYN avoids the "Matthew Effect" (where a few good skills
  get all the attention) and prevents the mode collapse that limits the
  diversity of skills learned by prior methods.
]

=== Harnessing Learned Skills

#question[
  Can we use learned skills to directly maximize the task reward?
][
  This experiment tested DIAYN as a *pre-training* mechanism. First, skills were
  learned without reward. Then, the single skill that performed best on a
  specific task was selected, and its network weights were used to initialize a
  new agent. This agent was then fine-tuned on the task reward and compared
  against a standard agent trained from a random initialization.
][
  *Metrics:* *Task reward vs. training hours* for the fine-tuning phase.

  *Results:* On all tested environments (Cheetah, Hopper, Ant), the agent
  initialized with DIAYN's pretrained skill learned much faster and achieved
  better final performance than the randomly initialized agent.
][
  This shows that DIAYN provides an effective method for unsupervised
  pre-training. It can accelerate learning on downstream tasks, which is
  especially useful in settings where interacting with the environment is cheap
  but obtaining a reward signal is expensive.

  *Limitations:* The analysis omits the time spent on unsupervised pre-training
  from the plots, assuming it can be amortized across many tasks.
]

#question[
  Are skills discovered by DIAYN useful for hierarchical RL?
][
  The discovered skills were used as primitives in a hierarchical setup. The
  low-level skill policies were frozen, and a high-level "meta-controller"
  policy was trained to select which skill to execute. This hierarchical agent
  was tested on two challenging tasks with sparse rewards (Half Cheetah Hurdle,
  Ant Navigation) and compared against strong, non-hierarchical baselines like
  SAC, TRPO, and VIME.
][
  *Metrics:* *Task reward vs. training hours*.

  *Results:* The DIAYN-based hierarchical agent was the only method to solve the
  tasks, significantly outperforming all baselines. It learned to have the
  cheetah jump over hurdles and the ant navigate to a sequence of waypoints,
  feats the other algorithms failed to achieve.
][
  This is a powerful demonstration of DIAYN's ability to combat challenges of
  exploration and sparse rewards. The discovered skills act as meaningful
  temporal abstractions, simplifying complex problems into a higher-level
  decision-making process.

  *Limitations:* The hierarchical framework is simple, using a fixed time
  duration ($k$) for each selected skill. More sophisticated hierarchical models
  could potentially yield even better performance.
]

#question[
  How can DIAYN leverage prior knowledge?
][
  The authors proposed a simple modification to DIAYN where the discriminator is
  conditioned on a manually chosen function of the state, $f(s)$, instead of the
  full state $s$. For the Ant Navigation task, they set $f(s)$ to be the ant's
  center of mass, thereby encouraging the discovery of skills that explicitly
  move the agent around.
][
  *Metrics:* Performance (reward vs. time) on the Ant Navigation hierarchical
  task.

  *Results:* The version with the injected prior knowledge "DIAYN+prior" learned
  faster and achieved a higher final reward than the standard DIAYN that used
  the full state.
][
  This shows that while DIAYN is primarily an unsupervised algorithm, it has a
  simple mechanism for incorporating domain knowledge when available. This
  flexibility allows a user to guide skill discovery towards more relevant
  behaviors for a specific task.
]

#question[
  Can we use learned skills to imitate an expert?
][
  This experiment explored using DIAYN for imitation from observations. Given an
  expert's trajectory of states (no actions), the learned discriminator is used
  to find which of the learned skills is most likely to produce that trajectory.
  The policy for that skill is then executed. This was evaluated qualitatively
  on Half Cheetah and quantitatively against ablations on classic control tasks.
][
  *Metrics:* *Visual comparison* for qualitative tests and *L2 distance in state
  space* between the expert and imitation trajectories for quantitative tests.

  *Results:* DIAYN could successfully imitate several expert behaviors like
  flipping over. The quantitative results showed that the standard DIAYN setup
  performed best, and that the discriminator's confidence score was a reliable
  predictor of the final imitation quality.
][
  This presents a method for one-shot imitation learning that does not require
  expert actions, relying instead on finding the closest behavior within the
  agent's pre-learned repertoire.

  *Limitations:* The method's success is fundamentally limited by the diversity
  of the learned skill set. If the agent has not learned a skill that is similar
  to the expert's behavior (e.g., a handstand), it cannot imitate it.
]
