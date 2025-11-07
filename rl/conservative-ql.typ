#import "../styles/things.typ": challenge, hypothesis, question

= Conservative Q-Learning for Offline RL

== Overview

An overview of the paper "Conservative Q-Learning for Offline Reinforcement
Learning":

=== Overview

This paper introduces *Conservative Q-Learning (CQL)*, a novel algorithm for
offline reinforcement learning (RL). It addresses the primary challenge of
learning effective policies from static datasets without environment
interaction.

#challenge[
  Q-Value Overestimation
][
  Standard off-policy RL algorithms often fail in the offline setting because of
  *distributional shift*. The learned policy starts to favor out-of-distribution
  (OOD) actions—actions not well-represented in the static dataset—leading to
  erroneously high, overestimated Q-values for these actions. Without
  interaction, the agent cannot correct these errors by observing the true
  outcomes.

  #hypothesis[
    The overestimation problem can be solved by learning a *conservative*
    Q-function that systematically provides a *lower bound* on the true value of
    the policy. This prevents the policy from exploiting incorrectly optimistic
    OOD actions.
  ]

  CQL modifies the standard Bellman error objective by adding a *regularizer*.
  This regularizer has two components:
  1. It minimizes the Q-values of actions sampled from a proposal distribution
    $mu(a|s)$ (e.g., actions that the current policy believes are optimal).
  2. It maximizes the Q-values of actions sampled from the dataset's behavior
    policy distribution $hat(pi)_beta(a|s)$.
  - The combined effect is to push down the values of potentially unseen, OOD
    actions while pushing up the values of actions known to be in the data,
    thereby creating a conservative value estimate. This is referred to as a
    *"gap-expanding"* property.
]

#question[
  How does CQL perform against state-of-the-art offline RL methods across
  diverse and complex environments?
][
  The primary actor-critic variant, *CQL(H)*, was benchmarked against prior
  offline RL algorithms (*BEAR*, *BRAC*), an online RL algorithm adapted for the
  offline setting (*SAC*), and *Behavioral Cloning (BC)*. The evaluation was
  conducted on the *D4RL benchmark*, which includes a wide range of tasks:
  - *Gym domains* (e.g., Hopper, Walker2d) with datasets generated from single
    policies (random, medium, expert) and more complex, mixed policies.
  - High-dimensional, difficult tasks like the *Adroit* robotic hand
    manipulation (using human demonstration data), *AntMaze* navigation, and
    *Franka Kitchen* sequential manipulation.
][
  *Normalized Return/Score*: The primary metric, averaged over 4 random seeds to
  ensure robustness. On simple datasets from a single policy, CQL's performance
  was comparable to or slightly better than the best prior methods. However, on
  complex, multi-modal datasets (e.g., "medium-expert," "-mixed"), *CQL
  significantly outperformed prior methods, often by a margin of 2-5x*. This is
  a crucial result, as these datasets better reflect realistic scenarios where
  data comes from multiple sources. For the most challenging Adroit, AntMaze,
  and Kitchen tasks, CQL was frequently the *only algorithm to achieve
  meaningful, non-zero returns* or to outperform simple behavioral cloning,
  demonstrating its superior capability in complex, sparse-reward, and
  high-dimensional settings.
][
  The evaluation, while extensive, is confined to benchmark environments.
  Performance on novel, real-world problems is not guaranteed.
]

=== Proposed Component: Conservative Q-Learning (CQL)

- *Description*: CQL is not a standalone algorithm but an *algorithmic
  framework* in the form of a regularized objective function. It can be
  implemented by adding a few lines of code to existing deep Q-learning (e.g.,
  QR-DQN) or actor-critic (e.g., SAC) algorithms. The CQL objective is given by:

$
  min_Q alpha dot.op(
    EE_(s ~ cal(D), a ~ mu(a|s)) [Q(s, a)] - EE_(s ~ cal(D), a ~ hat(pi)_beta (a|s)) [Q(s, a)]
  ) +/12 EE_(s, a ~ cal(D)) [(Q(s, a) - hat(cal(B))^pi hat(Q)^k (s, a))^2 ]
$

where the first term is the CQL regularizer and the second is the standard
Bellman error.
- *Inputs*:
  - A static dataset $cal(D)$ of $(s, a, r, s')$ transitions collected by a
    behavior policy $pi_beta$.
  - A Q-function $Q_theta$ and an optional policy $pi_phi$ to be trained.
- *Outputs*:
  - A conservatively learned Q-function $Q_theta$ where the expected value of a
    policy under it lower-bounds the policy's true value.
  - An improved policy $pi_phi$ optimized using this conservative Q-function.

=== Dependencies

- *Environments*:
  - *D4RL Benchmark*: A collection of datasets for data-driven RL.
    - Gym locomotion tasks (Hopper, Walker2d, HalfCheetah).
    - Adroit robotic hand manipulation tasks (Pen, Hammer, Door, Relocate).
    - AntMaze navigation tasks.
    - Franka Kitchen robot manipulation tasks.
  - *Arcade Learning Environment (ALE)*: Atari games including Pong, Breakout,
    Q*bert, Seaquest, and Asterix.
- *Datasets*:
  - The various datasets provided by the *D4RL benchmark*, which include data
    from random, medium, expert, and mixed policies.
  - *DQN-replay dataset* from Agarwal et al. (2019) for Atari experiments, using
    the 1%, 10%, and first 20% data splits.
- *Baseline Algorithms (for implementation)*:
  - *Soft Actor-Critic (SAC)*: The actor-critic variant of CQL for continuous
    control experiments is built upon an SAC implementation.
  - *Quantile Regression DQN (QR-DQN)*: The Q-learning variant of CQL for
    discrete Atari experiments is built upon a QR-DQN implementation.

=== Key Assumptions

- *Function Approximation (Theory)*: The theoretical analysis for non-linear
  function approximators (i.e., deep neural networks) relies on the *Neural
  Tangent Kernel (NTK) framework*. This framework formally holds in the
  infinite-width limit, which is an approximation for the finite-sized networks
  used in practice. The authors note a rigorous analysis of deep neural nets as
  future work.
- *Safe Policy Improvement*: The safe policy improvement guarantees are formally
  proven for the *empirical MDP* ($hat(M)$), which is the MDP induced directly
  by the transitions in the finite dataset $cal(D)$. Bounds relating performance
  in $hat(M)$ to the true MDP $M$ depend on sampling error, which can be large
  for sparsely visited states.

== Problem Formulation

The paper addresses the challenge of learning effective policies in *offline
reinforcement learning*. This setting is defined by learning from a fixed,
static dataset without any further interaction with the environment.

=== Reinforcement Learning Preliminaries

The problem is modeled as a *Markov Decision Process (MDP)*, defined by the
tuple $(cal(S), cal(A), T, r, gamma)$:
- $cal(S)$: The state space.
- $cal(A)$: The action space.
- $T(s'|s, a)$: The state transition dynamics function.
- $r(s, a)$: The reward function.
- $gamma$: The discount factor, where $gamma in (0, 1)$.

The goal is to find a policy $pi(a|s)$ that maximizes the expected discounted
cumulative reward. The value of a policy is given by its *Q-function* and
*V-function*:
- *Q-function*:
  $Q^pi(s, a) = EE_pi, T[sum_t=0^infinity gamma^t r(s_t, a_t) | s_0=s, a_0=a]$.
- *V-function*: $V^pi(s) = EE_a ~ pi(dot|s)[Q^pi(s, a)]$.

Standard off-policy algorithms, like Q-learning, train a Q-function by
iteratively applying the *Bellman optimality operator*, $cal(B)^*$:

$
  cal(B)^* Q(s, a) = r(s, a) + gamma EE_(s' ~ T(s'|s, a)) [max_(a') Q(s', a')]
$

=== The Offline RL Challenge: Distributional Shift

In offline RL, we are given a static dataset $cal(D) = (s, a, r, s')$ collected
by some unknown *behavior policy* $pi_beta$. The core challenge is the
*distributional shift* between the behavior policy $pi_beta$ and the learned
policy $pi$. When the learned policy $pi$ queries the value of actions that are
out-of-distribution (OOD) with respect to the dataset, the Q-function,
represented by a neural network, can produce arbitrarily high, erroneous values.
Because the agent cannot gather new data to correct these errors, it may learn
to exploit them, leading to poor performance at test time.

=== The CQL Formulation

CQL addresses this by learning a *conservative Q-function* that provides a lower
bound on the true policy value, thereby preventing overestimation for OOD
actions. This is achieved by augmenting the standard Bellman error objective
with a novel regularizer.

The primary objective for the actor-critic version of CQL, denoted *CQL(H)*, is:

$
  min_Q alpha dot.op(
    EE_(s ~ cal(D)) [log sum_a exp(Q(s, a))] - EE_(s ~ cal(D), a ~ hat(pi)_beta (a|s)) [Q(s, a)]
  ) + 1/2 EE_(s, a, s' ~ cal(D)) [(Q(s, a) - hat(cal(B))^pi hat(Q)^k (s, a))^2 ]
$

(Equation 4)

This objective consists of two main parts:
1. *Bellman Error Term*: The term
  $1/2 EE_s,a,s' ~ cal(D) [ (Q(s,a) - hat(cal(B))^pi hat(Q)^k(s,a))^2 ]$ is the
  standard mean squared Bellman error, which ensures that the learned Q-values
  are consistent with the transitions observed in the dataset $cal(D)$.
2. *CQL Regularizer*: The term $alpha dot ( dots )$ is the key innovation of
  CQL.
  - The first part of the regularizer,
    $EE_s ~ cal(D) [ log sum_a exp(Q(s, a)) ]$, acts as a soft maximum over the
    Q-values for all possible actions at a given state. Minimizing this term
    *pushes down* the Q-values of actions that the current policy believes are
    optimal, which are often OOD actions with erroneously high values.
  - The second part, $- EE_s ~ cal(D), a ~ hat(pi)_(beta(a|s))[Q(s, a)]$, when
    minimized, effectively *pushes up* the Q-values for actions that were
    actually taken in the dataset.
  - The parameter $alpha$ controls the strength of this regularization.

Together, these terms force the model to learn conservative Q-values, creating a
"gap" where Q-values for in-distribution actions are systematically higher than
those for OOD actions.

== Implementation Pipeline (CQL Actor-Critic)

This pipeline describes a practical implementation of the *CQL(H)* variant built
on top of the Soft Actor-Critic (SAC) algorithm, as used for the continuous
control experiments in the paper.

=== Initialization

- *Description*: This stage involves setting up the necessary neural networks,
  target networks, and optimizers before training begins. The architecture
  follows the twin Q-function trick from SAC to mitigate overestimation bias.
- *Inputs*: None (configuration only).
- *Outputs*:
  - *Q-Networks* ($Q_theta_1$, $Q_theta_2$): Two neural networks that map a
    (state, action) pair to a scalar Q-value.
  - *Target Q-Networks* ($Q_theta'_1$, $Q_theta'_2$): Two separate networks,
    initialized as clones of the main Q-networks, used to provide stable targets
    for the Bellman update.
  - *Policy Network* ($pi_phi$): A neural network that maps a state to the
    parameters of a distribution over the action space (e.g., mean and variance
    for a Gaussian).
  - *Hyperparameters*: learning rates $eta_Q, eta_pi$, trade-off weight $alpha$
    (or parameters for its automatic tuning via a Lagrange multiplier).
- *Tensor Shapes*:
  - Input State $s$: `(batch_size, state_dim)`
  - Input Action $a$: `(batch_size, action_dim)`
  - Q-Network Output: `(batch_size, 1)`

=== Data Sampling

- *Description*: Inside the main training loop, this stage samples a random
  mini-batch of transitions from the static, offline dataset $cal(D)$.
- *Inputs*: The offline dataset $cal(D)$.
- *Outputs*: A mini-batch of transitions `(s, a, r, s', d)`, where `d` is a done
  flag.
- *Tensor Shapes*:
  - `s`, `s'`: `(batch_size, state_dim)`
  - `a`: `(batch_size, action_dim)`
  - `r`, `d`: `(batch_size, 1)`

=== Q-Function Update (The CQL Core)

- *Description*: This is the most critical stage, where the Q-networks are
  updated using the composite loss function from *Equation 4*.
- *Inputs*: The sampled mini-batch, all current networks
  ($Q_theta_1, Q_theta_2, pi_phi$), and the target Q-networks
  ($Q_theta'_1, Q_theta'_2$).
- *Outputs*: Updated parameters $theta_1, theta_2$ for the Q-networks.
- *Pipeline*:
  1. *Compute Bellman Target*: Calculate the target value $y$ for the Bellman
    error term, following the SAC procedure.
    - Sample next actions $a'$ and their log-probabilities $log pi_phi(a'|s')$
      from the current policy $pi_phi$ using the next states $s'$ from the
      batch.
    - Compute the target Q-value using the minimum of the two target networks:
      $Q_"target" = min(Q_theta'_1(s', a'), Q_theta'_2(s', a')) - alpha_"entropy" log pi_phi(a'|s')$.
    - The final Bellman target is: $y = r + gamma(1 - d)Q_"target"$.
  2. *Compute Bellman Loss*: $L_"Bellman" = "MSE"(Q_theta_i(s, a), y)$ for each
    Q-network ($i in 1, 2$).
  3. *Compute CQL Regularizer Loss*: This is the term added by CQL.
    - For each state $s$ in the batch, sample $N$ actions from a uniform
      distribution and $N$ actions from the current policy $pi_phi(dot|s)$.
    - Use these samples to compute the `log-sum-exp` term via importance
      sampling. Let this be $Q_"LSE"$.
    - Get the Q-values for the actions from the dataset batch:
      $Q_"data" = Q_theta_i(s, a)$.
    - The regularizer loss is: $L_"CQL" = alpha dot (Q_"LSE" - Q_"data")$.
  4. *Compute Total Q-Loss*: The final loss for each Q-network is
    $L_Q_i = L_"Bellman" + L_"CQL"$.
  5. *Perform Gradient Step*: Update the parameters $theta_1$ and $theta_2$ by
    performing one step of gradient descent on their respective losses.

=== Policy and Target Network Updates

- *Description*: This stage updates the actor (policy network) and the
  slowly-moving target networks.
- *Inputs*: The updated Q-networks and the current policy network.
- *Outputs*: Updated parameters $phi$ for the policy network and
  $theta'_1, theta'_2$ for the target networks.
- *Pipeline*:
  1. *Policy Update*: The policy is updated to maximize the conservative
    Q-values provided by the newly trained Q-networks.
    - Sample actions $a_"new"$ and log-probabilities from the policy $pi_phi$
      for the batch of states $s$.
    - Compute the policy loss:
      $L_pi = EE_s ~ cal(D)[alpha_"entropy" log pi_phi(a_"new"|s) - min(Q_theta_1(s, a_"new"), Q_theta_2(s, a_"new"))]$.
    - Perform a gradient step on $L_pi$ to update $phi$. The paper notes using a
      low learning rate for the policy (e.g., 3e-5) is crucial for stability.
  2. *Target Network Update*: The target network parameters are updated to be a
    weighted average of their current and the main Q-network's parameters (a
    "soft" update), which helps stabilize training.
    - $theta'_i <- tau theta_i + (1 - tau) theta'_i$ for $i in 1, 2$, where
      $tau$ is a small constant (e.g., 0.005).

== Discussion

Here is a detailed outline of the main questions investigated in the paper's
results and discussion sections.

#question[
  How does CQL perform against state-of-the-art offline RL methods across
  diverse and complex environments?
][
  The primary actor-critic variant, *CQL(H)*, was benchmarked against prior
  offline RL algorithms (*BEAR*, *BRAC*), an online RL algorithm adapted for the
  offline setting (*SAC*), and *Behavioral Cloning (BC)*. The evaluation was
  conducted on the *D4RL benchmark*, which includes a wide range of tasks:
  - *Gym domains* (e.g., Hopper, Walker2d) with datasets generated from single
    policies (random, medium, expert) and more complex, mixed policies.
  - High-dimensional, difficult tasks like the *Adroit* robotic hand
    manipulation (using human demonstration data), *AntMaze* navigation, and
    *Franka Kitchen* sequential manipulation.
][
  *Normalized Return/Score*: The primary metric, averaged over 4 random seeds to
  ensure robustness. On simple datasets from a single policy, CQL's performance
  was comparable to or slightly better than the best prior methods. However, on
  complex, multi-modal datasets (e.g., "medium-expert," "-mixed"), *CQL
  significantly outperformed prior methods, often by a margin of 2-5x*. This is
  a crucial result, as these datasets better reflect realistic scenarios where
  data comes from multiple sources. For the most challenging Adroit, AntMaze,
  and Kitchen tasks, CQL was frequently the *only algorithm to achieve
  meaningful, non-zero returns* or to outperform simple behavioral cloning,
  demonstrating its superior capability in complex, sparse-reward, and
  high-dimensional settings.
][
  The evaluation, while extensive, is confined to benchmark environments.
  Performance on novel, real-world problems is not guaranteed.
]

#question[
  Is the CQL framework effective in discrete action spaces with high-dimensional
  image inputs?
][
  A discrete-action, Q-learning variant of CQL was compared to *REM* and
  *QR-DQN* on offline Atari datasets. The experiments used data from a
  pre-trained DQN agent, focusing on data-scarce scenarios: (1) using the first
  20% of samples from an online run, and (2) using only 1% or 10% of the total
  samples.
][
  *Raw Game Score*: The standard metric for Atari environments. CQL performed
  comparably or better than baselines in the standard data setting. In the
  low-data regimes (1% and 10%), *CQL substantially outperformed both REM and
  QR-DQN*, achieving, for example, a 36x higher return on Q*bert in the 1% data
  condition. This demonstrates that the core principles of CQL are general and
  highly effective even in data-limited, image-based, discrete-action domains,
  where function approximation errors can be severe.
][
  The evaluation was performed on a subset of five Atari games.
]

#question[
  Does CQL empirically learn a conservative Q-function that lower-bounds the
  true policy value?
][
  This was an analytical experiment to validate CQL's core mechanism. The
  authors measured the *difference between the policy value predicted by an
  algorithm and the true policy value* (evaluated via rollouts). They compared
  *CQL(H)*, a simpler variant of CQL (*CQL (Eqn. 1)*), *BEAR*, and Q-learning
  with *ensembles* of various sizes (a common technique to reduce
  overestimation).
][
  *Predicted Value Difference*: Defined as (Predicted Policy Value - True Policy
  Value). A negative result indicates the algorithm is conservative (i.e., it
  lower-bounds the true value). *CQL was the only method that consistently
  yielded a negative difference*, confirming it successfully learns a
  conservative lower bound on the policy's value. All baselines, including an
  ensemble of 20 Q-functions, showed *massive value overestimation*, with
  predicted values being orders of magnitude higher than the true returns. The
  experiment also showed that CQL(H) provides a *tighter* lower bound than the
  simpler CQL (Eqn. 1), which is theoretically desirable for learning better
  policies.
][
  This direct validation was performed on three D4RL Hopper datasets.
]

#question[
  What are the key design choices within the CQL algorithm and how do they
  impact performance?
][
  Ablation Study 1: Choice of Proposal Distribution (`CQL(H)` vs. `CQL(ρ)`). How
  does using a uniform prior (`CQL(H)`) compare to using the previous policy
  iterate as the prior (`CQL(ρ)`). `CQL(H)` performed better on most standard
  MuJoCo tasks. However, `CQL(ρ)` was more stable and effective on tasks with
  very high-dimensional action spaces (e.g., Adroit), where the importance
  sampling required for `CQL(H)` can suffer from high variance. Ablation Study
  2: Necessity of the In-Distribution Maximization Term. Is it necessary to both
  minimize OOD Q-values and maximize in-distribution Q-values? This compares
  `CQL(H)` to a variant based on Equation
  1.
  Removing the maximization term (i.e., using the simpler Equation 1) generally
  *decreased performance*. This empirically confirms the theoretical finding
  that this term is crucial for learning a tighter, more useful lower bound.
  Ablation Study 3: Fixed vs. Automated Regularization Weight (`α`). Does
  automatically tuning `alpha` via a Lagrange multiplier outperform using a
  fixed value? The Lagrange version that automatically tunes `alpha`
  *consistently performed better*, especially on complex tasks like the
  AntMazes, where the improvement was substantial. This highlights the benefit
  of adapting the regularization strength during training.
][
  These ablations cover key design choices but are not exhaustive across all
  possible hyperparameters and domains.
]

#question[
  What are the unresolved theoretical and practical limitations of CQL?
][
  *Theoretical Gap with Deep Nets*: The authors state that while their theory
  covers tabular, linear, and some non-linear function approximation cases
  (under NTK assumptions), a *rigorous theoretical analysis of CQL's behavior
  with deep neural networks remains an open problem* for future work. *Practical
  Overfitting*: As with any method trained on a fixed dataset, CQL is
  susceptible to overfitting. A significant challenge for future work is to
  develop *simple and effective early stopping methods for offline RL*,
  analogous to using a validation set in supervised learning, to know when to
  stop training.
][
  *Discussion & Limitations from the Authors*:
  - *Theoretical Gap with Deep Nets*: The authors state that while their theory
    covers tabular, linear, and some non-linear function approximation cases
    (under NTK assumptions), a *rigorous theoretical analysis of CQL's behavior
    with deep neural networks remains an open problem* for future work.
  - *Practical Overfitting*: As with any method trained on a fixed dataset, CQL
    is susceptible to overfitting. A significant challenge for future work is to
    develop *simple and effective early stopping methods for offline RL*,
    analogous to using a validation set in supervised learning, to know when to
    stop training.
]
