= Maximum Entropy RL (Provably) Solves Some Robust RL Problems

== Overview

This paper provides the first formal proof that *Maximum Entropy Reinforcement
Learning (MaxEnt RL)*, a common RL paradigm, inherently solves a class of robust
RL problems. It demonstrates that the MaxEnt RL objective maximizes a lower
bound on the performance of a policy under certain adversarial disturbances to
the environment's dynamics and reward function.

=== Challenges and Approaches

- *Challenge 1: Lack of Formal Proof for MaxEnt RL's Robustness*
  - *Description*: While prior work empirically observed and conjectured that MaxEnt
    RL learns robust policies, a rigorous theoretical proof was missing. Existing
    robust RL methods often required complex, bespoke solutions like adversarial
    training schemes.
  - *Hypothesis*: The stochastic policies learned by MaxEnt RL, which are encouraged
    to explore multiple pathways to a goal, are naturally resilient to perturbations
    that might block a single, optimal path.
  - *Approach*: The paper proves that maximizing the MaxEnt RL objective for a
    specific "pessimistic" reward function is equivalent to maximizing a lower bound
    on the standard reward objective under a worst-case adversarial perturbation to
    the dynamics. The proof uses duality theory (KKT conditions) to connect the
    robust objective to the entropy-regularized objective.
  - *Alternative Solutions Mentioned*: Prior methods often involve a two-player
    game, requiring the training of an additional adversary policy or solving an
    inner-loop optimization problem, which adds complexity and hyperparameters.

- *Challenge 2: Characterizing the Set of Disturbances*
  - *Description*: Without a formal proof, the exact set of environmental
    disturbances that MaxEnt RL is robust against was unknown.
  - *Hypothesis*: The set of disturbances (the "robust set") to which MaxEnt RL
    policies are resilient is non-trivial and can be formally characterized.
  - *Approach*: The paper formally derives the robust sets for both reward and
    dynamics perturbations. The dynamics robust set, $tilde(cal(P))$, is defined by
    a constraint on the expected divergence between the original dynamics ($p$) and
    the perturbed dynamics ($tilde(p)$). The size of this set, and thus the degree
    of robustness, is shown to be lower-bounded by the policy's entropy.

=== High-Level Component Description

The paper does not propose a new algorithm but rather provides a theoretical
analysis and a new perspective on an existing one: *Maximum Entropy
Reinforcement Learning*. The core finding is how to use a standard MaxEnt RL
algorithm to solve a robust RL problem.

- *Analyzed Method*: Using a standard MaxEnt RL algorithm (e.g., Soft
  Actor-Critic) to optimize a specifically constructed pessimistic reward
  function.
- *Inputs*:
  - A standard Markov Decision Process (MDP) with original dynamics $p(s_( t+1 )|s_t, a_t)$ and
    a target reward function $r(s_t, a_t)$.
  - An entropy coefficient, $alpha$, which controls the trade-off between reward and
    policy entropy.
- *Core Transformation (The "trick")*: To achieve a policy robust for a reward
  function $r(s_t, a_t)$, the agent should not be trained on $r$ directly.
  Instead, it should be trained using MaxEnt RL on a *pessimistic reward
  function*, $overline(r)$, defined as:

$
  overline(r)(s_t,a_t,s_( t+1 )) eq.delta 1/T log r(s_t,a_t) + cal(H)[s_( t+1 )|s_t,a_t]
$

where $cal(H)[s_( t+1 )|s_t,a_t]$ is the entropy of the original transition
dynamics.
- *Output*:
  - A stochastic policy, $pi(a_t|s_t)$, that maximizes the expected cumulative
    reward under worst-case adversarial dynamics chosen from the derived robust set $tilde(cal(P))$.

=== Dependencies for Reproduction

- *Algorithms/Libraries*:
  - *MaxEnt RL*: Soft Actor-Critic (SAC) implemented via TF Agents.
  - *Standard RL*: Twin Delayed Deep Deterministic Policy Gradient (TD3) and DDPG.
  - *Robust RL Baselines*: PR-MDP and NR-MDP from Tessler et al. (2019) .
  - *Optimization*: CMA-ES (for finding adversarial perturbations) and CVXPY (for
    bandit experiments).
- *Environments/Datasets*:
  - *OpenAI Gym*: HalfCheetah-v2, Walker2d-v2, Hopper-v2, Ant-v2, Pusher-v2.
  - *Meta-World*: SawyerButtonPressEnv.
  - *Custom*: A 2D navigation task from Eysenbach et al. (2019) and a peg insertion
    task from Eysenbach et al. (2018) .
- *Pre-trained Models*: None were used.

=== Additional Perspectives & Assumptions

- *Key Perspective Missing from Abstract*: The abstract suggests MaxEnt RL *is*
  robust. The paper clarifies the crucial nuance that to make a policy robust for
  a target reward function $r$, one must apply MaxEnt RL to a *different,
  pessimistic reward function* $overline(r)$. MaxEnt RL is not inherently robust
  with respect to the same reward function it is trained on.
- *Glaring Assumptions*:
  - *Knowledge of Dynamics*: The construction of the pessimistic reward $overline(r)$ requires
    computing the dynamics entropy $cal(H)[s_( t+1 )|s_t, a_t]$, which requires
    knowledge of or a model of the environment's dynamics. The authors acknowledge
    this is a limitation for model-free applications.
  - *Positive Rewards*: The theoretical derivation requires the target reward
    function $r(s_t, a_t)$ to be strictly positive to ensure the log transform is
    well-defined.
  - *Full Policy Support*: The proof of reward robustness (Theorem 4.1) assumes the
=== Recommended Prerequisite Knowledge

- *Haarnoja et al. (2018a).* Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor. Understanding SAC is crucial as
  it is the primary MaxEnt RL algorithm used in the paper's experiments to
  validate the theory.
- *Ziebart et al. (2008).* Maximum Entropy Inverse Reinforcement Learning. This
  paper is foundational for understanding the principle of maximum entropy in the
  context of learning behaviors, which is the core concept being analyzed.

== Problem Formulation

=== 1. Standard Reinforcement Learning (RL)

The standard goal in RL is to find a policy, $pi$, that maximizes the expected
cumulative reward.

- *Trajectory*: A sequence of states and actions over $T$ steps, denoted as $tau eq.delta(s_1, a_1, ..., s_T, a_T)$.
- *Trajectory Distribution*: The probability of a trajectory under policy $pi$ and
  dynamics $p$ is given by:

$
  p^pi (tau) = p_1 (s_1) product_(t = 1)^T p(s_(t + 1)|s_t, a_t) pi(a_t |s_t)
$

- *Standard RL Objective*:

$
  arg max_pi EE_(tau ~ p^pi (tau)) [sum_(t = 1)^T r(s_t, a_t)]
$

=== 2. Maximum Entropy Reinforcement Learning (MaxEnt RL)

MaxEnt RL augments the standard objective with a policy entropy term,
encouraging exploration and stochasticity.

- *Policy Entropy*: The entropy of the action distribution at a given state is:

$
  cal(H)_pi [a_t |s_t ] = integral_cal(A) pi(a_t |s_t) log 1/(pi(a_t |s_t)) d a_t
$

- *MaxEnt RL Objective*: The objective is to maximize a weighted sum of the
  expected reward and the expected policy entropy, where $alpha$ is the entropy
  coefficient.

$
  J_"MaxEnt" (pi; p, r) eq.delta EE_(a_t ~ pi(a_t |s_t) \ s_(t + 1) ~ p(s_(t + 1)|s_t, a_t)) [sum_(t = 1)^T r(s_t, a_t) + alpha cal(H)_pi [a_t |s_t ]]
$

=== 3. Robust Reinforcement Learning

Robust RL aims to find a policy that performs well even under the worst-case
perturbations to the environment's dynamics or reward function, drawn from
predefined sets.

- *General Robust RL Objective*: The agent (policy $pi$) plays against an
  adversary that chooses the dynamics $tilde(p)$ and reward function $tilde(r)$ from
  robust sets $tilde(cal(P))$ and $tilde(cal(R))$ respectively to minimize the
  return.

$
  max_pi min_(tilde(p) in tilde(cal(P)) \ tilde(r) in tilde(cal(R))) EE_(a_t ~ pi(a_t |s_t) \ s_(t + 1) ~ tilde(p)(s_(t + 1)|s_t, a_t)) [sum_(t = 1)^T tilde(r)(s_t, a_t)]
$

=== 4. Problem Formulation for the Paper's Core Claim

The paper's central thesis connects MaxEnt RL to robust RL. It shows that
solving a MaxEnt RL problem with a specific *pessimistic* reward function is
equivalent to solving a robust RL problem.

- *Pessimistic Reward Function*: To learn a policy that is robust for a target
  reward $r(s_t, a_t)$ under dynamics perturbations, one must use MaxEnt RL with
  the following reward function:

$
  overline(r)(s_t, a_t, s_(t + 1)) eq.delta 1/T log r(s_t, a_t) + cal(H) [s_(t + 1)|s_t, a_t ]
$

where $cal(H)[s_( t+1 )|s_t, a_t]$ is the entropy of the *original* dynamics.

- *Robust Set for Reward Perturbations ($tilde(cal(R))$)*: The set of adversarial
  reward functions $tilde(r)$ that are "close" to the original reward $r$.

$
  tilde(cal(R))(pi) eq.delta { tilde(r)(s_t, a_t) bar EE_pi [sum_t log integral_cal(A) exp(r(s_t, a'_t ) - tilde(r)(s_t, a'_t)) d a'_t ] <= epsilon.alt}
$

- *Robust Set for Dynamics Perturbations ($tilde(cal(P))$)*: The set of
  adversarial dynamics $tilde(p)$ constrained by a divergence metric $d(p, tilde(p); tau)$.
- *Divergence Metric*:

$
  d(p, tilde(p); tau) eq.delta sum_(s_t in tau) log integral.double_(cal(A) times cal(S)) (p(s'_(t + 1) bar s_t, a'_t))/(tilde(p)(s'_(t + 1) bar s_t, a'_t)) d a'_t d s'_(t + 1)
$

- *Robust Set Definition*:

$
  tilde(cal(P))(pi) eq.delta {tilde(p)(s'|s, a)|EE_(p^pi (tau)) [d(p, tilde(p); tau)] <= epsilon.alt}
$

- *Main Result (Theorem 4.2, simplified)*: The MaxEnt RL objective with the
  pessimistic reward $overline(r)$ provides a lower bound on the robust RL
  objective where the adversary perturbs the dynamics.

$
  min_(tilde(p) in tilde(cal(P))(pi)) J(pi; tilde(p), r) >= exp(J_"MaxEnt" (pi; p, overline(r)) + log T)
$

where $J(pi; tilde(p), r)$ is the standard expected return under the perturbed
dynamics $tilde(p)$.

== Pipeline

Of course. Based on my analysis of the paper and your preference for technical
detail, here is a detailed pipeline explaining the project's implementation,
structured in stages with descriptions of inputs and outputs. All mathematical
notation uses LaTeX as you requested.

The core implementation validates the paper's theory by training RL agents and
then evaluating their robustness to environmental perturbations.

=== Stage 1: Environment and Perturbation Definition

This stage involves setting up the simulation environment and defining the
specific disturbances that will be used to test for robustness during
evaluation.

- *Inputs*:
  - A chosen simulation environment from benchmarks like OpenAI Gym or Meta-World.
  - A specific perturbation type to be applied during the evaluation phase. The
    paper explores several, including:
    - *Static Changes*: Modifying object masses for the training environment.
    - *New Obstacles*: Adding new objects not present during training.
    - *Dynamic Perturbations*: Applying an external force to an object mid-episode.
    - *Adversarial Perturbations*: Finding the worst-case change to a parameter (e.g.,
      goal location) using an optimization algorithm like CMA-ES.

- *Description*: The environment is formalized as a *Markov Decision Process
  (MDP)*. A standard environment provides the state space $cal(S)$, action space $cal(A)$,
  the original transition dynamics $p(s_( t+1 ) | s_t, a_t)$, and the reward
  function $r(s_t, a_t)$. The perturbation mechanism defines how the dynamics are
  altered to $tilde(p)(s_( t+1 )|s_t, a_t)$ during the evaluation stage.

- *Outputs*:
  - *Training Environment*: An instance of the MDP with the original, unperturbed
    dynamics $p$.
  - *Evaluation Environment*: An instance of the MDP with the perturbed dynamics $tilde(p)$.

=== Stage 2: Agent and Network Initialization

Here, the learning agent is defined based on the chosen algorithm (e.g., MaxEnt
RL, standard RL) and its components, like neural networks, are initialized. The
paper primarily uses *Soft Actor-Critic (SAC)* for MaxEnt RL.

- *Inputs*:
  - *Algorithm Choice*: e.g., SAC for MaxEnt RL or TD3 for standard RL.
  - *Hyperparameters*: Learning rates, replay buffer size, discount factor ($gamma$),
    and for SAC, the entropy coefficient ($alpha$).
  - *Environment Specs*: State dimension $D_s = abs(cal(S))$ and action dimension $D_a = abs(cal(A))$.

- *Description*: For an actor-critic method like SAC, this involves initializing
  the parameters for the policy network and the Q-value networks.
  - *Policy Network (Actor)*: $pi_theta(a_t|s_t)$, which maps a state to a
    distribution over actions.
  - *Q-Value Networks (Critic)*: $Q_( phi_1 )(s_t, a_t)$ and $Q_( phi_2 )(s_t, a_t)$,
    which estimate the expected return from a state-action pair. Target networks ($pi_( theta_"target" )$, $Q_( phi_( 1, "target" ) )$, $Q_( phi_( 2, "target" ) )$)
    are also created as lagging copies for stable learning.

- *Outputs*:
  - An initialized agent, consisting of:
    - *Actor Network Weights*: $theta$. For a simple multi-layer perceptron (MLP) with
      one hidden layer of size $H$ (e.g., $H=256$ ), the weight matrices would have
      shapes: $W_1: (D_s, H)$, $W_2: (H, D_a)$.
    - *Critic Network Weights*: $phi_1, phi_2$. For a similar MLP architecture, the
      input is a state-action pair, so the first weight matrix shape is: $W_1: (D_s + D_a, H)$.
      The output is a single Q-value, so the last weight matrix shape is $W_2: (H, 1)$.
  - An empty *Replay Buffer* with a fixed capacity.

=== Stage 3: Agent Training Loop

The agent interacts with the *unperturbed* training environment to learn a
policy.

- *Inputs*:
  - The initialized agent from Stage 2.
  - The training environment from Stage 1.
  - Number of training steps or episodes.

- *Description*: This is an iterative process:
  1. The agent observes state $s_t$ and samples an action $a_t ~ pi_theta(a_t|s_t)$.
  2. The environment returns the next state $s_( t+1 )$, reward $r_t$, and a
    termination signal $d_t$.
  3. The transition tuple $(s_t, a_t, r_t, s_( t+1 ), d_t)$ is stored in the replay
    buffer.
  4. A mini-batch of transitions is sampled from the replay buffer.
    - *Input Batch Tensors*: $S: (B, D_s)$, $A: (B, D_a)$, $R: (B, 1)$, $S': (B, D_s)$,
      where $B$ is the batch size.
  5. The critic networks are updated by minimizing the soft Bellman error. The loss
    function for each critic $phi_i$ is:

$
  L(phi.alt_i) = EE_((s, a, r, s') ~ "Buffer") [(Q_(phi.alt_i)(s, a) - y(r, s'))^2 ]
$

where the target value $y$ is:

$
  y(r, s') = r + gamma(
    min_(j = 1, 2) Q_(phi.alt_(j, "target"))(s', a') - alpha log pi_theta (a'|s')
  ), quad a' ~ pi_theta (dot.op|s')
$

6. The actor network is updated by maximizing the MaxEnt objective:

$
  L(theta) = EE_(s ~ "Buffer") [alpha log pi_theta (a'|s) - min_(j = 1, 2) Q_(phi.alt_j)(s, a')], quad a' ~ pi_theta (dot.op|s)
$

7. The target networks are updated via a soft update (polyak averaging).

- *Outputs*:
  - *Trained Policy Network*: The optimized weights $theta^*$ of the actor network.

=== Stage 4: Evaluation in Perturbed Environment

The learned policy is now tested in the *perturbed* environment to measure its
robustness.

- *Inputs*:
  - The trained policy network ($theta^*$) from Stage 3.
  - The evaluation environment (with dynamics $tilde(p)$) from Stage 1.
  - Number of evaluation episodes (e.g., 30 episodes ).

- *Description*: For a fixed number of episodes, the agent interacts with the
  perturbed environment.
  1. At each timestep $t$, the agent observes state $s_t$ from the perturbed
    environment.
  2. It selects an action deterministically or by sampling from its learned policy: $a_t = pi_( theta^* )(s_t)$.
  3. The environment transitions to $s_( t+1 )$ according to the *perturbed dynamics* $tilde(p)(s_( t+1 ) | s_t, a_t)$.
  4. The cumulative reward for the episode is recorded. No learning or network
    updates occur during this phase.

- *Outputs*:
  - A set of cumulative rewards, one for each evaluation episode.
    - *Reward Vector*: Shape $(N_"eval", 1)$, where $N_"eval"$ is the number of
      evaluation episodes

=== Stage 5: Performance Analysis

The results from multiple agents and across different perturbation levels are
aggregated and visualized.

- *Inputs*:
  - The reward vectors from Stage 4 for each trained agent (e.g., MaxEnt RL,
    standard RL, baselines).
  - The corresponding perturbation level for each evaluation (e.g., relative mass
    from 0.5 to 2.0 ).

- *Description*: The mean and standard deviation of the cumulative rewards are
  calculated for each agent at each perturbation level. These statistics are then
  plotted to compare the performance and robustness of the different methods.

- *Outputs*:
  - *Plots and Figures*: Visualizations comparing the performance of different
    algorithms as a function of the environmental perturbation, such as Figure 3 in
    the paper. These plots serve as the empirical evidence for the paper's claims.

== Discussion

=== 1. Is MaxEnt RL competitive with specialized robust RL algorithms?

- *Experiment Design*
  - *Goal*: To benchmark MaxEnt RL against methods explicitly designed for
    robustness.
  - *Algorithms Compared*:
    - *MaxEnt RL (SAC)* with a small entropy coefficient ($alpha = 0.0001$) and a
      large one ($alpha = 0.1$).
    - *Standard RL*: DDPG.
    - *Robust RL Baselines*: PR-MDP and NR-MDP from Tessler et al. (2019) .
  - *Environments*: HalfCheetah-v2, Walker2d-v2, and Hopper-v2 from OpenAI Gym.
  - *Perturbation*: The agents were trained in a standard environment, but evaluated
    in versions where the agent's body mass was altered, plotted as "Relative mass".
- *Results*
  - MaxEnt RL with a *large entropy coefficient* performed competitively with, and
    in some cases better than, the purpose-built PR-MDP and NR-MDP methods.
  - MaxEnt RL with a *small entropy coefficient* and the standard DDPG baseline both
    performed poorly under these perturbations.
- *Metrics Used*
  - The primary metric was the average *cumulative reward* over 30 evaluation
    episodes, plotted against the relative mass of the agent.
- *Significance of Results*
  - This provides strong empirical evidence that MaxEnt RL, a conceptually simpler
    algorithm, can serve as a powerful and effective robust RL method without
    requiring the additional adversarial training components and hyperparameters of
    specialized algorithms. This validates the paper's central thesis in a practical
    setting.
- *Limitations*
  - The experiments were conducted on benchmarks proposed by the baseline methods'
    authors. However, the paper includes ablation studies in Appendix C, which show
    that simple upgrades to the baseline algorithms (e.g., larger networks, dual
    critics) do not significantly improve their performance, strengthening the main
    conclusion.

=== 2. Is MaxEnt RL simply equivalent to standard RL with added noise?

- *Experiment Design*
  - *Goal*: To determine if the robustness from MaxEnt RL is more sophisticated than
    simply injecting noise into a deterministic policy.
  - *Environment*: A 2D navigation task where the agent must reach a goal while
    avoiding costly "red regions". A new L-shaped obstacle was added during
    evaluation.
  - *Algorithms Compared*:
    - MaxEnt RL (SAC).
    - Standard RL (deterministic policy).
    - Standard RL with varying levels of Gaussian noise added to its actions ($sigma=0.3$ and $sigma=1$).
- *Results*
  - The MaxEnt RL policy learned to have low entropy (low noise) when near the red
    obstacles to avoid costs, and higher entropy later on. It successfully navigated
    around the new obstacle.
  - The standard RL policy always collided with the new obstacle.
  - Adding noise to the standard RL policy either was insufficient to avoid the new
    obstacle (low noise) or caused the agent to frequently enter the costly red
    regions (high noise).
- *Metrics Used*
  - The primary metrics were the final *return* (reward) and a qualitative
    visualization of the agent's trajectories.
- *Significance of Results*
  - This experiment demonstrates that MaxEnt RL is not merely standard RL with
    injected noise. It learns to *dynamically adjust its stochasticity* based on the
    state, using high entropy to explore when it's safe and low entropy to be
    precise when required. This state-dependent entropy is key to its effective
    robustness.
- *Limitations*
  - This is an illustrative example in a single, relatively simple 2D environment.
    While it provides strong intuition, it doesn't generalize this finding across
    all possible tasks.

=== 3. How does the entropy coefficient ($alpha$) affect robustness to dynamic perturbations?

- *Experiment Design*
  - *Goal*: To test the theoretical prediction that the size of the robust set is
    determined by the policy's entropy, which is controlled by the hyperparameter $alpha$.
  - *Environment*: The Pusher task from Figure 1.
  - *Perturbation*: Instead of a static obstacle, a dynamic perturbation was
    applied: the puck's XY position was instantly shifted by a random amount after
    20 timesteps. The size of this shift ("disturbance to puck") was varied.
  - *Algorithms Compared*: MaxEnt RL (SAC) was trained with several different
    entropy coefficients: $alpha = {0, 1e-05, 1e-04, 1e-03}$.
- *Results*
  - All methods performed well with zero disturbance.
  - However, only the policy trained with the *largest entropy coefficient* ($alpha = 1e-03$)
    remained robust to larger disturbances. The policies with smaller $alpha$ values
    showed a sharp drop in performance as the disturbance size increased.
- *Metrics Used*
  - Average *reward* plotted as a function of the disturbance size.
- *Significance of Results*
  - This result provides direct empirical support for the theory presented in Lemma
    4.3. It confirms that practitioners can *increase the robustness* of the learned
    policy by using a larger entropy coefficient, which corresponds to enlarging the
    size of the robust set the policy is optimized for.
- *Limitations*
  - The experiment was performed on a single task with one type of dynamic
    perturbation. The relationship between $alpha$ and robustness might be more
    complex in other scenarios.

=== 4. Is MaxEnt RL robust to adversarial reward perturbations?

- *Experiment Design*
  - *Goal*: To empirically verify Theorem 4.1, which states that MaxEnt RL solves a
    robust RL problem for a specific set of adversarial reward functions.
  - *Environments*: Four continuous control tasks from OpenAI Gym: HalfCheetah-v2,
    Hopper-v2, Walker2d-v2, and Ant-v2.
  - *Perturbation*: The evaluation metric was changed from the standard cumulative
    reward to a *minimax reward*. The adversary chooses a reward function from the
    robust set $tilde(cal(R))(\pi)$ defined in Equation 2 to minimize the agent's
    score. The worst-case reward function has an analytical form: $tilde(r)(s_t, a_t) = r(s_t, a_t) - log pi(a_t|s_t)$.
  - *Algorithms Compared*:
    - *MaxEnt RL (SAC)*.
    - *Standard RL (SVG-0)*, a stochastic policy gradient method.
    - *Fictitious Play*, a method that explicitly trains against an adversarial reward
      chooser.
- *Results*
  - While both MaxEnt RL and standard RL were effective at maximizing the standard
    cumulative reward, *only MaxEnt RL succeeded in maximizing the worst-case
    (minimax) reward*.
  - Standard RL's minimax performance was very poor, while MaxEnt RL's was high and
    stable.
- *Metrics Used*
  - The *minimax reward*, plotted over the course of training steps. The paper also
    shows the standard cumulative reward for comparison.
- *Significance of Results*
  - This is a direct empirical validation of the paper's theoretical result on
    reward robustness (Theorem 4.1). It shows that MaxEnt RL is not just robust by
    chance, but that it formally optimizes for worst-case performance under a
    well-defined set of adversarial reward perturbations.
- *Limitations*
  - The experiment relies on an analytically computed worst-case reward function,
    which is possible because the robust set has a specific structure. For more
    general, arbitrary sets of reward functions, this evaluation would be more
    complex.

=== 5. What are the main limitations of the paper's theoretical analysis?

- *Identified Limitations (from Section 4.4)*: The authors explicitly identify two
  main limitations of their theoretical work.
  1. *Unconventional Robust Set Definition*: The robust set $tilde(cal(P))$ is
    defined with a novel divergence metric (Equation 4). It is unclear how this set
    relates to more standard robust sets used in other literature, such as those
    based on KL-divergence or $H_infinity$ constraints. MaxEnt RL might not be
    robust to those other, more conventional sets.
  2. *Construction of the Pessimistic Reward*: The theory states that to learn a
    policy robust for reward $r$, one must train with a pessimistic reward $overline(r)(s_t, a_t, s_( t+1 )) = 1/T log r(s_t, a_t) + cal(H)[s_( t+1 )|s_t, a_t]$.
    Computing the dynamics entropy term, $cal(H)[s_( t+1 )|s_t, a_t]$, is
    challenging in practice as it requires knowledge of the environment's dynamics,
    which may not be available in a model-free setting. This poses a practical
    challenge to directly applying the theory as prescribed.
