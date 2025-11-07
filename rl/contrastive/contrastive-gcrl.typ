#import "../styles/things.typ": challenge, hypothesis, question

= Contrastive Learning as Goal-Conditioned Reinforcement Learning

== Overview

=== Challenges

#challenge[
  Standard deep reinforcement learning (RL) algorithms often struggle to learn
  effective representations from high-dimensional inputs (like images) in an
  end-to-end fashion. This instability necessitates decoupling representation
  learning from RL by using auxiliary losses or data augmentation.
][
  #hypothesis[
    Instead of supplementing an RL algorithm with a separate representation
    learning component, a well-formulated representation learning algorithm can
    itself serve as an RL algorithm. Specifically, the objective of contrastive
    learning, which learns to pull representations of "positive" pairs together
    and push "negative" pairs apart, is structurally analogous to a
    goal-conditioned value function that assigns high values to reachable states
    and low values to unreachable ones.
  ]

  The paper casts goal-conditioned RL directly as a contrastive learning
  problem. It applies contrastive learning to trajectories, where a state-action
  pair $(s, a)$ is contrasted with future states. The inner product of the
  learned representations for $(s, a)$ and a goal state $s_g$ is shown to
  correspond directly to a Q-function, $Q(s, a, s_g)$.

  *Alternatives*: The paper explicitly compares its approach against methods
  that add perception-specific losses (like an autoencoder), use data
  augmentation (DrQ), or employ a separate contrastive loss for representation
  learning on top of a standard RL algorithm (CURL).
]

#challenge[
  Devising a simple and effective algorithm for goal-conditioned RL that works
  well from both low-dimensional states and high-dimensional images, without
  requiring manual reward engineering.
][
  #hypothesis[
    By formalizing the relationship between contrastive learning and
    Q-functions, a family of RL algorithms can be derived. This includes
    algorithms that are simpler than prior methods (by avoiding components like
    temporal difference learning) and potentially more performant.
  ]

  The paper proposes *Contrastive RL (NCE)*, an algorithm that uses a binary
  classification loss (Noise-Contrastive Estimation) to distinguish future
  states from random states. The policy is then updated to select actions that
  maximize the critic's output for a desired goal, effectively maximizing the
  probability of reaching that goal. This approach reinterprets a prior method,
  C-learning, as a member of this family and proposes new variants.

  *Alternatives*: The paper's baselines include actor-critic methods using
  Hindsight Experience Replay (TD3+HER), Goal-Conditioned Behavioral Cloning
  (GCBC), and model-based methods that learn a density model over future states.
]

=== Proposed Component: Contrastive RL (NCE)

- *High-Level Description:* Contrastive RL (NCE) is a goal-conditioned RL
  algorithm that alternates between learning a critic function via contrastive
  learning and updating a policy to maximize this critic. The critic,
  $f(s, a, s_g)$, is parameterized as an inner product of two neural network
  encoders: $f(s, a, s_g) = phi(s, a)^T psi(s_g)$. It is trained to output a
  high value if state $s_g$ is a likely future state following action $a$ in
  state $s$, and a low value otherwise. The policy $pi(a|s, s_g)$ is then
  trained to select actions that maximize this critic function for the given
  goal $s_g$.

- *Inputs:*
  - Triplets of $(s_t, a_t, s_f)$ sampled from a replay buffer of trajectories.
  - *Positive Pairs:* A state-action pair $(s_t, a_t)$ and a future state
    $s_f^+$ sampled from the discounted state occupancy measure of the same
    trajectory.
  - *Negative Pairs:* A state-action pair $(s_t, a_t)$ and a random state
    $s_f^-$ sampled from the replay buffer.

- *Outputs:*
  - A goal-conditioned policy $pi(a|s, s_g)$ that can drive the agent to a
    specified goal state $s_g$.

=== Dependencies for Reproduction

To reproduce the results, the following non-novel dependencies are required:

- *RL Environments & Datasets:*
  - Fetch Reach, Fetch Push
  - Sawyer Push, Sawyer Bin (from Meta-World)
  - AntMaze (including offline datasets from D4RL)
  - Point Spiral11x11
  - Nine-Room Environment
- *Software Libraries:*
  - JAX
  - ACME Reinforcement Learning library
- *Network Architectures:*
  - A CNN encoder from Mnih et al. (2013) for processing image observations.
  - An image decoder architecture from Ha and Schmidhuber (2018) for relevant
    baselines.

=== Additional Perspectives & Assumptions

- *Missing Perspectives from Abstract:*
  - The paper establishes that a prior method, *C-learning*, is a specific
    instance of their contrastive RL framework that uses temporal difference
    (TD) learning.
  - The framework introduces a *family of contrastive RL algorithms*, including
    a simpler Monte-Carlo based method (*Contrastive RL (NCE)*) , a variant
    using the InfoNCE objective (*Contrastive RL (CPC)*) , and a hybrid method
    that combines NCE and C-learning, which often achieves the highest
    performance.
  - A key theoretical finding is that the optimal critic function $f^*$ is
    directly proportional to the log Q-function:
    $exp(f^*(s,a,s_f)) prop Q_( s_f )^pi(s, a)$.
  - The approach shows significant success in the *offline RL setting* on the
    D4RL benchmark, outperforming specialized offline methods like IQL and
    imitation learning methods like GCBC.

- *Glaring Assumptions:*
  - *Convergence Guarantee:* The formal proof of policy improvement (and
    convergence) relies on a "filtering step" which is empirically shown to harm
    performance and is therefore *not used* in the presented experiments. Thus,
    the practical algorithm does not have a convergence guarantee.
  - *Reward Formulation:* The entire theoretical analysis hinges on a specific
    definition of the reward function as the discounted probability of reaching
    the goal: $r_g (s_t,a_t) eq.delta (1-gamma) p(s_( t+1 )=s_g|s_t,a_t)$. While
    the algorithm does not explicitly use this reward, its correctness as a
    Q-function estimator depends on it.
  - *Simplifying Assumptions for Proofs:* The convergence proof assumes tabular
    states and actions and that the learned critic is Bayes-optimal, neither of
    which holds in the practical, high-dimensional, function-approximation
    settings of the experiments.

== Problem Formulation

=== Standard Goal-Conditioned RL Formulation

The paper begins by defining the goal-conditioned reinforcement learning problem
as a multi-task RL problem where each task corresponds to reaching a specific
goal state.

- *Problem Definition*: The environment is defined by:
  - States: $s_t in cal(S)$.
  - Actions: $a_t$.
  - Initial state distribution: $p_0(s)$.
  - Transition dynamics: $p(s_( t+1 ) | s_t, a_t)$.
  - A distribution over goals: $p_g(s_g)$.

- *Reward Function*: A key aspect of the formulation is the definition of the
  reward function, which is defined not by a manual metric but as the
  probability density of reaching the goal $s_g$ at the very next timestep.

$
  r_{g}(s_t, a_t) eq.delta (1-gamma)p(s_( t+1 )=s_g | s_t, a_t)
$

- At the initial state ($t=0$), this definition also includes the probability
  that the agent started at the goal.

- *Objective*: The agent's objective is to learn a goal-conditioned policy
  $pi(a|s, s_g)$ that maximizes the expected discounted future rewards.

$
  max_pi EE_(p_g (s_g), pi(tau|s_g)) [sum_(t = 0)^infinity gamma^t r_g (s_t, a_t)]
$

- *Q-Function*: The corresponding goal-conditioned Q-function, or state-action
  value function, is the expected return after taking action $a$ in state $s$
  for a given goal $s_g$.

$
  Q_(s_g)^pi (s, a) eq.delta EE_(pi(tau|s_g)) [sum_(t' = t)^infinity gamma^(t' - t) r_g (s_(t'), a_(t')) bar s_t = s, a_t = a]
$

=== Probabilistic Reformulation

The paper reformulates the value function in terms of probabilities by
leveraging the *discounted state occupancy measure*, which is the discounted
probability distribution over future states visited by a policy.

- *Discounted State Occupancy Measure*: This measure for a policy $pi$
  conditioned on a goal $s_g$ is defined as:

$
  p^(pi(dot.op|dot.op, s_g))(s_(t +) = s) eq.delta(1 - gamma) sum_(t = 0)^infinity gamma^t p_t^(pi(dot.op|dot.op, s_g))(s_t = s)
$

- Here, $p_t^pi(s)$ is the probability density over states that the policy
  visits after exactly $t$ steps. Sampling from this involves drawing a time
  offset $t tilde "Geom"(1-gamma)$ and observing the state at that time.

- *Proposition 1: Q-Functions as Probabilities*: The central insight of this
  reformulation is that the Q-function for the defined reward is equivalent to
  the probability of observing the goal state under this discounted state
  occupancy measure.

$
  Q_(s_g)^pi (s, a) = p^(pi(dot.op|dot.op, s_g))(s_(t +) = s_g |s, a)
$

=== Contrastive Learning Formulation

This probabilistic view allows the problem to be cast as a contrastive
representation learning task, where the goal is to learn a critic function $f$
that estimates the Q-function.

- *Critic Parameterization*: The critic function $f$ is parameterized as the
  inner product of two encoders:
  - A state-action encoder $phi(s, a)$.
  - A goal encoder $psi(s_g)$.
  - $f(s, a, s_g) = phi(s, a)^T psi(s_g)$.

- *Contrastive Setup*: The critic is trained by distinguishing between
  "positive" and "negative" future states.
  - *Positive Pairs*: For a given state-action pair $(s_t, a_t)$, the positive
    future state $s_f^+$ is sampled from the discounted state occupancy measure
    of the policy that collected the data:
    $s_f^+ ~ p^pi(dot|dot)(s_( t+ ) | s_t, a_t)$.
  - *Negative Pairs*: The negative future state $s_f^-$ is sampled from the
    marginal distribution of all future states: $s_f^- ~ p(s_( t+ ))$.

- *Contrastive Objective (NCE)*: The critic $f$ is trained to optimize a binary
  cross-entropy loss based on Noise-Contrastive Estimation (NCE).

$
  max_f EE [log sigma(f(s, a, s_f^+)) + log(1 - sigma(f(s, a, s_f^-)))]
$

- The full loss term as presented in the paper is
  $cal(L)(s, a, s_f^+, s_f^-) eq.delta log sigma(f(s, a, s_f^+)) + log(1 - sigma(f(s, a, s_f^-)))$.

- *Lemma 4.1: Critic as a Q-Function*: The optimal critic $f^*$ that maximizes
  this objective is directly related to the Q-function.

$
  exp(f^* (s, a, s_f)) = 1/(p(s_f)) dot.op Q_(s_f)^(pi(dot.op|dot.op))(s, a)
$

- This shows that the learned critic is an unnormalized density model that
  captures the Q-value, where the partition function $p(s_f)$ can be ignored
  during action selection.

=== Policy Learning Formulation

Finally, the learned critic is used to update the goal-conditioned policy
$pi(a|s, s_g)$.

- *Policy Objective*: The policy is trained to select actions that maximize the
  critic's score for a given goal, which corresponds to maximizing the
  log-likelihood of reaching that goal.

$
  max_(pi(a|s, s_g)) EE_(pi(a|s, s_g) p(s) p(s_g)) [f(s, a, s_f = s_g)]
$

- This objective is approximately equivalent to maximizing the learned
  Q-function: $EE[ log Q_(s_g)^pi(s, a) - log p(s_g) ]$.

== Pipeline

=== Data Collection & Storage

This initial stage involves populating a replay buffer with experience by
interacting with the environment.

- *Description:* The agent, controlled by the current policy $pi(a|s, s_g)$,
  interacts with the RL environment. Unlike standard methods that store
  individual transitions, this approach stores entire *trajectories* to preserve
  the temporal relationship between states, which is crucial for sampling future
  states.
- *Inputs:*
  - The current goal-conditioned policy, $pi(a|s, s_g)$.
  - The reinforcement learning environment.
- *Process:*
  1. At the start of an episode, sample a goal $s_g$.
  2. Execute the policy $pi(a_t|s_t, s_g)$ in the environment to collect a
    trajectory of interactions: $tau = (s_0, a_0, s_1, a_1, ..., s_T)$.
  3. Store the entire trajectory $tau$ in the replay buffer.
  4. Repeat until the replay buffer, with a capacity of 1,000,000 transitions,
    is sufficiently full.
- *Outputs:*
  - *Replay Buffer:* A collection of trajectories, ready for sampling.

=== Batch Sampling

In this stage, a batch of data is carefully sampled from the replay buffer to
construct the inputs required for the contrastive learning update.

- *Description:* A mini-batch of size $B=256$ is formed by sampling trajectories
  and then constructing state-action pairs, positive future states, and negative
  future states from them. Goals for the actor update are also sampled.
- *Inputs:*
  - The replay buffer containing trajectories.
  - Batch size $B$.
- *Process:*
  1. Sample $B$ state-action pairs $(s_i, a_i)$ from the trajectories in the
    replay buffer.
  2. For each pair $(s_i, a_i)$, sample a *positive future state* $s_( f,i )^+$
    by first sampling a time offset $t' tilde "GEOM"(1-gamma)$ and then
    selecting the state from the same trajectory that occurred $t'$ steps after
    $(s_i, a_i)$
  3. The set of $B$ positive future states ${s_( f,i )^+}_( i=1 )^B$ will serve
    as both the positive examples and the pool of negative examples for other
    samples in the batch.
  4. Sample $B$ *random goals* ${g_i}_( i=1 )^B$ from the replay buffer to be
    used in the actor update. The paper finds that using random states as goals
    works best.
- *Outputs:*
  - `states`: A tensor of current states. Shape: $(B, D_"state")$.
  - `actions`: A tensor of corresponding actions. Shape: $(B, D_"action")$.
  - `future_states`: A tensor of the positive future states. Shape:
    $(B, D_"state")$.
  - `goals`: A tensor of random goals for the actor update. Shape:
    $(B, D_"state")$.

=== Critic Update

This is the core representation learning step where the critic, composed of the
state-action encoder $phi$ and goal encoder $psi$, is updated via contrastive
learning. This follows `critic_loss` in Algorithm 1.

- *Inputs:*
  - `states` tensor. Shape: $(B, D_"state")$.
  - `actions` tensor. Shape: $(B, D_"action")$.
  - `future_states` tensor. Shape: $(B, D_"state")$.
- *Process:*
  1. *Encode State-Actions:* The `states` and `actions` tensors are passed
    through the state-action encoder $phi$ to produce representations.
    - `sa_repr` = $phi("states, actions")$. Shape: $(B, D_"repr")$.
  2. *Encode Future States (Goals):* The `future_states` tensor is passed
    through the goal encoder $psi$ to produce goal representations.
    - `g_repr` = $psi("future_states")$. Shape: $(B, D_"repr")$.
  3. *Compute Logit Matrix:* An efficient outer product is used to compute the
    similarity score between every state-action representation and every goal
    representation.
    - $"logits"_{i j} = angle.l "sa_repr"_i, "g_repr"_j angle.r$.
    - The resulting `logits` tensor has a shape of $(B, B)$.
  4. *Calculate Contrastive Loss:* A binary cross-entropy loss is computed. The
    target is an identity matrix `eye(B)`, which labels the relationship between
    a state-action pair and its *actual* future state as positive (1) and all
    other combinations within the batch as negative (0).
    - `loss` = `sigmoid_binary_cross_entropy(logits, eye(B))`.
  5. *Gradient Update:* The gradients of the loss with respect to the parameters
    of encoders $phi$ and $psi$ are computed and an optimizer step (e.g., Adam)
    is taken to minimize the loss.
- *Outputs:*
  - Updated weights for the state-action encoder $phi$ and goal encoder $psi$.

=== Actor Update

The policy (actor) is updated using the newly trained critic to improve its
ability to reach desired goals. This follows `actor_loss` in Algorithm 1.

- *Inputs:*
  - `states` tensor from the batch. Shape: $(B, D_"state")$.
  - `goals` tensor from the batch. Shape: $(B, D_"state")$.
  - The updated encoders $phi$ and $psi$.
- *Process:*
  1. *Sample Actions from Policy:* For the batch of `states` and `goals`, sample
    actions from the current policy $pi(a|s, g)$. This step must be
    differentiable (e.g., using the reparameterization trick).
    - `policy_actions` = $pi(dot | "states, goals")$. Shape: $(B, D_"action")$
    2. *Encode State-Actions:* Pass the `states` and the `policy_actions`
      through the updated state-action encoder $phi$.
      - `sa_repr` = $phi("states, policy_actions")$. Shape: $(B, D_"repr")$.
    3. *Encode Goals:* Pass the `goals` through the updated goal encoder $psi$.
      - `g_repr` = $psi("goals")$. Shape: $(B, D_"repr")$.
    4. *Calculate Actor Loss:* Compute the critic's score for each $(s, a, g)$
      triplet using an element-wise inner product. The actor's objective is to
      maximize this score, so the loss is its negation.
      - $"scores"_i = angle.l "sa_repr"_i, "g_repr"_i angle.r$.
      - `loss` = -`mean(scores)`.
    5. *Gradient Update:* Compute the gradients of the loss with respect to the
      policy $\pi$'s parameters and take an optimizer step to minimize this
      loss.
- *Outputs:*
  - Updated weights for the policy network $\pi$.

The entire pipeline (Stages 1-4) is repeated, alternating between data
collection, critic updates, and actor updates, to progressively improve the
agent's goal-reaching capabilities.

== Discussion

#question[
  How does Contrastive RL compare to existing goal-conditioned RL methods?
][
  Contrastive RL (NCE) was benchmarked against three distinct classes of prior
  methods: Actor-Critic (TD3+HER), Behavioral Cloning (GCBC), and Model-Based
  methods.
][
  The primary metric was *success rate* over millions of environment steps. On
  state-based tasks, Contrastive RL (NCE) outperformed all baselines on the
  pushing tasks and was the only method to solve the most difficult task,
  `sawyer bin`. On image-based tasks, the performance gap was even larger.
][
  The results demonstrate that Contrastive RL is not just a theoretical
  curiosity but a *highly competitive algorithm* in practice. Its strong
  performance on image-based tasks suggests that the implicit representation
  learning baked into the contrastive objective is particularly effective for
  high-dimensional state spaces where traditional methods falter.
]

#question[
  Is it better to build an RL algorithm that *is* a representation learner, or
  to add representation learning on top of an existing RL algorithm?
][
  Contrastive RL (NCE), which has no auxiliary losses or data augmentation, was
  compared against a strong baseline (TD3+HER) enhanced with popular,
  state-of-the-art representation learning techniques.
][
  While adding DrQ or AE provided some performance improvements to the TD3+HER
  baseline on specific tasks, *Contrastive RL (NCE) outperformed all of these
  enhanced baselines on all tasks*.
][
  This is a key finding of the paper. It provides strong evidence that designing
  an RL algorithm to *structurally resemble a representation learning method is
  more beneficial* than simply adding representation learning "tricks" to a
  conventional algorithm.
]

=== How do different members of the Contrastive RL family compare?

After establishing Contrastive RL as a valid framework, this question explores
the performance of different algorithmic variants within that framework.

- *Experiment Design:*
  - The paper compared four distinct instantiations of Contrastive RL:
    1. *Contrastive RL (NCE):* The simple, proposed algorithm using a binary
      contrastive loss.
    2. *C-learning:* A prior method re-framed as a contrastive algorithm that
      uses temporal difference (TD) learning.
    3. *Contrastive RL (CPC):* A variant based on the InfoNCE objective
      (categorical cross-entropy loss) instead of the binary loss.
    4. *Contrastive RL (NCE + C-learning):* A hybrid method that combines the
      objectives of the NCE and C-learning variants.
  - These variants were compared across the full suite of state- and image-based
    tasks.
- *Results & Metrics:*
  - Using *success rate* as the metric, the experiments showed that the hybrid
    *Contrastive RL (NCE + C-learning) consistently ranked among the
    best-performing methods* across tasks.
  - C-learning performed well but was sometimes outperformed by the other
    methods.
  - The simpler NCE variant achieved strong performance, often comparable to or
    better than C-learning, despite being much simpler to implement.
- *Significance of Results:*
  - This demonstrates the utility of the generalized Contrastive RL framework.
    It not only provides a new understanding of a prior method (C-learning) but
    also enables the design of new algorithms that are either *simpler (NCE) or
    achieve higher performance (NCE + C-learning)*.
- *Limitations:*
  - The paper does not provide a deep theoretical analysis for why the hybrid
    `NCE + C-learning` method performs best; it remains an empirical
    observation.

=== How does the Contrastive RL framework perform in the challenging offline RL setting?

This question tests the versatility of the proposed method by applying it to the
offline reinforcement learning problem, where the agent can only learn from a
fixed, pre-collected dataset.

- *Experiment Design:*
  - Contrastive RL (NCE) was adapted for the offline setting by adding a
    *behavioral cloning (BC) term* to its policy objective, a common technique
    for offline RL.
  - The algorithm was evaluated on the goal-conditioned *D4RL AntMaze
    benchmark*, a standard for offline RL evaluation.
  - It was compared against TD-free methods (GCBC, Decision Transformer) and
    state-of-the-art TD-based offline RL methods (TD3+BC, IQL).
- *Results & Metrics:*
  - The metric was the final *normalized success score* reported in prior work
    and D4RL.
  - *Contrastive RL outperformed all baselines on 5 out of 6 AntMaze tasks*.
  - The performance gains were most significant on the hardest tasks, with a
    7-9% absolute improvement over IQL. The median improvement over GCBC was
    15%.
- *Significance of Results:*
  - This shows that the benefits of the contrastive framework *transfer directly
    to the offline setting*, demonstrating its robustness and power.
  - The results suggest that for offline goal-reaching, the core mechanism of
    hindsight relabeling (which underpins Contrastive RL) combined with value
    estimation is more effective than complex TD-learning schemes that do not
    use relabeling.
- *Limitations:*
  - The offline adaptation requires an additional behavioral cloning term, so
    the success is not due to the "pure" contrastive objective alone.
  - The paper notes that performance improved when increasing the number of
    critic networks used, suggesting that success in offline RL may depend
    heavily on model capacity, but this trade-off is not fully explored.

=== What are the primary limitations and open questions for this line of work?

The conclusion explicitly discusses the main boundary of the work and points to
future directions.

- *Identified Limitation:*
  - The most significant limitation is that the entire framework and all
    experiments are restricted to *goal-conditioned RL problems*. The reward
    function is implicitly tied to reaching a goal state.
- *Open Questions:*
  - It remains an open question *how these methods can be applied to general RL
    problems* with arbitrary, non-goal-oriented reward functions.
  - The authors also pose whether the rich set of ideas and techniques from the
    broader contrastive learning literature could be leveraged to *construct
    even better RL algorithms* in the future.

