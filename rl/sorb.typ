#import "../styles/things.typ": challenge, hypothesis, question

= Search on the Replay Buffer

== Overview

This paper introduces *Search on the Replay Buffer (SoRB)*, a control algorithm
designed to solve complex, long-horizon tasks with sparse rewards, particularly
in high-dimensional environments like those with image-based observations. It
achieves this by bridging the gap between classical planning algorithms and
modern reinforcement learning (RL).

=== Challenges

#challenge[
  Standard reinforcement learning (RL) struggles to plan over long horizons, and
  goal-conditioned RL policies fail when the goal is distant.
][
  A long, difficult task can be decomposed into a sequence of shorter, easier
  sub-problems.

  #hypothesis[
    A long, difficult task can be decomposed into a sequence of shorter, easier
    sub-problems.
  ]

  Instead of learning a single policy to reach the final goal, the agent uses
  graph search over previously seen states to find a sequence of intermediate
  waypoints. A goal-conditioned policy is then used to navigate between these
  waypoints. Hierarchical RL methods also learn to sequence skills, but SoRB
  replaces a learned high-level policy with a deterministic graph search
  algorithm to improve stability.
]

#challenge[
  Classical planning algorithms, while good at long-horizon reasoning, are
  difficult to apply in high-dimensional state spaces (e.g., images) because
  they require a predefined graph of the environment, a distance metric, and a
  local policy to travel between nodes.
][
  The necessary components for planning (a graph of states, edge distances, and
  a local policy) can be learned directly from environment interaction using RL.

  #hypothesis[
    The necessary components for planning (a graph of states, edge distances,
    and a local policy) can be learned directly from environment interaction
    using RL.
  ]

  - *Graph Nodes*: The graph is constructed using states from the agent's replay
    buffer, treating them as a non-parametric model of valid, reachable states.
    This avoids the need to generate photo-realistic images.
  - *Edge Distances*: A goal-conditioned Q-function is trained with a reward of
    $-1$ for every step. This formulation makes the optimal Q-value,
    $Q(s, a, s_g)$, directly correspond to the negative shortest-path distance,
    $-d_( s p )(s, a, s_g)$. This learned function provides the edge weights for
    the graph.
  - *Local Policy*: The policy learned by the same goal-conditioned RL algorithm
    serves as the local controller to navigate between the planned waypoints.
]

#challenge[
  Learning accurate distance estimates with RL is difficult. Unreachable states
  have an infinite distance (negative infinity value), which is hard for
  standard neural networks to represent. Furthermore, spurious distance
  predictions can create "wormholes" that mislead the planner.
][
  Specialized RL techniques can produce more robust and accurate distance
  estimates suitable for planning.

  #hypothesis[
    Specialized RL techniques can produce more robust and accurate distance
    estimates suitable for planning.
  ]

  1. *Distributional RL*: The paper uses distributional Q-learning to represent
    distances as a probability distribution over a set of discrete bins. This
    approach elegantly handles large or infinite distances by having a final
    "catch-all" bin for any distance greater than a threshold, $N$.
  2. *Ensembles of Value Functions*: To prevent the graph search from exploiting
    erroneously short distance predictions "wormholes", the algorithm trains an
    ensemble of independent Q-networks. The distance between two states is then
    an aggregation (e.g., the maximum or average) of the predictions from all
    networks in the ensemble, making the estimate more robust.
  - *Alternatives*: The paper notes that simply clipping Q-values is an
    alternative for handling large distances, but it proved difficult to train
    effectively.
]

=== SoRB Algorithm Overview

- *Component*: A hybrid algorithm named *Search on the Replay Buffer (SoRB)*
  that combines a learned goal-conditioned policy with a graph-based planner.
- *Inputs*:
  - The agent's current state, $s$.
  - The final goal state, $s_g$.
  - A replay buffer, $cal(B)$, containing previously visited states.
  - A learned, goal-conditioned value function, $V$ (derived from a Q-function),
    which estimates distances.
  - A learned, goal-conditioned policy, $pi$, which can reach nearby goals.
- *Processing*:
  1. A graph is constructed where nodes are states in the replay buffer
    $cal(B)$.
  2. The edge weight between any two nodes $s_1, s_2 in cal(B)$ is the distance
    estimated by the value function, $-V(s_1, s_2)$. Edges longer than a
    hyperparameter `MAXDIST` are ignored.
  3. Given a current state $s$ and goal $s_g$, they are temporarily added to the
    graph.
  4. *Dijkstra's algorithm* is used to find the shortest path of waypoints
    $(s_w_1, s_w_2, ...)$ from $s$ to $s_g$ through the graph.
  5. The algorithm then selects an action by feeding the *next waypoint*
    ($s_w_1$) as the goal to the goal-conditioned policy $pi$.
- *Output*: An action, $a$, to be executed in the environment.

=== Dependencies and Prerequisites

- *Environments & Datasets*:
  - *Didactic Tasks*: 2D navigation environments named `Point-U` and
    `Point-FourRooms`.
  - *Visual Navigation*: 3D house models from the *SUNCG dataset*.
- *Novelty-Exclusive Algorithms*:
  - *Core RL Framework*: Goal-conditioned RL using off-policy algorithms like
    *DQN* and *DDPG*.
  - *Distance Estimation*: *Distributional Reinforcement Learning (C51)* is a
    critical component for learning the value function used for distances.
  - *Search*: *Dijkstra's Algorithm* for finding the shortest path and the
    *Floyd-Warshall algorithm* for pre-computing all-pairs shortest paths to
    improve efficiency.
  - *Robustness*: Ensemble methods based on *bootstrapping*.

=== Additional Perspectives & Assumptions

- *Key Missing Abstract Detail*: The abstract mentions combining planning and RL
  but omits the crucial mechanism: defining a reward function of
  $r(s,a,s_g) eq.delta -1$ allows the optimal value function to directly learn
  the negative shortest-path distance, $V(s,s_g)=-d_(s p)(s,s_g)$.
- *Critical Implementation Details*: The success of SoRB is heavily dependent on
  two techniques not highlighted in the abstract:
  1. *Distributional RL*, without which the method performs worse than a random
    policy.
  2. *Ensembles of value functions*, which are vital for preventing planners
    from exploiting inaccurate distance predictions "wormholes" in
    high-dimensional tasks.
- *Glaring Assumptions*:
  - The replay buffer is assumed to contain a representative sample of states
    that sufficiently cover the environment, including critical "bottleneck"
    states needed to connect different regions. If the graph is disconnected,
    planning will fail.
  - The agent's learned local policy is assumed to be capable of reliably
    navigating between any two states deemed "close" by the learned distance
    function (i.e., distance < `MAXDIST`).

=== Recommended Prerequisites

For a deeper understanding of the components used in this paper, the following
are recommended:

- *Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional
  perspective on reinforcement learning.* This paper is essential as SoRB's
  success relies on the distributional RL approach for learning distances.
- *Andrychowicz, M., et al. (2017). Hindsight experience replay.* This paper
  introduces goal-relabelling, a key technique for training the goal-conditioned
  policies that SoRB is built upon.

== Problem Formulation

The problem is framed as a goal-conditioned reinforcement learning task, aiming
to learn a policy that can reach a diverse set of goal states, particularly
those that are distant and require long-horizon reasoning.

=== Goal-Conditioned Reinforcement Learning

The environment is modeled as a standard Markov Decision Process (MDP) with the
addition of a goal state.

- *State and Goal Space:* The agent observes its current state $s in cal(S)$ and
  a goal state $s_g in cal(S)$.
- *Policy:* The agent's behavior is defined by a goal-conditioned policy,
  $pi(a|s, s_g)$, which outputs an action $a$ given the current state $s$ and
  the goal state $s_g$.
- *Objective:* The agent's objective is to maximize its cumulative, undiscounted
  reward.
- *Goal-Conditioned Value Functions:* The core of the method involves learning a
  goal-conditioned Q-function and its corresponding value function using an
  off-policy algorithm. The policy is derived by acting greedily with respect to
  the Q-function.
  - *Q-Function:*

$
  Q(s, a, s_g) = EE_pi [sum_(t = 1)^T r(s_t, a_t, s_g)]
$

This represents the expected cumulative reward for taking action $a$ in state
$s$ and subsequently following policy $pi$ to reach goal $s_g$.
- *Value Function:*

$
  V(s, s_g) = max_a Q(s, a, s_g)
$

This represents the maximum expected cumulative reward from state $s$ to reach
goal $s_g$.

=== Learning Distances from Values

A key insight is to define a specific reward function that directly connects the
learned value functions to the shortest-path distance between states.

- *Reward Function:* The reward is defined as $-1$ for every step taken,
  encouraging the agent to reach the goal as quickly as possible.

$
  r(s, a, s_g) eq.delta - 1
$

- *Value as Negative Distance:* With the above reward function, the optimal
  value function $V(s, s_g)$ equals the negative shortest-path distance, defined
  as the minimum expected number of steps to get from $s$ to $s_g$ under the
  optimal policy, denoted $d_s p(s, s_g)$.

$
  V(s, s_g) = - d_(s p)(s, s_g)
$

Similarly, the optimal Q-function equals the negative shortest-path distance
conditioned on first taking action $a$.

$
  Q(s, a, s_g) = - d_(s p)(s, a, s_g)
$

=== Distributional Reinforcement Learning for Distances

To handle unreachable states (where distance is infinite) and improve the
accuracy of distance estimates, the problem is formulated using distributional
RL.

- *Value Distribution:* The Q-function is modeled to predict a probability
  distribution over a discrete set of $N$ possible distances (bins),
  $B = (B_1, ..., B_N)$. The output, $Q(s_t, s_g, a_t)$, is a probability vector
  where the $i$-th element is the predicted probability that the distance is $i$
  steps. The final bin, $B_N$, is a catch-all for distances $>= N$, providing a
  well-defined way to handle large or infinite distances.
- *Distributional Bellman Update:* The target distribution, $Q^*$, for the
  Bellman update is defined as follows:

$
  Q^* = cases(
    (1 comma 0 comma ... comma 0) & "if " s_t = s_g, "shift"(V(s_(t + 1) comma s_g)) & "if " s_t != s_g,
  )
$

If the goal is not reached, the target distribution is a one-step right-shift of
the predicted value distribution for the next state, $V(s_t+1, s_g)$, which
intuitively means the distance has increased by one step.
- *Loss Function:* The Q-network is trained by minimizing the Kullback-Leibler
  (KL) divergence between the target distribution $Q^*$ and the predicted
  distribution $Q^theta$:

$
  min_theta D_(K L)(Q^* ||Q^theta)
$

== Pipeline

The SoRB algorithm is implemented in two main phases: a training phase to learn
the distance function and a planning/execution phase that uses this function to
find and follow waypoints.

=== Training the Goal-Conditioned Agent

This stage learns the foundational components: a goal-conditioned policy and an
ensemble of value functions that estimate distances.

- *Description:* An ensemble of goal-conditioned Q-networks is trained using an
  off-policy RL algorithm (like DDPG or DQN) modified for distributional output.
  The training uses a large replay buffer $cal(B)_"train"$ and goal relabeling
  strategies to improve sample efficiency. The agent is trained on a specific
  task (e.g., visual navigation in houses) with a reward of $-1$ at each step.
- *Inputs:*
  - *Current State ($s_t$):* A tensor representing the agent's observation. For
    visual navigation, this is a panoramic image.
    - *Shape:* `(batch_size, H, W, C)`, where e.g., $H=24, W=32, C=6$ for
      concatenated current/goal RGB images.
  - *Goal State ($s_g$):* A tensor representing the goal observation.
    - *Shape:* `(batch_size, H, W, C)`.
- *Outputs:*
  - *Ensemble of Q-Functions ($Q_k^theta_k=1^K$):* A set of $K$ trained
    Q-networks (e.g., $K=3$ ). Each network can predict a distribution of
    distances for a given state-action-goal triplet.
  - *Goal-Conditioned Policy ($pi(a|s, s_g)$):* A policy derived from the
    Q-functions, typically by acting greedily with respect to the (ensembled)
    Q-values.

=== Graph Construction

Using the trained agent, a weighted, directed graph is constructed over a set of
previously seen states.

- *Description:* The nodes of the graph are states sampled from a replay buffer,
  $cal(B)_"search"$. An edge exists between any two nodes
  $s_1, s_2 in cal(B)_"search"$ if the distance between them is less than a
  hyperparameter `MAXDIST`. The edge weight is the distance estimated by the
  trained value function ensemble. To make this efficient, the paper suggests
  pre-computing all-pairs shortest paths within the buffer using the
  Floyd-Warshall algorithm.
- *Inputs:*
  - *Search Replay Buffer ($cal(B)_"search"$):* A collection of previously
    visited states.
    - *Shape:* A set of `N_"search"` state tensors, e.g., `1000` states of
      `(H, W, C)`.
  - *Ensemble of Value Functions ($V_k_k=1^K$):* Derived from the trained
    Q-functions in Stage 1.
- *Outputs:*
  - *Graph ($cal(G)$):* A weighted, directed graph where
    $cal(V) = cal(B)_"search"$ and edge weights $cal(W)(s_1, s_2)$ are computed
    as follows:
    1. For each value function $V_k$ in the ensemble, calculate the distance
      estimate $d_k(s_1, s_2) = -V_k(s_1, s_2)$.
    2. Aggregate the estimates (e.g., by taking the average or maximum) to get a
      robust distance $d(s_1, s_2)$.
    3. The weight is $d(s_1, s_2)$ if $d(s_1, s_2) < "MAXDIST"$, otherwise it's
      effectively infinite.

=== Path Planning via Graph Search

At execution time, the algorithm finds the shortest sequence of waypoints from
the current state to the goal.

- *Description:* Given a current state $s$ and a final goal $s_g$, they are
  temporarily added to the graph. Dijkstra's algorithm is then used to find the
  shortest path from $s$ to $s_g$ through the waypoints in $cal(G)$. This
  process is outlined in Algorithm 1 as `SHORTESTPATH(s, s_g, B, V)`.
- *Inputs:*
  - *Current State ($s$):* The agent's current observation tensor.
  - *Goal State ($s_g$):* The agent's final goal observation tensor.
  - *Graph ($cal(G)$):* The graph constructed in Stage 2.
  - *Ensemble of Value Functions ($V_k_k=1^K$):* Used to compute edge weights
    from $s$ to the graph nodes and from the graph nodes to $s_g$.
- *Outputs:*
  - *Waypoint Sequence:* An ordered list of states from the replay buffer, e.g.,
    $(s_w_1, s_w_2, dots, s_w_n)$, representing the shortest path to the goal.

=== Execution (SEARCHPOLICY)

The agent executes the plan by navigating sequentially to each waypoint.

- *Description:* This stage, detailed in `SEARCHPOLICY` (Algorithm 1), is the
  agent's action-selection mechanism. The agent identifies the *first* waypoint
  $s_w_1$ from the planned path. It then compares the distance to this waypoint,
  $d_s -> w_1$, against the distance to the final goal, $d_s -> g$. It will
  typically use the policy to move towards the first waypoint, $pi(a|s, s_w_1)$.
  However, if the final goal is closer and not too far away (i.e., not greater
  than `MAXDIST`), it will switch to targeting the final goal directly,
  $pi(a|s, s_g)$.
- *Inputs:*
  - *Current State ($s$):* The agent's current observation.
  - *Goal State ($s_g$):* The agent's final goal.
  - *First Waypoint ($s_w_1$):* The first state in the sequence from Stage 3.
  - *Goal-Conditioned Policy ($pi$):* From Stage 1.
  - *Value Function Ensemble (V):* From Stage 1, used to calculate distances
    $d_s -> w_1 = -V(s, s_w_1)$ and $d_s -> g = -V(s, s_g)$.
- *Outputs:*
  - *Action ($a$):* A single action for the agent to take in the environment at
    the current timestep.
    - *Shape:* Dependent on action space (e.g., a discrete integer for
      navigation or a continuous vector `(dx, dy)`).

== Discussion

#question[
  How does SoRB perform on long-horizon tasks compared to a standard
  goal-conditioned policy?
][
  The authors compared their full SoRB algorithm ("search") against the
  underlying goal-conditioned RL policy acting alone ("default"). This was
  tested in two simple 2D navigation environments (`Point-U` and
  `Point-FourRooms`) and a more complex 3D visual navigation task. The
  difficulty was varied by increasing the distance between the agent's start and
  goal positions.
][
  *Metric:* The primary metric was *success rate*, plotted against the optimal
  number of steps to the goal. Success was defined as reaching the goal state
  within a given step limit. *Results:* In the 2D tasks, the standard RL
  policy's success rate dropped sharply as the goal distance increased, while
  SoRB maintained a high success rate even for goals over 100 steps away. For
  example, the standard policy often failed completely for goals 60 steps away,
  whereas SoRB succeeded consistently. Visualizations show the standard policy
  getting stuck, while SoRB successfully finds paths through complex corridors
  by navigating to intermediate waypoints.
][
  This result is significant because it demonstrates that the core idea of
  searching over a replay buffer can dramatically extend the planning horizon of
  a goal-conditioned policy. It directly addresses the common failure mode of RL
  agents in tasks with sparse rewards and long temporal distances, where
  learning a coherent long-term plan is difficult.
]

=== What are the crucial components for SoRB's performance?

- *Experiment Design:* The authors conducted several ablation studies on the
  visual navigation task to isolate the contributions of SoRB's key components.
  1. *Distributional RL:* They evaluated a variant of SoRB trained without the
    distributional RL formulation for estimating distances.
  2. *Ensembles:* They tested the impact of the value function ensemble by
    running experiments with ensemble sizes of 1, 2, and 3.
  3. *Hyperparameters:* They analyzed sensitivity to the replay buffer size
    (testing 100, 200, and 1000 states) and the `MAXDIST` hyperparameter, which
    controls the connectivity of the search graph.

- *Results & Metrics:*
  - *Metric:* The metric was the *success rate* versus goal distance.
  - *Results:*
    - *Distributional RL was critical*; the variant without it performed worse
      than a random policy, showing it is a key component.
    - *Ensembles provided a notable boost*, increasing the success rate by
      10-20% for distant goals (10+ steps away). This prevents the planner from
      exploiting erroneous "wormholes" in the value landscape.
    - Performance was *robust to replay buffer size*, with no discernible drop
      even when decreasing the buffer by 10x (from 1000 to 100 states).
    - The algorithm was *sensitive to `MAXDIST`*, with performance degrading if
      the value was too small (the graph becomes disconnected) or too large
      (increases the chance of including erroneous edges).

- *Significance:* These ablations provide a clear understanding of *why* SoRB
  works. They show that accurate and robust distance estimation is the most
  important factor, and that distributional RL and ensembles are the primary
  mechanisms for achieving it. The robustness to a small replay buffer size is
  also a significant practical advantage, making the method relatively
  memory-efficient.

- *Limitations:* The sensitivity to the `MAXDIST` hyperparameter suggests that
  the method requires some tuning for optimal performance. The authors note that
  better uncertainty quantification in RL could potentially improve stability
  with respect to this hyperparameter in the future.

=== Why does SoRB outperform a closely related method like SPTM?

- *Experiment Design:* The authors identified two key differences between SoRB
  and Semi-Parametric Topological Memory (SPTM): (1) the underlying
  goal-conditioned policy and (2) the learned distance metric used for graph
  search. To isolate these factors:
  1. They compared the performance of the SoRB policy (learned via
    goal-conditioned RL) and the SPTM policy (an inverse model learned with
    supervised learning) *without* using graph search.
  2. They directly evaluated the quality of the distance metrics from both
    methods.

- *Results & Metrics:*
  - *Metrics:* Policy performance was measured by *success rate*. Distance
    metric quality was measured by the *Area Under the Curve (AUC) of a
    precision-recall curve*, which quantifies how well a predicted distance
    corresponds to the policy's actual ability to navigate between two states.
  - *Results:*
    - The underlying goal-conditioned policies of both methods were found to be
      comparable, with SPTM's policy even slightly outperforming SoRB's when
      search was not used. This showed the policy was not the reason for SoRB's
      superior performance.
    - SoRB's distance metric was found to be far more accurate, achieving an
      *AUC that was 22% higher* than SPTM's (0.97 vs 0.75). This indicates
      SoRB's distance predictions are much better aligned with what its policy
      can actually achieve.

- *Significance:* This is a crucial finding because it pinpoints the source of
  SoRB's advantage: its method for *learning a better distance function*. While
  SPTM uses a random policy to learn its distance metric, SoRB uses
  goal-conditioned RL, creating a tighter coupling between the distance function
  and the policy's capabilities. This insight is valuable for future research in
  planning-based RL.

- *Limitations:* The comparison is specific to these two methods. While
  insightful, the conclusion that the distance metric is more important than the
  policy might not hold universally for all planning-and-RL algorithms.

=== Does SoRB's learned knowledge generalize to new, unseen environments?

- *Experiment Design:* To test generalization, the authors trained a single SoRB
  agent on 100 different 3D houses from the SUNCG dataset. They then evaluated
  this trained agent on a held-out test set of 22 new houses it had never seen
  before. For the search component in these new houses, the replay buffer was
  populated with just 1000 observations collected by a random agent, meaning the
  planner had to work with a sparse, unstructured sampling of the new
  environment.

- *Results & Metrics:*
  - *Metric:* *Success rate* versus goal distance was measured across the 22
    held-out houses, and the experiment was repeated with three different random
    seeds for robustness.
  - *Results:* SoRB demonstrated strong generalization. In new houses, it
    reached almost *80% of goals that were 10 steps away*, compared to less than
    20% for the underlying goal-conditioned RL policy. For very distant goals
    (20 steps away), SoRB still succeeded about 40% of the time, while the base
    RL policy's success rate was near zero. The results were highly consistent
    across the three random seeds, indicating the method is robust.

- *Significance:* This is arguably the paper's strongest result. It shows that
  the learned distance function and policy are not just memorizing routes in a
  specific environment but are learning a generalizable navigation skill. The
  ability to plan effectively in a new environment using only a small, random
  collection of states for the graph is a significant step towards creating
  truly autonomous agents that can adapt to novel settings.

- *Limitations:* The discussion doesn't mention any specific limitations of the
  generalization experiment itself. However, a general limitation of the SoRB
  framework is its "stage-wise" procedure: it first learns a policy and then
  applies search, without the search process feeding back to improve the policy
  during training. The authors suggest that exploring this feedback loop,
  perhaps via policy distillation, is a key area for future work.
