#import "notes_template.typ": *
#import "@preview/algorithmic:1.0.0"
#import algorithmic: style-algorithm, algorithm-figure

#show: dvdtyp.with(
  title: "Reinforcement Learning Notes", subtitle: [Spring 2025], author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

= From Markov Chains to Markov Decision Processes

== Markov Chain

- A *Markov chain* is a probabilistic model used to describe a system that
  transitions from one state to another
- It is often used to model how the *state of a randomly changing system*, changes
  over discrete time steps
- It is parameterized by the Transition/Dynamics model, $cal(T)$, a matrix of
  transition probabilities, where each entry in the matrix represents the
  probability of moving from one state to another

$
  cal(T) = mat(
    P(s_1|s_1), P(s_2|s_1), ..., P(s_n|s_1);P(s_1|s_2), P(s_2|s_2), ..., P(s_n|s_2);dots.v, dots.v, dots.down, dots.v;P(s_1|s_n), P(s_2|s_n), ..., P(s_n|s_n);
  )
$

- Notice $cal(T)_(i, j) = P(s_j|s_i)$, thus
  - the $i^"th"$ row consists of the probabilities of entering each possible state
    from state $s_i$
  - the $j^"th"$ column consists of the probabilities of entering state, $s_j$ from
    each of the possible states
  - the diagonal consists of the probabilities of remaining in the same state

#definition(
  "Markov Property",
)[
  The probability of transitioning to any future state depends only on the current
  state and not its history:
  $
    P(s_(t+1) | s_t) = P(s_(t+1) | s_0, ..., s_t)
  $
]

- The Markov property of Markov chains and its extensions, allows us to assume
  that we only have to consider the current state in any analyses of the future
- The Markov Chain formulation is useful as it can help us describe the state
  dynamics of a system -- however, just having a mathematical formulation of the
  dynamics of a system is like just having a probability distribution of a random
  variable, but not knowing the anything else about the variable -- _how are we going to analyze it if it doesn't come with a quantity?_

== Markov Reward Process

#definition[*Markov Reward Process* $cal(M) = angle.l S, cal(T), R, gamma angle.r$][
  - State $s in S$: The agent’s perception of the environment
  - Reward $r in R$: Immediate feedback received _after_ a state transition
  - Environment $cal(T)(s_(t+1) | s_t, a_t)$: World the agent interacts with
]

- The Markov Reward Process extends the Markov Chain with the notion of reward, a
  real value quantifying the general 'goodness' of a state _transition_
- However, a reward only quantifies the desirability of a single state transition;
  it fails to provide any notion of the future reward -- which is especially
  important since our analysis pertains to a system that changes over time (a
  Markov Process)

== Return Function

#definition[*Return Function*][
  Estimate how good a state is, in terms of expected sum of future rewards, _with respect to a specific policy_
  $
    G_t = sum_(k=0)^H gamma^k r_(t+k+1)
  $
]

- Since we are

== Markov Decision Process

#definition[*Markov Decision Process* $cal(M) = angle.l S, A, cal(T), R, gamma angle.r$][
  - State $s in S$: The agent’s perception of the environment
  - Action $a in A$: Choices the agent makes
  - Reward $r in R$: Immediate feedback received after an action
  - Policy $pi: S arrow.r A$: The agent’s strategy for choosing actions
  - Environment $cal(T)(s_(t+1) | s_t, a_t)$: World the agent interacts with
]

== Value Function

#definition[*Value Function*][
  Estimate how good a state is, in terms of expected sum of future rewards, _with respect to a specific policy_
  $
    V^pi (s) = sum_(t=0)^infinity gamma^t R lr((s_t, a_t = pi(s_t)))
  $
]
i So.. easy right? Just do this:
+ Collect trajectories, $tau = {(s_0, a_0, r_1), ... (s_T, a_T,)}$
+ Estimate $V^pi$ from trajectories
+ Update $pi$ parameters $theta := theta + alpha gradient_theta V^(pi_theta)$
- Not really, since $V^pi$ is by definition, the rewards w.r.t. a *_specific_* $pi$,
  for *_all future timesteps_*, updating $theta$ alters $cal(P)_text("state")^pi (s)$,
  invalidating $V^pi$

- Like in SL, distribution shifts are a large, overarching problem in RL
  - which PPO mitigates pretty well, by clipping/regularizing $theta$ updates

== theme: Recursive Definitions

$
  G_t &= sum_(k=0)^H gamma^k r_(t+k+1) \
      &= gamma^0 r_(t+1) + sum_(k=0)^H gamma^(k+1) r_(t+k+2) \
      &= gamma^0 r_(t+1) + gamma sum_(k=0)^H (gamma^k r_((t+1)+k+1)) \
      &= r_(t+1) + gamma G_(t+1)
$

- Bootstrapping:
- *Concept*: In RL, bootstrapping refers to the idea that an estimate of the value
  function is updated using another estimate of the value function
  - That is, instead of waiting until the actual return (cumulative reward) is
    observed, we use the current estimates to update future estimates
- *Why it's useful*: It enables *efficient learning* by leveraging prior
  knowledge, reducing the need for full rollouts of an episode before updating the
  value function.
- *Key property*: It relies on the assumption that previous estimates are
  reasonably accurate, even though they might be biased or incomplete

- in RL, bootstrapping is linked to [dynamic
  programming](2_concepts/dynamic-programming-rl.md)
- In [Temporal Difference (TD)
  Learning](temporal-difference-policy-evaluation.md), for example, the value of a
  state $V(s)$ is updated using the value of the next state $V(s')$, rather than
  waiting for the actual return:

$
  V(s_t) := V(s_t) + alpha (r + gamma V(s_(t+1)) - V(s_t))
$

- This is an example of bootstrapping because we're using an *existing estimate* ($V(s')$)
  rather than actual future rewards.

= Learning Policies with Models

== Control and Policy Evaluation

- *Policy Evaluation* and *Control* are two distinct *tasks* in reinforcement
  learning, and many algorithms integrate both of them
- Policy evaluation can be viewed as a sub-task of control, where the value of the
  current policy is estimated in order to improve it iteratively toward an optimal
  policy

=== Policy Evaluation
- *TASK*: estimate the [value function](2_concepts/value-functions.md) $V^pi(s)$ or $Q^pi(s, a)$ for
  a given policy $pi$
  - i.e. "How good is it to follow this policy $pi$ from a given state or
    state-action pair?"

- *Goal*: To determine the expected cumulative reward when the agent follows a
  specific policy
- Examples:
  - [Temporal Difference](2_concepts/temporal-difference-policy-evaluation.md)
    methods like *TD(0)*
  - [Monte Carlo](2_concepts/monte-carlo-policy-evaluation.md) methods
  - Dynamic Programming methods for policy evaluation (Bellman expectation equation)

=== Control
- *TASK*: *improve* the policy, aiming to find the *optimal policy* that maximizes
  long-term rewards
  - evaluates the current policy
  - ALSO updates it iteratively to become more optimal

- *Goal*: To find the best policy $pi^*$ that maximizes the expected reward.
- Examples that combine policy evaluation and policy improvement. In these control
  algorithms, the value function is updated, and the policy is improved
  simultaneously:
  - [q-learning](2_concepts/q-learning.md)
  - [SARSA](2_concepts/SARSA.md)

=== Relationship Between Policy Evaluation and Control:
In many reinforcement learning algorithms, *policy evaluation* and *control* are
combined into a single process. Control algorithms often include both evaluation
and improvement steps:

- *Policy Iteration*: This is a classic algorithm that explicitly separates policy
  evaluation and control (policy improvement). It alternates between:
  1. *Policy evaluation*: Evaluates the current policy.
  2. *Policy improvement*: Updates the policy to improve it based on the current
    value estimates.

  These steps are repeated until convergence to the optimal policy.

- *Value Iteration*: This is a control algorithm that effectively combines policy
  evaluation and policy improvement in each step. It updates the value function
  using the Bellman optimality equation without explicitly evaluating a policy.
  The process inherently improves the policy as the value function is updated.

- *SARSA and Q-learning*: These TD-based control methods integrate both tasks.
  They evaluate the current policy by updating the value function and implicitly
  improve the policy through exploration-exploitation strategies.

== Dynamic Programming

- The main methods for dynamic programming in RL are [policy and value
  iteration](2_concepts/policy-value-iteration.md)
- These methods are useful only in *model-based, tabular settings* with perfect
  knowledge of the environment
  - *they assume full knowledge of the environment's dynamics*, i.e., the transition
    probabilities $P(s'|s,a)$ and rewards $R(s,a)$
- These methods use initial estimates of a Value function and improve these
  estimates using [bootstrapping](2_concepts/bootstrapping.md)

- The goal of either of these methods is to find the *optimal policy* $pi^*$ maximizes
  the expected return from any state
- These are examples of [dynamic programming rl
  algorithms](2_concepts/dynamic-programming-rl.md)
- They are guaranteed to have monotonic improvement
- They are only for tabular methods since we must update our estimates for each
  state in our environment

=== Value Iteration

- Calculates the *exact, optimal* policy over a specific, finite horizon
- As $H arrow.r inf$, the value function should converge to the overall optimal
  policy

- Iteratively update the value function using the Bellman optimality equation:
$
  V_(k+1)(s) = max_a [ R(s, a) + gamma sum_{s'} P(s'|s, a) V_k(s') ]
$

- Once $V(s)$ converges, the optimal policy is:
$
  pi^*(s) = arg max_a [ R(s, a) + gamma sum_{s'} P(s'|s, a) V(s') ]
$

=== Policy Iteration

- Start with an estimate of the optimal policy and iterate upon it to improve it
- Alternates between policy evaluation (updating $V^pi (s)$ estimate) and policy
  improvement (updating $pi(s)$):

- *Policy Evaluation*: For all for all $s in SS$, compute $V^pi (s)$ for a given
  policy $pi$
$
  V_(k+1)(s) = sum_a pi(a|s) [ R(s, a) + gamma sum_(s') P(s'|s, a) V_k(s') ]
$

- *Policy Improvement*: For all for all $s in SS$, update the policy by choosing
  the action that maximizes the state-action value function:
$
  pi_(i+1)(s) = arg max_a [ R(s, a) + gamma sum_(s') P(s'|s, a) V^pi(s')]
$

==== Matrix Form

- For a finite state space, the Bellman equation for an MDP can be written in
  matrix form. Let:
  - $V$ be the value function vector
  - $R_a$ be the reward vector for action $a$
  - $P_a$ be the transition matrix for action $a$

- The Bellman equation for *policy evaluation* becomes:
$
  V^pi = R_pi + gamma P_pi V^pi
$

- Where:
  - $R_pi(s) = sum_a pi(a|s) R(s, a)$ is the reward vector under policy $pi$.
  - $P_pi(s'|s) = sum_a pi(a|s) P(s'|s, a)$ is the transition matrix under policy $pi$.

- For *value iteration*, the optimal Bellman equation is:
$
  V_{k+1}(s) = max_a [ R(s, a) + gamma sum_(s') P(s'|s, a) V_k(s') ]
$

- In matrix form, this becomes:
$
  V_(k+1) = max_a ( R_a + gamma P_a V_k )
$

= Policy Evaluation

== Monte Carlo Policy Evaluation

- *Key Idea*: Estimate value functions using the average of observed returns over
  multiple episodes (trajectories)

- does not require MDP dynamics/rewards ([model-free](2_concepts/model-free.md))
- no [bootstrapping](2_concepts/bootstrapping.md)
- does not assume state is Markov
- can only be applied to episodic MDPs
  - averaging over returns from a complete episode
  - episode must end before its data can be used to update the value function
- generally high variance estimator
  - reducing variance can require a lot of data

- *First-Visit Monte Carlo*: Only considers the first occurrence of each state
  within an episode
  - this makes $V^pi$ an *unbiased* estimator, as each episode is
    [i.i.d.](2_concepts/individually-identically-distributed.md)
- *Every-Visit Monte Carlo*: Considers every occurrence of each state within an
  episode
  - using multiple states per episode makes $V^pi$ a *biased* estimator, because
    while each episode is independent the individual state-action-reward pairs are
    within an episode are not
    [i.i.d.](2_concepts/individually-identically-distributed.md)
  - this method is still empirically better, we can have more updates, more often

=== First-Visit/Every-Visit MC Algorithms

- Initialize $V(s)$, $N(s)$, $G(s)$ to $arrow.l 0$ for all $s in S$
- For each episode, $i$: $[(s_{i,1}, a_{i,1}, r_{i,1}), (s_{i,2}, a_{i,2}, r_{i,2}), ..., (s_{i,T}, a_{i,T}, r_{i,T_i})]$
  - For each ($s$ encountered for the first time in the episode _OR_ time step $t$)
    - $G_{i,t} = r_t + gamma r_{i, t+1} + gamma^2 r_{i, t+2} + ... + gamma^{T_i-1} r_{i, T_i}$
    - $N(s) = N(s) + 1$ (increment the counter for state $s$)
    - $G(s) = G(s) + G_t$ (add the return to the total return for $s$)
    - Value estimate: $V(s) = G(s)/N(s)$
      - can be rewritten as $V(s) = V(s) + alpha(G_{i,t}-V(s))$ where $alpha=1/N(s)$
        - when $alpha>1/N(s)$ older values are forgotten over time

=== Extension to Q Function

- When extending the original state value function $V$ to the state-action value
  function, $Q$, the only thing modified is that we act on state-action tuples
  rather than just states (obviously)
- During the policy improvement step, this lets us do $pi = arg max_a Q(s, a)$

- Initialize $Q^pi(s, a)$, $N(s, a)$, $G(s, a)$ to $arrow.l 0$ for all $s in S, a in A$
- For each episode, $i$: $[(s_{i,1}, a_{i,1}, r_{i,1}), (s_{i,2}, a_{i,2}, r_{i,2}), dots, (s_{i,T}, a_{i,T}, r_{i,T_i})]$
  - For each ($s$ encountered for the first time in the episode _OR_ time step $t$)
    - $N(s, a) = N(s, a) + 1$
    - $G(s, a) = G(s, a) + G_t$
    - Value estimate: $Q^pi(s, a) = Q^pi(s, a) + alpha(G_{i,t}-Q^pi(s, a))$

- TD learning is a [model free](2_concepts/model-free.md) method for *policy
  evaluation* that combines ideas from *Monte Carlo* methods and *dynamic
  programming*
- It updates value estimates incrementally, using the current reward and the value
  of the next state, without requiring complete episodes.

== Temporal Difference Learning

=== Key Idea

- *[Bootstrapping](2_concepts/bootstrapping.md)*: Unlike Monte Carlo methods,
  which wait until the end of an episode to calculate the return, TD learning
  updates the value estimate $V(s)$ after each state transition

=== TD(0) Learning Algorithm

- Algorithm:
  - Initialize: $V(s) = 0$ for all $s in S$
  - Loop (until convergence):
    - sample $(s_t, a_t, r_t, s_{t+1})$
    - $V(s_t) arrow.l V(s_t) + alpha overbrace(
        underbrace(r_t + gamma V(s_(t+1)), "TD Target") - V(s_t), delta_t", TD Error",

      )$

- very similar to q-learning except we are fixing a policy here
- generally you sample in order

=== Key Concepts:

1. *TD Target*, $r_t + gamma V(s_{t+1})$
  - estimated return for the current state $s_t$ after observing the next state $s_{t+1}$.
    It combines the immediate reward $r_t$ and the discounted estimate of the value
    of the next state $V(s_{t+1})$
  - Normally, the TD target would be the expected value of $V(s_{t+1})$, however, we
    bootstrap by using the previous estimation
    - We don't have the transition model so we cannot easily calculate the previous
      estimation

2. *TD Error*: $delta_t = r_t + gamma V(s_{t+1}) - V(s_t)$
  - The TD error measures the difference between the current value estimate $V(s_t)$ and
    the updated target
  - The error is used to adjust the value estimate
  - TD Error doesn't necessarily go to zero, since it is a sample, and not the
    expectation
    - it only goes to zero when the transition is deterministic

3. *Incremental Update*:
  - The value estimate $V(s_t)$ is updated after each transition, using the TD error
    and learning rate $alpha$.
$
  V(s_t) arrow.l V(s_t) + alpha delta_t
$
- The learning rate $alpha$ determines how much to adjust the value based on the
  TD error

=== Characteristics of TD Learning:

- *Model-Free*: TD learning does not require knowledge of the MDP’s dynamics
  (i.e., transition probabilities or reward function), only the immediate
  experience
- *Bootstrapping*: TD learning updates the value estimates based on other
  estimated values (i.e., using $V(s_{t+1})$), rather than waiting for a complete
  return as in Monte Carlo methods
- *On-Policy*: The value function is learned under the policy that is currently
  being followed (though there are off-policy TD methods like Q-learning)

=== Advantages of TD Learning

- *Data Efficiency*: TD learning updates value estimates after every transition,
  making it more data-efficient than Monte Carlo methods, which require waiting
  until the end of an episode
- *Can be used in continuous tasks*: TD learning works well in tasks without
  terminal states, whereas Monte Carlo requires episodes to end
- *Low variance*: TD learning typically has lower variance than Monte Carlo
  because it updates more frequently and uses bootstrapped estimates
  - bootstrapping estimates makes it more

=== Disadvantages

- *Bias*: Since TD updates are based on current value estimates, there can be bias
  in early estimates, especially if the initialization of $V(s)$ is poor

=== Comparison to MC

- Monte Carlo in batch settings converges to minimizing
  [MSE](2_concepts/mean-squared-error.md)
  - minimize loss wrt observed returns
- TD(0) converges to DP policy $V^pi$ for the MDP with the maximum likelihood
  model estimates \#TODO : explain based on lecture 3 stanford end

= Policy Iteration

== SARSA

- An on-policy [control](2_concepts/control-policy-evaluation.md) algorithm based
  on [TD policy evaluation](2_concepts/temporal-difference-policy-evaluation.md)
- Chooses an action, not necessarily the best one, sees the result, then updates
  it’s value function with that knowledge
- will converge eventually, but more slowly than
  [q-learning](2_concepts/q-learning.md)

=== Algorithm

- Initialize the action-value function $Q(s, a)$ for all state-action pairs
  arbitrarily, except for terminal states where $Q(s_"terminal", dot) = 0$
- For each episode:
  - Initialize the starting state $S$
  - choose an action $A$ based on the current policy derived from $Q(s, a)$
  - For each step within the episode:
    - Take action $A$, observe the reward $R$ and the next state $S'$
    - Choose the next action $A'$ from the new state $S'$ using the policy
    - $Q(S, A) arrow.l Q(S, A) + alpha ( R + gamma Q(S', A') - Q(S, A))$
    - Update the state $S arrow.l S'$ and action $A arrow.l A'$

- note:
  - we update $Q(S, A)$ with the same $Q(S, A)$ choose actions

=== Expected Sarsa

- A variation of the SARSA algorithm, where instead of using the *sampled value of
  the next state-action pair*, it uses the *expected value* of the next
  state-action value function
- This method calculates the expected return by taking into account all possible
  actions in the next state and weighting them according to the policy.

==== Algorithm:

- Initialize the action-value function $Q(s, a)$ for all state-action pairs
  arbitrarily, except for terminal states where $Q("terminal-state", dot) = 0$
- For each episode:
  - Initialize the starting state $S_t$
  - Choose an action $A_t$ based on the current policy derived from $Q(s, a)$
  - For each step within the episode:
    - Take action $A_t$, observe the reward $R_{t+1}$ and the next state $S_{t+1}$
    - $Q(S_t, A_t) arrow.l Q(S_t, A_t) + alpha [ R_{t+1} + gamma sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]$

    - Update the state $S_t arrow.l S_{t+1}$ and action $A_t arrow.l A'$

=== Comparison:

- *SARSA:* Updates based on the actual next action taken
- *Expected Sarsa:* Updates based on the expected value of all actions at the next
  state, weighted by the policy probabilities.
- *Performance:* Expected Sarsa typically performs better than SARSA in terms of
  stability and convergence speed, though it involves more computation as it
  requires calculating the expected value over all actions.

== Q Learning

- An off-policy [control](2_concepts/control-policy-evaluation.md) algorithm based
  on [TD policy evaluation](2_concepts/temporal-difference-policy-evaluation.md)
- Unlike [SARSA](2_concepts/SARSA.md), Q-learning's off-policy nature
- Q-learning updates the action-value function $Q(s, a)$ using the *maximum*
  future value, regardless of the action taken by the current policy
- This allows the algorithm to converge faster towards the *optimal policy*

=== Algorithm

- Initialize the action-value function $Q(s, a)$ for all state-action pairs
  arbitrarily, except for terminal states where $Q(s_"terminal", dot) = 0$
- For each episode:
  - Initialize the starting state $S$
  - For each step within the episode:
    - choose an action $A$ based on the current policy derived from $Q(s, a)$
    - take action $A$, observe the reward $R$ and the next state $S'$
    - Choose the next action $A'$ from the new state $S'$ using the policy
    - $Q(S, A) arrow.l Q(S, A) + alpha ( R + gamma max_a Q(S', a) - Q(S, A) )$
    - Update the state $S arrow.l S'$

=== Double Q-Learning

- A variant of Q-learning designed to mitigate the overestimation bias in
  Q-learning by decoupling action selection from action evaluation.
- Instead of using one Q-value function, Double Q-Learning trains two independent
  Q-value functions, $Q_1$ and $Q_2$.

=== Algorithm

- Initialize two action-value functions $Q_1(s, a)$ and $Q_2(s, a)$ arbitrarily,
  except for terminal states where $Q_1(s_"terminal", dot) = Q_2(s_"terminal", dot) = 0$.
- *For each episode*:
  - Initialize the starting state $S_t$
  - For each step within the episode:
    - Choose action $A_t$ based on $Q(s, a) = Q_1(s, a) + Q_2(s, a)$
      - Take action $A_t$, observe reward $R_{t+1}$, and transition to the next state $S_{t+1}$
    - Update either $Q_1$ or $Q_2$ with equal probability
      - If updating $Q_1$, use $Q_2$ for the next state's value $Q_1(S_t, A_t) arrow.l (1 - alpha)Q_1(S_t, A_t) + alpha (R_{t+1} + gamma Q_2(S_{t+1}, arg max_a Q_1(S_{t+1}, a))$
      - If updating $Q_2$, use $Q_1$ for the next state's value: $Q_2(S_t, A_t) arrow.l (1-alpha)Q_2(S_t, A_t) + alpha (R_{t+1} + gamma Q_1(S_{t+1}, arg max_a Q_2(S_{t+1}, a)))$
    - Update the state $S arrow.l S'$

= Deep Reinforcement Learning

= Policy Gradient Methods

== Derivation

- Previous methods didn't require that we learned a policy at all the 'policy' was
  simply defined implicitly via $Q(s, a)$ as $pi(s) = arg max_(a in A) Q(s, a)$,
- Notice that we perform an $arg max$ over $a in A$, this operation is only
  tractable given tabular $A$; thus is we want a policy with a continuous action
  space, a different method is required
- Challenge: can we directly optimize parameters of a policy to maximize $V$?
- Let:
  - $pi_theta$: stochastic policy defined as a probability distribution of actions, $a$,
    given state, $s$: $P(a|s)$
  - $R(tau)$: reward of a trajectory, previously we would describe this as the
    return, $G(s)$
  - $V_theta (s)$: state-value function; expected sum of rewards acting on the
    current policy
  - $Q_theta (s, a)$: state action value function, implicitly defined as:
$
  Q_theta(s, a) = R(s, a) + EE_(tau ~ pi_theta)[R(tau)]
$

== REINFORCE

== Vanilla Policy Gradient

== Proximal Policy Optimization

= Q-Learning Based Policy Gradient Methods

== Common Features

- off policy
- replay buffer
- higher sample efficiency via n-step returns

#pagebreak()

== Soft Q-Learning

- Challenges 1: most deep rl methods assume that the optimal solution, at least
  under full observability, is always a deterministic policy
  - since stochasticity is desirable for exploration, we often use heuristic
    methods, such as injecting noise or initializing stochastic policies with high
    entropy

- Hypotheses: we might actually prefer to learn stochastic behaviors to
  + exploration in the presence of multimodal objectives and compositionality
    attained via pretraining
  + robustness in the face of uncertain dynamics

- Challenge 1: what objective can we define that promotes stochasticity?
  - Hypothesis: framing control as inference produces policies that aim to capture
    not only the single deterministic behavior that has the lowest cost, but the
    entire range of low-cost behaviors, explicitly maximizing the entropy of the
    corresponding policy

- Goal: Learn a policy with high entropy:

- standard goal
$
  pi^*_"std" = arg max_pi sum_t EE_((s_t, a_t) ~ rho_pi) [r(s_t, a_t)]
$

- modified goal

$
  pi^*_"MaxEnt" = arg max_pi sum_t EE_((s_t, a_t) ~ rho_pi) [r(s_t, a_t) + alpha cal(H)(pi(dot | s_t))]
$

- this differs from previous methods that greedily maximize entropy at the current
  time step, but do no explicitly optimize for policies that aim to reach _states_ where
  they will have high entropy in the _future_
-

== Soft-Actor Critic

== TD3: Addressing Function Approximation Error in Actor-Critic Methods

=== Key Contribution 1: Addressing Overestimation Bias

==== A. Double Q-Learning

- In [DQN](2_concepts/deep-q-networks.md), two sets of parameters are used for the
  Deep Q Network, giving a "Prediction Network" and a "Target Network" to avoid _maximization bias_
- The "Prediction" parameters were updated each batch as per standard practice,
  while the policy was evaluated using the "Target" parameters, which are a
  delayed copy of the "Prediction" parameters which is periodically synchronized
- This method was called Double DQN, not to be confused with [Double
  Q-learning](2_concepts/double-q-learning.md)
- Unfortunately, due to the slow-changing policy in an actor-critic setting, the
  current and target value estimates remain too similar to avoid maximization bias

==== B. Clipped Double Q-Learning

- While Double Q-Learning allows for a less biased value estimation, even an
  unbiased estimate with high variance can still lead to future overestimations in
  local regions of state space, which in turn can negatively affect the global
  policy
- Double Q-learning is more effective, it does not entirely eliminate the
  overestimation, to improve this _clipping_ was introduced
- In original [Double Q-learning](2_concepts/double-q-learning.md), the two Q
  functions are used to update each other in their TD error updates, here, we
  further avoid overestimation bias by taking the $\min$ between both estimates

$
  y_1 = r + gamma min_(i in {1, 2}) Q_i (s', pi_phi_1(s'))
$

- In implementation, computational costs can be reduced by using a single actor
  optimized with respect to $Q_(theta_1)$, and use the same target $y_2 = y_1$ for $Q_theta_2$
- If $Q_theta_2 gt Q_theta_1$
  - the update is identical to the standard update and induces no additional bias
- else:
  - overestimation has occurred and the value is reduced by clipping

=== Key Contribution 2: Addressing Variance

==== A. Delayed Updates

= Model Based Methods: Using a Learned Model

== TD-MPC

=== Model Predictive Control

- given environment and horizon, we want to figure out what actions will maximize
  a the reward

- so, at each step we want select the action, $a$, such that:

$
  arg max_a = sum_(l=0)^(H-1) R(s_(t+l), a_(t+l)) + Q(s_(t+h), a_(t+h))
$

- this is repeated at each time step, however, we are going to have to learn a the
  state dynamics and reward function

- this will need to be efficient if we want real-time (low-latency) control

= Contrastive Reinforcement Learning

== Contrastive Goal Conditioned Reinforcement Learning

= Imitation Learning

- Given a dataset how can we train a policy that imitates the behavior of that
  dataset?

== Inverse Optimal Control / Inverse RL

- given:
  - state & action space
  - rollouts from an expert policy, $pi^*$
  - dynamics model
- goal:
  - recover reward function
  - user reward to get policy

Challenges:
+ underdefined problem
+ difficult to evaluate a learned reward
+ demonstrations may not be precisely optimal

== Maximum Entropy Inverse RL

- Trajectory $tau = {s_1, a_1, ..., s_t, a_t, ..., s_T}$
- learned reward: $R_psi (tau) = sum_t r_psi (s_t, a_t)$
- expert demonstrations $cal(D): {tau_i} ~ pi^*$

=== MaxEnt Formulation

- Hypothesis: while there may be infinitely many solutions for the reward
  function, we shall solve for the one which maximizes entropy

- Given our assumption that our dataset contains rollouts from an "expert" policy,
  we can assume that the more often a trajectory is found in the dataset, the more
  desirable it is
- So, we define the relationship between the probability of a trajectory, $p(tau)$ and
  the reward function we want to learn, $r_psi (s)$ as:
$
  p(tau) = 1/Z exp(R_psi (tau)), "where:" \
  R_psi = sum_(s in tau) r_psi (s) \
  Z = sum_(tau in cal(P)) exp(R_psi (tau)) \
$

- Then with this probability, we can simply perform MLE (maximizing the log
  likelihood) to optimize the reward function
$
  max_psi sum_(tau in cal(D)) ln p_r_psi (tau)
$

- To do so, we must simplify our objective function:

$
  max_psi cal(L)(psi) &= sum_(tau in cal(D)) ln p_r_psi (tau)\
                      &= sum_(tau in cal(D)) ln 1/Z exp R_psi (tau) \
                      &= sum_(tau in cal(D)) R_psi (tau) - ln Z \
                      &= (sum_(tau in cal(P)) R_psi (tau)) - ( M ln Z ) \
                      &= sum_(tau in cal(D)) R_psi (tau) - M ln sum_(tau in cal(P)) exp(R_psi (tau)) \
$

- To optimize $R_psi$'s parameters with SGD we must calculate $gradient_psi cal(L)$:

$
  gradient_psi cal(L)(psi) &= sum_(tau in cal(D)) (R_psi (tau))/(dif psi) - gradient_psi M ln sum_(tau in cal(P)) exp(R_psi (tau)) \
                           &= sum_(tau in cal(D)) (R_psi (tau))/(dif psi) - M 1/(sum_(tau in cal(P)) exp(R_psi (tau))) sum_(tau in cal(P)) exp(R_psi (tau)) (R_psi (tau))/(dif psi) \
$

- So far, we have been avoiding an integral problem, how we plan on evaluating the
  partition function, $Z$, the sum over all possible trajectories
- Notice, here, we denote the trajectories in the current dataset/batch as $tau in cal(D)$ but
  denote all possible trajectories as $tau in cal(P)$ where $cal(P)$ is the
  population of trajectories

- By further rearranging $gradient_psi cal(L)$, we can find a term that we
  previously defined as $p(tau)$
$
  gradient_psi cal(L)(psi) &= sum_(tau in cal(D)) (dif R_psi (tau))/(dif psi) - M 1/(sum_(tau in cal(P)) exp(R_psi (tau))) sum_(tau in cal(P)) exp(R_psi (tau)) (dif R_psi (tau))/(dif psi) \
                           &= sum_(tau in cal(D)) (dif R_psi (tau))/(dif psi) - M sum_(tau in cal(P)) underbrace(1/Z exp(R_psi (tau)), eq.delta p(tau)) (dif R_psi (tau))/(dif psi) \
                           &= sum_(tau in cal(D)) (dif R_psi (tau))/(dif psi) - M sum_(tau in cal(P)) underbrace(
    1/Z exp(R_psi (tau)) (dif R_psi (tau))/(dif psi), eq.delta gradient_psi p(tau),

  )
$
$
  therefore gradient_psi cal(L)(psi) = sum_(tau in cal(D)) (dif R_psi (tau))/(dif psi) - M sum_(tau in cal(P)) gradient_psi p(tau)\
  "where:" gradient_psi p(tau) = 1/Z exp(R_psi (tau)) (dif R_psi (tau))/(dif psi)
$

- We can further evaluate to find that we can remove the sum over $tau in cal(P)$ by
  equating this term to the state visitation frequency, $p(s|psi)$:

$
  sum_(tau in cal(P)) gradient_psi p(tau) &= sum_(tau in cal(P)) p(tau|psi)(dif R_psi (tau))/(dif psi) \
                                          &= sum_(s in S) p(s|psi)(dif r_psi (s))/(dif psi) \
$

#show: style-algorithm
#algorithm-figure(
  "Max Entropy", vstroke: .5pt + luma(200), {
    import algorithmic: *
    Procedure(
      "MaxEnt-Inverse-RL", ($tau$,), {
        Comment[Initialize the search range]
        Assign[$mu_t (s)$][$?$]
        Comment[Initialize $psi$, gather demonstrations $cal(D)$]
        LineBreak
        While(
          [not done], {
            Comment[Solve for optimal policy $pi(a|s)$ w.r.t. reward, $r_psi$]
            Comment[Solve for state visitation frequencies $mu(s|psi)$]
            Comment[Compute gradient]
            Assign[$Delta_psi$][$1/(|cal(D)|) sum_(tau_d in cal(D)) (dif r_psi)/(dif psi) ( tau_d ) - sum_s p(s|psi) (dif r_psi)/( dif psi ) (s)$]
            Comment[Update $psi$ with one gradient step]
            Assign[$psi$][$psi + alpha Delta_psi$]
          },
        )
      },
    )
  },
)

- note: this method requires that we solve for both the optimal policy, $pi^*$ and
  the state visitation frequencies, $mu(s)$, at each time step
- how can we (1) handle unknown dynamics, (2) avoid solving the mdp in the inner
  loop

== Guided Cost Learning

- only takes one policy step at once\ s
- if we dont know dyn, we cant analytically zompute $Z$
- for IS to get the best estimate, of Z, we want to sample from the distribution
  whose probabilities are proportional to the absolute value of the exponential of
  the reward function
- note that this is the minimum variance solution for IS for the distribution to
  sample from
- its gonna be hard pick a distribution to sample from and sample from that
- instead we adaptively sample to estimate $Z$, as we get a better estimate of the
  R fn were gonna construct a sampling distribution that is proportional to the
  exponential of the reward function
- were gonna do this by constructing and sampling from a policy

+ generate samples from pi then using the samples, update the reward function with
  the samples generated from the policy as $cal(P)$ and the samples from the
  demonstration as $cal(D)$
+ once you have the estimate of the reward function, we update the policy using
  that reward function, which in turn gives a better estimate of our partition
  function

#include "rl/diffusion-policy.typ"
#include "rl/streaming-drl.typ"
#include "rl/streaming-diffusion-rl.typ"
#include "rl/student-informed-teacher-rl.typ"
#include "rl/sorb.typ"
