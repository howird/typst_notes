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

- Given our assumption that our dataset contains rollouts from an "expert"
  policy, we can assume that the more often a trajectory is found in the
  dataset, the more desirable it is
- So, we define the relationship between the probability of a trajectory,
  $p(tau)$ and the reward function we want to learn, $r_psi (s)$ as:
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

- To optimize $R_psi$'s parameters with SGD we must calculate
  $gradient_psi cal(L)$:

$
  gradient_psi cal(L)(psi) &= sum_(tau in cal(D)) (R_psi (tau))/(dif psi) - gradient_psi M ln sum_(tau in cal(P)) exp(R_psi (tau)) \
  &= sum_(tau in cal(D)) (R_psi (tau))/(dif psi) - M 1/(sum_(tau in cal(P)) exp(R_psi (tau))) sum_(tau in cal(P)) exp(R_psi (tau)) (R_psi (tau))/(dif psi) \
$

- So far, we have been avoiding an integral problem, how we plan on evaluating
  the partition function, $Z$, the sum over all possible trajectories
- Notice, here, we denote the trajectories in the current dataset/batch as
  $tau in cal(D)$ but denote all possible trajectories as $tau in cal(P)$ where
  $cal(P)$ is the population of trajectories

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

- We can further evaluate to find that we can remove the sum over
  $tau in cal(P)$ by equating this term to the state visitation frequency,
  $p(s|psi)$:

$
  sum_(tau in cal(P)) gradient_psi p(tau) &= sum_(tau in cal(P)) p(tau|psi)(dif R_psi (tau))/(dif psi) \
  &= sum_(s in S) p(s|psi)(dif r_psi (s))/(dif psi) \
$

#show: style-algorithm
#algorithm-figure(
  "Max Entropy",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "MaxEnt-Inverse-RL",
      ($tau$,),
      {
        Comment[Initialize the search range]
        Assign[$mu_t (s)$][$?$]
        Comment[Initialize $psi$, gather demonstrations $cal(D)$]
        LineBreak
        While(
          [not done],
          {
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

- note: this method requires that we solve for both the optimal policy, $pi^*$
  and the state visitation frequencies, $mu(s)$, at each time step
- how can we (1) handle unknown dynamics, (2) avoid solving the mdp in the inner
  loop

== Guided Cost Learning

- only takes one policy step at once\ s
- if we dont know dyn, we cant analytically zompute $Z$
- for IS to get the best estimate, of Z, we want to sample from the distribution
  whose probabilities are proportional to the absolute value of the exponential
  of the reward function
- note that this is the minimum variance solution for IS for the distribution to
  sample from
- its gonna be hard pick a distribution to sample from and sample from that
- instead we adaptively sample to estimate $Z$, as we get a better estimate of
  the R fn were gonna construct a sampling distribution that is proportional to
  the exponential of the reward function
- were gonna do this by constructing and sampling from a policy

+ generate samples from pi then using the samples, update the reward function
  with the samples generated from the policy as $cal(P)$ and the samples from
  the demonstration as $cal(D)$
+ once you have the estimate of the reward function, we update the policy using
  that reward function, which in turn gives a better estimate of our partition
  function
