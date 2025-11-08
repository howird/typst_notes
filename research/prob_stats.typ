#import "styles/notes_template.typ": *

#show: note.with(
  title: "Probablility and Statistics Notes",
  subtitle: "",
  author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

= Entropy

#definition[Entropy][
  The entropy of a random variable, x, is:
  $
    cal(H)(x) = - integral p(x) ln p(x) dif x
  $
]

#problem[
  If a less certain probability distribution has a higher entropy, why is it the
  case that less certain distributions have samples, x, with $p(x) arrow 0^+$
  which causes $ln p(x) arrow infinity$?

  We can prove this by evaluating $lim_(p(x)->0^+) p(x)ln p(x)$

  Let $z$ be $p(x)$, $f(z) = ln z$ $g(z) = 1/z$,
  $
    lim_(z->0) (f(z))/(g(z)) &= lim_(z->0) (f'(z))/(g'(z)): "l'hopitals rule" \
    lim_(p(x)->0) (ln p(x))/( 1 / p(x)) &= lim_(z->0) (f(z))/(g(z)) \
    => lim_(p(x)->0) (ln p(x))/( 1 / p(x)) &= lim_(z->0) (f'(z))/(g'(z)) \
    &= lim_(z->0) (1/z)/(-1/z^2) \
    &= lim_(z->0) -z \
    &= 0 \
    therefore lim_(p(x)->0^+) p(x)ln p(x) = 0
  $
]

== Frequentist Inference

- maximize the llh fn $cal(L)(theta; "Data")$ to find a point estimate
  $hat(theta)$ which is MLE, this bives model of best fit
  - sometimes we find this by sampling distribution exactly (z-stat or t-stat
    with samples from a gaussian pop)
  - sometimes we use asymptotic constraints
- inference is based in some sens upon data we do not observe
thee calssical approach assumes there are tru vales or papulation params that we
try to estimate with the data
- the model params are fixed and anknown
- we cannot make any probability statements about parameters
- the data are realizations from a randowm variable

= Bayesian inference

- under the bayesian approach we use prob to describe degrees of subjective
  belief, not limiting frequency
  - ie we can make prob. statement about parameters
  - and such params are not fixed, but vary according to some distribution

- that is under the bayesian approach, both
  - parameters are RVs
  - data are a realization from a RV
  - we use sample update our prob distrib used to describe subjective belief in
    the parameter

- bayesian inference leads to a distributional estimate of $theta$ which means
  that both
  - location of params and
  - uncertainty about params are contained in that distributional estimate, this
    is called posterior distribution ($theta | "Data"$)

- the sampling distribution plays no part in bayesian statistics
- the posterior distribution quantifies our uncerainty of $theta$ given the
  presence of the data

$
  f(theta|"data") = (f("data"|theta)h(theta))/(f("data")) \
$

$
  f(x|y) = (f(x,y))/(f(y)) = (f(x,y)h(x))/(f(y)) \
$

$
  f("data") = integral f("data"|theta)h(theta) dif theta \
$

- if you don't want to find the "integrating constant", $f("data")$, say:

$
  f(theta|"data") alpha f("data"|theta)h(theta)
$

- we can also apply bayes theorem in the inference situation, where our goal is
  to estimate the posteriror distr for the params _given_ the data characterized
  by the posterior density:

$
  pi(theta|D) = (l(D|theta)pi(theta))/(pi(D)) \
$

- $pi(D|theta)$ is likelihood function
- $pi(theta)$ is prior density/uncertainty of $theta$
- $pi(D) = integral pi(D|theta)pi(theta) dif theta$ is normalization constant

== Posterior Distribution

- the posterior summarizes all the information about the parameters
- for inference we might want a point estimate of $theta$, we could use
  posterior {maximum/median/mean/mode}
- uncertainty about the params is naturally containedd in the posterior distr
  (i.e. its variance)
- we can construct credible intervals from percentiles of the posterior distr
$
  P(theta in I | "data") = 1-alpha
$
- i can use the statement: the probability that $theta$ is in interval, $I$, ...

=== Example
- our experiment can be represented as $X_i ~ "Bin"(theta, n)$, then likelihood
  is then:
$
  g(x|theta) = (n, x) theta^x (1-theta)^(n-x)
$
- we could use a uniform prior: we choose arbitrarily, different priors lead to
  different answers
$
  p(theta) = 1 theta in (0, 1) \
  0 "otherwise"
$

- then we obtain the posterior (using bayes thm)
$
  pi(theta|x) = (f(x|theta)p(theta))/(p(x)) alpha f(x|theta)p(theta) \
$
because $p(x)$ is the marginal distribution of the data and does not depend on
the param, $theta$. Inputting the densities we obtain
$
  pi(theta|x) alpha (n, x) theta^x (1-theta)^(n-x) alpha theta^x (1-theta)^(n-x)
$

- since we can see that the last term (without normalization factor) is like a
  beta distribution, we can easily say that it is a beta distribtuion and infer
  the normalization factor when stating the posterior

== Priors

- priors are a sistributional representation of our belief about the params
  begore we have seen the data
  - we then update these priors with information from data via the likelihood
    function
  - the prior can represent expert knowledge about the params expressed in the
    form of suiable statistical distribution
  - or it can be the posterior from the analysis of different data
  - these are known as informative or subjective priors

- in some cases little or no prior knwledge may be available and we can use:
  - non-informative priors (eg jeffreys):
    $pi(theta) alpha sqrt(|cal(I)(theta)|)$ I is fisher info
  - vague priors (e.g. gaussian with large variance)
  - conjugate prior if the posterior and the prior belong to the same family of
    distributions. these types of priors are only available to distributions
    within the regular exponential family
- prior is improper if
  $integral_(-infinity)^(infinity) p(theta)dif theta = infinity$

== An Issue

- what happens when we get a difficult posterior
- mathematically even the simplest posterior can be very difficult to deal with,
  this is due (in the first instance) to the problem of integration to obtain
  the normalization constant
- we could consider the posterior only up to proportionality, however
  integrating to find the posterior mean will still be problematic
- even if we can find the maximum posterior estimate via differentiation, we
  still need to integrate to find confidence intervals

- *solution*: if we can sample from the posterior distribution, we could use the
  sampled sequence of observations $theta^(1), theta^(2), ... , theta^(T)$ to
  provide an approximation to the posterior, then we can estimate things like
  the posterior mean
- this is known as monte carlo integration
- summary we can us MC integration if we are able to sample from the posterior
- now we need a general method to sample from any distribution: Monte Carlo
  Method and Markov chains, which we will discuss individualy first

== Monte Carlo Method

- recall the problem of ealuating an integral which is too comples to calculate
  explicitly
- we can use the simulation technique of monte carlo integration to gain an
  estimate of a given integral
  - the method is based upon drawing obervations from the distribution of the
    variable of interest and simply calculating whatever stat we are interested
    in
  - methods such as rejection and importance sampling can be used to generate
    independent random samples from any ddistribution, however such methods can
    be ineffiicient in higer dimensions

- MCMC is an example of the MC method which we implement through use of markov
  chains
  - mcmc ddoes not produce independent samples from the target density, $pi(x)$,
    but dependent samples

- for example given obs:
$
  x_1, ..., x_n ~ pi(x) \
  "we estimate with: "
  EE_pi [f(X)] = integral f(x) pi(x) dif x
$
- by the average
$
  bar(f)_n = sum
$

- for independent samples the law of large numbs ensures

$
  bar(f)_n arrow EE_pi [f(X)] "as" n arrow infinity
$

== Markov Chains

- a markov chain ${ X^t }$ is a sequence of dependent random variables:

$
  X^0, X^2, ..., X^t, ...
$

- a markov chain is generated by sampling ht enew state of the chain, based only
  upon information regarding the current state, i.e., at time $t$
- the is what we generate the new state of the chain from a density dependent
  only only upon $x^t$:

$
  X^(t+1) ~ cal(K)(x^t, x) (= cal(K)(x|x^t))
$

- we call $cal(K)$ the *transition kernel* or *Markov kernel* for the chain and
  uniquely descirbes the dynamics of the chain
- In the case of Markov chains for discrete state-spaces, the markov Kernel is
  usually called the *transition matrix*
- in general, we will be interested in continuous state-spaces since our random
  variables of interest will be parameteers in a model, which are usually but
  not always, continuous

example:

- consider the markov chain with th transition kernel given by:
$
  X^(t+1) ~ (1/2 x^t, )
$

- simulating this with starting points $x^0 = 15$ and $x^0 = -15$:
- we see that the sample paths is different from the others but settle down to a
  similar pattern within a few iterations, this distribution was the same,
  regardless of the initial condition
- the distribution that is followed once it has settled down is called the
  stationary distribution
- once settled down that chain is said to have readched stationarity/equlibrium

- the markov chains that we consider for mcmc are constructed in such a way
  that:
  + they have a stationary distribution
    - that is there exists a p distribution $pi(x)$ such that
      $X^t ~ pi(x) => X^(t+1) ~ pi(x)$
  + kernel $cal(K)(x^t, x)$ has property: irreducibility
    - for whatever $X^0$ the sequence ${X^t}$ has a positive probability of
      reaching any region of the state-space
  + thay are positive recurrent
    - a chain run for infinite tiem will return to any states an infinite number
      of times
  + they are aperiodoic
    - a state $i$ has period $k$ if any return to state $i$ must occurs in
      multiples of $k$ time steps. $k=1$ for all states implies that the chain
      is aperiodic

- ergodic theorem: suppose that we have 1234 markov chain w stationary
  distribution $pi(x)$ then the ergodic theorem states that

$
  bar f_n = 1/n sum_(t=1)^n f(x^t) arrow EE_pi [f(X)] "as" n arrow infinity
$

- we call $bar f_n$ the ergodic average

- central limit theorem

$
  sqrt(n) [1/n sum_(t=1)^n f(x^t) - EE_pi [f(X)]] arrow^cal(D) N(0, phi^2)
$

where the asymptitoic variance is given by:

$
  phi^2 = "Var"_pi [f(X^1)] + 2 sum_(k=2)^infinity "Cov"_pi [f(X^1, f(X^k))]
$

- the covariance terms are due to the correlation among the MCMC sample path.
  - MCMC does not produce independent samples from $F$ but dependent samples

- discussion:
  - the usual approach to Markov chain theory is to start with some transition
    kernel, determine conditions uder which there exists an invariant or
    stationary distribution, and then identify the form of that limiting
    distribution
  - MCMC methods involve the solution of the inverse of thes problem whereby the
    stationary distributionis known but...

== Metropolis Hasting Algorithm

recall: goal
$
  pi (theta|x)=( f(x|theta) p (theta) )/pi(x)
$
- however we don't know the posterior $pi(x)$ -- its the integrating constant,
  but we do know the above, $pi(theta)$ is $alpha f(x|theta) p(theta)$

- so we need $E(theta|X)$, thus we generate

we generate a sample path of $y_t, y_1, ..., y_n$ using $y^(t+1) ~ g(y|X^t)$,
this is our markov chain and the sample are correlated
- they are sampled with $g(y|X^t)$ to represent? $pi(y)$
- the samples represent $hat(pi)(y)$

- this metropolic-hastings algorithm is one of the most important stat
  developments

- the algorithm gives a very genereal way to construct MCMC chains that have a
  stationary distribution of interest
  - the distribution of interest is know as the target distribution denoted by
    $pi(x)$
  - then we can treat the MCMC sample path as the sample path from the actual
    distribution

- the rule for moving accross a space givena any target denstity
- metropolic-hastings method can be written algorithmically as follows:

+ given the current position, $x^t$, generate a candidate value, $y$ via
  proposal density $q(y|x^t)$
+ calulcate acceptance probability
$
  alpha(y| x^t) = min (1, (pi(y) q(x^t|y))/(pi(x^t) q(q|x^t)))
$
3. with probability $alpha(y|x^t)$, set $x^(t+1)=y$, else set $x^(t+1)=x^t$
+ return to step 1 until a prescibed number of MCMC iterations have been run

notes:
- in theory, we have a great deal of flexibility in our choice of proposal
  density, $q$
  - e.g. if $pi(x)$ is cts on $-infinity < x < infinity$, then $q$ can be any
    continuous desinty over the real line
  - a lot of the owrk we will do is finding out which proposal types that are
    better than others
- noe also that we only need to know $pi$ up to proportionality, since constants
  cancel out
  - whats nice about is the target density, it does not need the integrating
  constant as the division cancels it out
- if $q$ is chosen poorly, then the number of rejections can be high, so that
  the efficiency of the procedure can be low, any will work, bad ones will just
  take very long
  - choosing $q$ well is therefore key to successfully implementing a
    metropolis-hastings MCMC algorithm

- we now consider a series of commonly used proposal densities used in the
  Metropolis-Hastings algorithm

=== Random Walk MH or Metropolic Algorithm

- has the proposal of the form $y = x^t + Z$
  - where $Z ~ f$ and $f$ is symmetric around 1,
- some choices for $f$
  - uniform distribution on the unit disk
  - a (multivariate) normalization
  - t-distribution

- note that since the proosal for a random walk is symmetric, we have:
  $q(y|x)=q(x|y)$
- thus
$
  alpha(y| x^t) = min (1, (pi(y) q(x^t|y))/(pi(x^t) q(q|x^t))) = min (1, pi(y)/pi(x^t))
$

- example suppose that we were interestid in sampling from the standard normal
  distribution so:
$
  pi(x) = 1/sqrt(2 pi) exp(-x^2/2) => pi(x) prop exp(exp(-x^2/2))
$

$
  q(y|x) prop exp(-(y-x)^2/(2 sigma^2))
$
- acceptance prob

$
  alpha(y|x)=min(1, exp(-1/2 (y^2-x^2)))
$

- sample paths with $sigma = 0.1, 10$ are chosen
- with the right $sigma$ it converges to the stationary distribution the most
  quickly

example: gamma distribution

- target distribution is gamma,
- proposal distribution is normal
- in multivariate distrs we can only look at the proposal plot to see if
  something is good instead of seeing how well the histogram matches the target
  distribution

- MH Algorithim is an MC whihc allows us to generate from any target density
  - however, we need to provide a symmetric proposal
  - some proposals are better than others

== MCMC Convergence

- recall: MH algo is a rule for moving around a space such that you can use any
  target density as your posterior, then the stationary distribution is the
  target distribution

- we discuss to assess if a MCM has converged to the stationary distribution
- suppose we want to generate data from the laplace distribution with

