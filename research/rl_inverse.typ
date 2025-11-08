#import "styles/notes_template.typ": *

#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm-figure, style-algorithm

#show: note.with(
  title: "Inverse RL Notes",
  subtitle: [Spring 2025],
  author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

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

