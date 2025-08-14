#import "notes_template.typ": *
#import "@preview/algorithmic:1.0.0"
#import algorithmic: style-algorithm, algorithm-figure

#show: dvdtyp.with(
  title: "RL Architectures Notes", subtitle: [Spring 2025], author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

#include "rl/archs/diffusion-policy.typ"
#include "rl/archs/streaming-diffusion-rl.typ"
#include "rl/archs/normalizing-flow-policy.typ"
