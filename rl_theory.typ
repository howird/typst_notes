#import "styles/notes_template.typ": *

#show: note.with(
  title: "RL Theory Notes",
  subtitle: "",
  author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

#include "rl/maxent-rl.typ"
#include "rl/auxiliary-self-prediction.typ"
#include "rl/diayn.typ"
