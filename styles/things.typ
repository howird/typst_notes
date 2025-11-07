#import "@preview/ctheorems:1.1.3": thmbox, thmplain, thmproof
#import "@preview/showybox:2.0.4": showybox
#import "@preview/numbly:0.1.0": numbly

#let theorem = thmbox("theorem", "Theorem", fill: rgb("#eeffee"))
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong,
)
#let example = thmplain("example", "Example").with(numbering: none)
#let proof = thmproof("proof", "Proof")

#let thmDefiner = thmbox.with(
  padding: (top: 0em, bottom: 0em),
  breakable: true,
  inset: (top: 0em, left: 0em, right: 0em),
  namefmt: name => emph([(#name)]),
  titlefmt: emph,
)

#let definitionDfn = thmDefiner("definition", "Definition")
#let questionDfn = thmDefiner("question", "Question").with(numbering: numbly(
  "",
  "{2}",
  "{2}.{3}",
  "{2}.{3}.{4}",
  "{2}.{3}.{4}.{5:A}",
))
#let challengeDfn = thmDefiner("challenge", "Challenge").with(numbering: numbly(
  "",
  "{2}",
  "{2}.{3}",
  "{2}.{3}.{4}",
  "{2}.{3}.{4}.{5:A}",
))
#let hypothesisDfn = thmDefiner(
  "hypothesis",
  "Hypothesis",
  base: "challenge",
).with(numbering: none)

#let roundSimple(
  title,
  color: teal,
  thmDfn: definitionDfn,
  ..args,
) = showybox(
  body-style: (
    color: color.darken(40%),
    align: left,
  ),
  title-style: (
    color: color.darken(40%),
    body-color: color.lighten(80%),
    sep-thickness: 0pt,
    align: left,
  ),
  frame: (
    border-color: color.darken(40%),
    body-color: color.lighten(80%),
    title-color: color.lighten(80%),
  ),
  breakable: true,
  thmDfn(title),
  ..args,
)

#let squareBoldTitle(
  title,
  color: olive,
  thmDfn: challengeDfn,
  ..args,
) = showybox(
  title-style: (
    weight: 900,
    color: color.darken(40%),
    sep-thickness: 0pt,
    align: center,
  ),
  frame: (
    title-color: color.lighten(80%),
    border-color: color.darken(40%),
    thickness: (left: 1pt, right: 1pt),
    radius: 0pt,
  ),
  sep: (
    dash: "dotted",
  ),
  breakable: true,
  title: thmDfn(title),
  ..args,
)

#let challenge = squareBoldTitle
#let definition = roundSimple
#let hypothesis = roundSimple.with(color: olive, thmDfn: hypothesisDfn)
#let question = squareBoldTitle.with(color: orange, thmDfn: questionDfn)

