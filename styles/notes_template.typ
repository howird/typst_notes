#import "@preview/ctheorems:1.1.3": thmrules
#import "@preview/numbly:0.1.0": numbly

#let fonts = (
  serif: "New Computer Modern",
  sans: "New Computer Modern Sans",
  mono: "DejaVu Sans Mono font",
)

#let note(
  title: "",
  subtitle: "",
  author: "",
  date: "",
  abstract: none,
  accent: green,
  paper: "us-letter",
  margin: (top: 1in, bottom: 0.85in, inside: 1.25in, outside: 0.85in),
  fontsize: 12pt,
  cols: 1,
  sectionnumbering: numbly(
    "",
    "{2}.",
    "{2}.{3}.",
    "{2}.{3}.{4}.",
    "{2}.{3}.{4}.{5:A}.",
  ),
  body,
) = {
  set document(title: title)

  show: thmrules

  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
    columns: cols,
    header: context {
      show smallcaps: set text(tracking: 0.14em)
      set text(12pt, font: fonts.sans)
      if (here().page()) > 1 {
        if calc.odd(here().page()) {
          align(right, text(fill: accent)[#smallcaps(all: true)[#title]])
        } else {
          align(left, text(fill: accent)[#smallcaps(all: true)[#author]])
        }
      }
    },
    footer-descent: 30% + 0pt,
    footer: context {
      set text(10pt, font: fonts.sans)
      if calc.odd(here().page()) {
        align(right, counter(page).display("1"))
      } else {
        align(left, counter(page).display("1"))
      }
    },
  )

  show smallcaps: set text(tracking: 0.14em)

  set text(
    lang: "en",
    font: fonts.serif,
    size: fontsize,
    spacing: 90%,
    alternates: false,
    discretionary-ligatures: false,
    historical-ligatures: false,
    number-type: "old-style",
    number-width: "proportional",
  )

  set par(
    spacing: 8pt,
    leading: 8pt,
  )

  set quote(block: true)
  show quote: set block(spacing: 18pt)
  show quote: set pad(x: 1.5em)
  show quote: set par(leading: 8pt)
  show quote: set text(style: "normal")

  show raw: set block(inset: (left: 2em, top: 1em, right: 1em, bottom: 1em))
  show raw: set text(fill: rgb("#116611"), size: 9pt, font: fonts.mono)

  set image(fit: "contain")
  show image: it => {
    align(center, it)
  }
  set figure(gap: 1em, supplement: none, placement: none)
  show figure.caption: set text(size: 9pt)
  show figure: set block(below: 1.5em)

  set footnote.entry(indent: 0em)
  show footnote.entry: set par(spacing: 0.5em, justify: false)
  show footnote.entry: set par(hanging-indent: 0.4em)
  show footnote.entry: set text(size: 9pt, weight: 200)

  show math.equation: set text(weight: 400)

  show heading: set text(hyphenate: false)
  set heading(numbering: sectionnumbering)

  let heading-number(it) = if it.numbering != none {
    text(font: fonts.sans, fill: accent, weight: 600, size: fontsize + 1pt)[
      #sym.section #counter(heading).display()
    ]
  } else {
    none
  }

  show heading.where(level: 1): it => align(left, block(
    above: 14pt,
    below: 12pt,
    width: 100%,
  )[
    #v(12pt)
    #par(justify: false, first-line-indent: 0em)[
      #if it.numbering != none [
        #heading-number(it)
        #h(0.4em)
      ]
      #set text(font: fonts.serif, weight: "regular", size: fontsize + 4pt)
      #it.body
    ]
    #v(6pt)
  ])

  show heading.where(level: 2): it => align(left, block(
    above: 12pt,
    below: 10pt,
    width: 80%,
  )[
    #par(justify: false, first-line-indent: 0em)[
      #if it.numbering != none [
        #text(
          font: fonts.sans,
          fill: accent,
          weight: 600,
          size: fontsize + 0.5pt,
        )[
          #counter(heading).display()
        ]
        #h(0.35em)
      ]
      #set text(font: fonts.serif, style: "italic", size: fontsize + 2pt)
      #it.body
    ]
  ])

  show heading.where(level: 3): it => align(left, block(
    above: 10pt,
    below: 8pt,
  )[
    #par(justify: false, first-line-indent: 0em)[
      #if it.numbering != none [
        #text(font: fonts.sans, fill: accent, weight: 600, size: fontsize)[
          #counter(heading).display()
        ]
        #h(0.3em)
      ]
      #set text(font: fonts.sans, size: fontsize, tracking: 0.14em)
      #smallcaps(all: true)[#it.body]
    ]
  ])

  show "LaTeX": smallcaps
  show regex("https?://\S+"): set text(style: "normal", rgb("#33d"))

  set outline(indent: 1em)
  show outline: set heading(numbering: none)
  show outline: set par(first-line-indent: 0em)

  show outline.entry.where(level: 1): it => {
    text(font: fonts.sans, fill: accent)[#it]
  }
  show outline.entry: it => {
    text(font: fonts.sans, fill: accent)[#it]
  }

  // Title block.
  v(1em)
  set par(justify: false)
  align(left, [
    #set text(font: fonts.serif, size: fontsize + 6pt)
    #title
    #if subtitle != none and subtitle != "" [
      : #emph[#subtitle]
    ]
  ])

  v(1em)
  if author != none and author != "" {
    align(left, text(size: fontsize, font: fonts.serif)[#author])
  }
  if date != none and date != "" {
    align(left, text(size: fontsize, font: fonts.serif)[#date])
  }

  if abstract != none {
    v(0.5em)
    align(
      left,
      text(size: fontsize - 1pt, tracking: 0.05em, font: fonts.sans)[ABSTRACT: ]
        + text(size: fontsize - 1pt, style: "italic")[#abstract],
    )
  }

  v(1em)
  line(start: (1%, 0%), end: (99%, 0%), stroke: 0.5pt + accent)

  counter(page).update(1)

  set par(justify: true, first-line-indent: 1.5em)

  body
}
