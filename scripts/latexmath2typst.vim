:%s/\\_/_/gI " get rid of '\_' often used within \text{foo\_bar}
:%s/\\text{\(.\{-}\)}/"\1"/gI
:%s/\\bar{\\\?\(.\{-}\)}/overline(\1)/gI
:%s/\\dot{\\\?\(.\{-}\)}/dot(\1)/gI
:%s/\\tilde{\\\?\(.\{-}\)}/tilde(\1)/gI
:%s/\\overline{\\\?\(.\{-}\)}/overline(\1)/gI
:%s/\\lfloor\(.\{-}\)\\rfloor/floor(\1)/gI
:%s/\\lceil\(.\{-}\)\\rceil/ceil(\1)/gI
:%s/\\lfloor\(.\{-}\)\\rceil/round(\1)/gI
:%s/||\(.\{-}\)||/norm(\1)/gI
:%s/|\(.\.\{-}\)|/abs(\1)/gI
:%s/\\hat{\\\?\(.\{-}\)}/hat(\1)/gI
:%s/\\mathcal{\(\w\)}/cal(\1)/gI " mathcal
:%s/\\mathbb{\(\w\)}/\1\1/gI " mathbb
:%s/\\rightarrow/->/gI
:%s/\\infty/infinity/gI
:%s/\\leftarrow/<-/gI
:%s/\\langle/angle.l/gI
:%s/\\rangle/angle.r/gI
:%s/\\sim/\~/gI
:%s/\\cdot/dot/gI
:%s/{\\\?\(.\{-}\)}/\1/gI " {single_word} -> single_word
:%s/{\(\_.\{-}\)}/(\1)/g " {many words} -> (many words)
:%s/("\(\_.\{-}\)")/"\1"/g " ("many words") -> "many words"
:%s/\([a-zA-Z]\)\\/\1 /gI " replace backslashes
:%s/\\//gI
