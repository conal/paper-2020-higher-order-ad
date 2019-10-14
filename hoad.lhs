% -*- latex -*-

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12,twoside]{article}  % fleqn,
\usepackage[margin=0.9in]{geometry}  % 0.12in, 0.9in

%% \documentclass{article}
%% \usepackage{fullpage}

\author{Conal Elliott}

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LO]{Higher-order automatic differentiation}
\fancyhead[RE]{%
Conal Elliott
}
\fancyhead[LE,RO]{\thepage}
% \rnc{\headrulewidth}{0pt}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include formatting.fmt

\input{macros}

\usepackage[square]{natbib}
\bibliographystyle{plainnat}

\title{Higher-Order Automatic Differentiation \\ \emph{\large (early draft---comments invited)}}

\date{\today}

%% \addtolength{\topmargin}{0.25in}
%% \addtolength{\headsep}{0.3in}

%% \addtolength{\textheight}{0.5in}
%% \addtolength{\footskip}{0.1in}

\setlength{\blanklineskip}{2ex} % blank lines in code environment

\nc\proofLabel[1]{\label{proof:#1}}
%if short
\nc\provedIn[1]{\textnormal{See proof \citep[Appendix C]{Elliott-2018-ccc-extended}}}
%else
\nc\proofRef[1]{Appendix \ref{proof:#1}}
\nc\provedIn[1]{\textnormal{Proved in \proofRef{#1}}}
%endif

\begin{document}

\maketitle

\begin{abstract}

...

\end{abstract}

%% \bibliography{bib}

\end{document}

