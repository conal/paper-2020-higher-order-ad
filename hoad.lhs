% -*- latex -*-

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12,twoside]{article}  % fleqn,
\usepackage[margin=0.9in]{geometry}  % 0.12in, 0.9in

%% \documentclass{article}
%% \usepackage{fullpage}

\input{macros}

\author{Conal Elliott}

\nc\tit{Higher-Order, Higher-Order Automatic Differentiation}

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LO]{\tit}
\fancyhead[RE]{%
Conal Elliott
}
\fancyhead[LE,RO]{\thepage}
% \rnc{\headrulewidth}{0pt}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include formatting.fmt

\usepackage[square]{natbib}
\bibliographystyle{plainnat}

\title{\tit \\ \emph{\large (early draft---comments invited)}}

\date{\today}

\setlength{\blanklineskip}{2ex} % blank lines in code environment

\nc\proofLabel[1]{\label{proof:#1}}
%if short
\nc\provedIn[1]{\textnormal{See proof \citep[Appendix A]{Elliott-2019-hoad-extended}}}
%else
\nc\proofRef[1]{Appendix \ref{proof:#1}}
\nc\provedIn[1]{\textnormal{Proved in \proofRef{#1}}}
%endif

\begin{document}

\maketitle

%% \begin{abstract}
%% ...
%% \end{abstract}

\sectionl{Introduction}

This note picks up where \cite{Elliott-2018-ad-icfp} left off, and I assume the reader to be familiar with that paper and have it close at hand.
I am circulating this follow-on in fairly rough form for early feedback, to then evolve in to a full research paper.
The main new contributions are two senses of ``\emph{higher-order} automatic differentiation'':
\begin{itemize}
\item derivatives of higher-order functions, and
\item higher-order derivatives of functions, and
\end{itemize}
The former has been addressed in a recent paper \citep{Vytiniotis-2019-differentiable-curry}, but in a way I find a dissatisfying extension of the work of \cite{Elliott-2018-ad-icfp} for a variety of reasons described in \secref{Related Work} and discussed at length with the authors of \cite{Vytiniotis-2019-differentiable-curry}.

Begin with the category of computably differentiable functions from \cite[Section 4.1]{Elliott-2018-ad-icfp}:
\begin{code}
newtype D a b = D (a -> b :* (a :-* b))
\end{code}
where |a :-* b| is the type of linear maps from |a| to |b|.
The function around which the automatic differentiation (AD) algorithm is organized simply zips together a function |f :: a -> b| and its derivative |der f :: a -> a :-* b|:
\begin{code}
adf :: (a -> b) -> D a b
adf f  = D (\ a -> (f a, der f a))
       = D (f &&& der f)
\end{code}
The whole specification of AD is then simply that |adf| is a homomorphism with respect to a standard compositional vocabulary of functions, namely that of cartesian categories, plus a collection of numeric primitives like (uncurried) addition and multiplication, |sin| and |cos|, etc.
An example of such an equation is |adf g . adf f == adf (g . f)|, in which the only unknown is the meaning of the LHS |(.)|, i.e., sequential composition in the category |D|.
Solving the collection of such homomorphism equations yields correct-by-construction AD.

AD is often described as coming in forward and backward ``modes''.
For many practical applications (including deep learning and other high-dimensional optimization problems), reverse mode is much more efficient than forward mode.
As typically presented, reverse mode is also much more complicated, but this difference appears to be due only to unfortunate choices in how to think about and implement AD.
Instead, a single, simple algorithm works for both reverse as well as forward (and other) modes, with the only difference being that reverse mode uses a different linear map representation resulting from a simple classic trick \citep{Elliott-2018-ad-icfp}.

This general AD algorithm is justified by three main theorems shown in \figref{derivative theorems}:
\begin{figure}
\begin{center}
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
$|der (g . f) a == der g (f a) . der f a|$.
\end{theorem}
\begin{theorem}[cross rule] \thmLabel{cross}
$|der (f *** g) (a,b) == der f a *** der g b|$.
\end{theorem}
\begin{theorem}[linear rule] \thmLabel{linear}
For all linear functions |f|, |der f a == f|.
\end{theorem}
\caption{Fundamental theorems}
\figlabel{derivative theorems}
\end{center}
\end{figure}





 and a collection of facts about the derivatives of various mathematical operations.

\sectionl{Related Work}

\bibliography{bib}

\end{document}

