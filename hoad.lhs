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

% \usepackage[square]{natbib}
\usepackage[round]{natbib}
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
Note that this definition is not computable, since |der| is not \citep{PourEl1978Diff, PourEl1983Comp}.
The whole specification of AD is then simply that |adf| is a homomorphism with respect to a standard compositional vocabulary of functions, namely that of cartesian categories, plus a collection of numeric primitives like (uncurried) addition and multiplication, |sin| and |cos|, etc.
An example of such an equation is |adf g . adf f == adf (g . f)|, in which the only unknown is the meaning of the LHS |(.)|, i.e., sequential composition in the category |D|.
Solving the collection of such homomorphism equations yields correct-by-construction AD.

AD is often described as coming in forward and backward ``modes''.
For many practical applications (including deep learning and other high-dimensional optimization problems), reverse mode is much more efficient than forward mode.
As typically presented, reverse mode is also much more complicated, but this difference appears to be due only to unfortunate choices in how to think about and implement AD.
Instead, a single, simple algorithm works for both reverse as well as forward (and other) modes, with the only difference being that reverse mode uses a different linear map representation resulting from a simple classic trick \citep{Elliott-2018-ad-icfp}.

This general AD algorithm is justified by three main theorems:
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
$|der (g . f) a == der g (f a) . der f a|$.
\end{theorem}
\begin{theorem}[cross rule] \thmLabel{cross}
$|der (f *** g) (a,b) == der f a *** der g b|$.
\end{theorem}
\begin{theorem}[linear rule] \thmLabel{linear}
For all linear functions |f|, |der f a == f|.
\end{theorem}
In addition to these three theorems, we need a collection of facts about the derivatives of various mathematical operations, e.g., |adf sin x = scale (cos x)|, where |scale :: a -> a :-* a| is uncurried scalar multiplication (so |scale s| is linear for all |s|).

\sectionl{Cartesian closed?}

%format apply = eval

While |D| is a category and a \emph{cartesian} category at that, as specified by |adf| being a cartesian functor, another question naturally arises.
Can |adf| also a \emph{closed} cartesian functor?
In other words, are there definitions of |apply|, |curry|, and |uncurry| on |D| such that
\begin{code}
apply = adf apply
curry (adf f) = adf (curry f)
uncurry (adf g) = adf (uncurry f)
\end{code}
These three operations come from the following interface:
\begin{code}
class Cartesian k => CartesianClosed k where
  type ExpOp k :: Type -> Type -> Type
  apply :: Exp k ((a :=> b) :* a) b
  curry :: ((a :* b) `k` c) -> (a `k` (Exp k b c))
  uncurry :: (a `k` (Exp k b c)) -> ((a :* b) `k` c)
\end{code}
where |Exp k a b| is a type of ``first class functions/arrows'' from |a| to |b| for the category |k|, also known as ``exponential objects''.
These operations support higher-order programming and arise during translation from a typed lambda calculus (e.g., Haskell) to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}.

Similarly, monoidal and cartesian categories have category-associated categorical \emph{products}:
\begin{code}
class Category k => MonoidalPCat k where
  (***) :: (a `k` c) -> (b `k` d) -> ((Prod k a b) `k` (Prod k c d))

class Monoidal k => Cartesian k where
  type ProdOp k :: Type -> Type -> Type
  exl  :: (Prod k a b) `k` a
  exr  :: (Prod k a b) `k` b
  dup  :: a `k` (Prod k a a)
\end{code}
A particularly important related operation:
\begin{code}
(&&&) :: Cartesian k => (a `k` c) -> (a `k` d) -> (a `k` (Prod k c d))
f &&& g = (f *** g) . dup
\end{code}

The choice of category-associated products and exponentials is a degree of freedom not exercised in the development of AD in \cite{Elliott-2018-ad-icfp} and one that is tied closely to another such choice available in the general notion of \emph{functor} in category theory.
In general, a functor has two aspects:
\begin{itemize}
\item a mapping from arrows to arrows, and
\item a mapping from objects to objects.
\end{itemize}
The functor |adf| defined (noncomputably) above implicitly chooses an \emph{identity object mapping}, as evident in its type |(a -> b) -> D a b|.

Just as the |Category| and |Cartesian| instances for |D| arose from solving corresponding homomorphism equations about |adf|, let's now try the same with |CartesianClosed|.
First note that we do not really have to define all three methods, since |apply| and |uncurry| can each be defined in terms of the other:\footnote{The pattern |g *** id| is also called ``|first g|'', because it applies |g| to the first element of a pair while leaving the second element unchanged.}
\begin{code}
apply = uncurry id
uncurry g = apply . (g *** id)
\end{code}
Since |apply| looks simpler, start there.
The corresponding homomorphism equation has a particularly simple form:
\begin{code}
apply = adf apply
\end{code}
It might appear that we are done already at the start, taking the equation to be a definition for |apply|.
Recall, however, that |adf| is noncomputable, being defined via |der| (differentiation itself).
Let us press forward undeterred, opening up the definition of |adf| to see if we can transform away the (noncomputable) |der|:
\begin{code}
    adf apply
==  D (apply &&& der apply)                       -- definition of |adf|
==  D (\ (f,a) -> (apply (f,a), der apply (f,a))  -- |(&&&) on functions|
==  D (\ (f,a) -> (f a, der apply (f,a))          -- |apply| on functions
\end{code}
Now we do not need the general |der|, but rather the specific |der apply|.
If |apply| were linear, we could apply \thmRef{linear}, but alas it is not.
No matter, as we can instead use the technique of partial derivatives, which is useful for functions of nonscalar domains.
Suppose we have a function |f :: a :* b -> c|, and we want to compute its derivative at a point in its (pair-valued) domain.
Then\footnote{There is a temporary abuse of notation here in that lambda expressions do not necessarily denote \emph{linear} functions, although the they do in this calculation, since derivative values are linear maps.}
%% %format da = "\Varid{\mathrm{\Delta} a}"
%% %format db = "\Varid{\mathrm{\Delta} b}"
%% %format derl = sub der l
%% %format derr = sub der r
%format derl = der"_l"
%format derr = der"_r"
\begin{code}
    der f (a,b)
==  \ (da,db) -> der f (a,b) (da,db)
==  \ (da,db) -> der f (a,b) ((da,0) + (0,db))
==  \ (da,db) -> der f (a,b) (da,0) + der f (a,b) (0,db)
==  \ (da,db) -> der f (a,b) (inl da) + der f (a,b) (inr db)
==  \ (da,db) -> (der f (a,b) . inl) da + (der f (a,b) . inr) db
==  \ (da,db) -> (der f (a,b) . inl ||| der f (a,b) . inr) (da,db)
==  der f (a,b) . inl ||| der f (a,b) . inr
==  derl f (a,b) ||| derr f (a,b)
\end{code}
where by convenient definition, |derl| and |derr| denote ``partial derivatives'' in which one half of a pair-valued argument is allowed to vary while the other half is held constant, i.e.,
\begin{code}
derl :: (a :* b -> c) -> a :* b -> (a :-* c)
derl f p = der f p . inl

derr :: (a :* b -> c) -> a :* b -> (b :-* c)
derr f p = der f p . inr
\end{code}

\mynote{Define |inl| and |inr|.}

\sectionl{Related Work}

\bibliography{bib}

\end{document}

