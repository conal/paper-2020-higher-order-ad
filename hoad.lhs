% -*- latex -*-

%% While editing/previewing, use 12pt or 14pt and tiny margin.
\documentclass[12pt,twoside]{article}  % fleqn,14pt
\usepackage[margin=0.12in]{geometry}  % 0.12in, 0.9in

%% \documentclass{article}
%% \usepackage{fullpage}

\input{macros}

\author{Conal Elliott\\[0.5ex]conal@@conal.net}

\usepackage{datetime}
\usdate

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

\usepackage[round]{natbib}  % square
\bibliographystyle{plainnat}

\title{\tit \emph{\\[1.5ex] \Large (early, incomplete draft---comments invited\footnote{For the latest version of this document and the repository location for questions, suggestions, bugs, etc, see \url{http://conal.net/papers/higher-order-ad}.})}}

\date{Draft of \today{} \currenttime}

%% \date{\today}

\setlength{\blanklineskip}{2ex} % blank lines in code environment

% \nc\proofRef[1]{Appendix \ref{proof:#1}}

\nc\proofRef[1]{\autoref{proof:#1}}
\nc\provedIn[1]{\textnormal{proved in \proofRef{#1}}}
\nc\proofLabel[1]{\label{proof:#1}}

%% \renewcommand{\theenumi}{\roman{enumi}}
\renewcommand{\theenumi}{\alph{enumi}}

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
The former has been addressed in a recent paper \citep{Vytiniotis-2019-differentiable-curry}, but in a way I find dissatisfying for a variety of reasons described in \secref{Related Work} and discussed at length with the authors.

Begin with the category of computably differentiable functions from \citet[Section 4.1]{Elliott-2018-ad-icfp}:
\begin{code}
newtype D a b = D (a -> b :* (a :-* b))
\end{code}
where |a :-* b| is the type of linear maps from |a| to |b|.
The function around which the automatic differentiation (AD) algorithm is organized simply ``zips'' together a function |f :: a -> b| and its derivative |der f :: a -> a :-* b|:\notefoot{Red bracketed remarks are notes to myself.}\footnote{This paper generally uses Haskell notation but deviates slightly by using a single colon rather than double colon for type signatures. \note{Experimental.}}\footnote{The infix operators for function types (``|->|'') and linear maps (``|:-*|'') both associate to the right and have equal, very low precedence.
For instance, ``|a -> a :-* b|'' means |a -> (a :-* b)|.}
\begin{code}
adh :: (a -> b) -> D a b
adh f  = D (\ a -> (f a, der f a))
       == D (f &&& der f)
\end{code}
Note that this definition is not computable, since |der| is not \citep{PourEl1978Diff, PourEl1983Comp}.
The whole specification of AD is then simply that |adh| is a homomorphism with respect to a standard compositional vocabulary of functions, namely that of cartesian categories, plus a collection of numeric primitives like (uncurried) addition and multiplication, |sin| and |cos|, etc.
An example of such a homomorphism equation is |adh g . adh f == adh (g . f)|, in which the only unknown is the meaning of the LHS |(.)|, i.e., sequential composition in the category |D|.
Solving the collection of such homomorphism equations yields correct-by-construction AD.

%format gh = "\Varid{\hat{g}}"
%format fh = "\Varid{\hat{f}}"
The function |adh| is invertible, i.e., |unadh . adh == id|, where |unadh| simply drops the derivative:\footnote{This paper uses ``|exl|'' and ``|exr|'' to name left and right product projections (defined on cartesian categories), rather than Haskell's (function-only) ``|fst|'' and ``|snd|''.}
\begin{code}
unadh :: D a b -> (a -> b)
unadh (D h) = exl . h
\end{code}
Indeed, |unadh| is a left inverse of |adh|:
\begin{code}
    unadh (adh f)
==  unadh (D (f &&& der f))  -- |adh| definition
==  exl . (f &&& der f)      -- |unadh| definition
==  f                        -- cartesian law: |exl . (g &&& h) == g|
\end{code}

As defined so far, |unadh| is \emph{not} a right inverse to |adh|, since the linear map portion might not be the true derivative.
We will thus \emph{restrict} the category |D| to be the image of |adh|, which is to say that |adh| is surjective, i.e., the derivative is correct.\footnote{
Haskell's type system is not expressive enough to capture this restriction by itself, so the restriction will be only implied in this draft.
For more rigor, one could use a language (extension) with refinement types such as Liquid Haskell \needcite{} or a dependently-typed language such as Agda \needcite{} or Idris \needcite{}.}
This restriction guarantees that |unadh| is indeed a right inverse of |adh|.
Given |fh :: D a b| (with the mentioned restriction), there is an |f :: a -> b| such that |fh = adh f|, so\footnote{This reasoning hold for \emph{any} surjective function with a left inverse.}
\begin{code}
    adh (unadh fh)
==  adh (unadh (adh f))  -- |fh = adh f|
==  adh f                -- |unadh . adh == id|
==  fh                   -- |adh f = fh|
\end{code}
Thus, |adh . unadh == id| as well.

AD is often described as coming in forward and reverse ``modes''.
For many practical applications (including deep learning and other high-dimensional optimization problems), reverse mode is much more efficient than forward mode.
As typically presented, reverse mode is also much more complicated, but this difference appears to be due only to unfortunate choices in how to understand and implement AD.
Instead, a single, simple algorithm works for forward, reverse, and other modes.
Reverse mode is distinguished only by using a different linear map representation resulting from a simple classic trick \citep{Elliott-2018-ad-icfp}.

This general AD algorithm is justified by three main theorems about differentiation:
\begin{quotation}
\vspace{-6ex}
\begin{theorem}[compose/``chain'' rule] \thmLabel{deriv-compose}
$$|der (g . f) a == der g (f a) . der f a|$$
\end{theorem}
\begin{theorem}[cross rule] \thmLabel{deriv-cross}
$$|der (f *** g) (a,b) == der f a *** der g b|$$
\end{theorem}
\begin{theorem}[linear rule] \thmLabel{deriv-linear}
For all linear functions |f|, $$|der f a == f|$$
\end{theorem}
\end{quotation}
\noindent
In addition to these three theorems, we need a collection of facts about the derivatives of various mathematical operations, e.g., |adh sin x = scale (cos x)|, where |scale :: a -> a :-* a| is uncurried scalar multiplication (so |scale s| is linear for all |s|).


\sectionl{Some Additional Properties of Differentiation}

A few additional properties of differentiation will prove useful in extending \cite{Elliott-2018-ad-icfp} to higher-order functions and higher-order derivatives.

%if False
\subsectionl{Linearity and invertibility}

As is well-known,
\begin{theorem}
Differentiation itself (i.e., |der|) is linear.
\end{theorem}

\note{|adh| as well. Move inversion and |unadh| here also. Also |fork| and |unfork|}
%endif

\subsectionl{Pair-Valued Domains}

One half of the |curry|/|uncurry| isomorphism involves functions of pair-valued domains.
The notion of partial derivatives is helpful for differentiating such functions.\out{\notefoot{I'm leaning toward eliminating |derl| and |derr| in favor of their meanings.
Whenever I use the names below, I then immediate inline them.}}\footnote{Recall that, on linear maps, |(f !!! g) (a,b) = f a + g b|, |inl a = (a,0)|, and |inr b = (0,b)|}
\begin{lemma}[\provedIn{deriv-pair-domain}]\lemLabel{deriv-pair-domain}
Given a function |f :: a :* b -> c|, $$
|der f (a,b) == derl f (a,b) !!! derr f (a,b)|
$$ where |derl| and |derr| construct the (``first'' and ``second'', or ``left'' and ``right'') ``partial derivatives'' of |f| at |(a,b)|, defined as follows:
\begin{code}
derl :: (a :* b -> c) -> a :* b -> (a :-* c)
derl f (a,b) = der (f . (,b)) a

derr :: (a :* b -> c) -> a :* b -> (b :-* c)
derr f (a,b) = der (f . (a,)) b
\end{code}
The notation ``|(a,)|'' and ``|(b,)|'' refers to right and left ``sections'' of pairing: |(,b) a == (a,) b == (a,b)|.
Equivalently,
\begin{code}
derl  f (a,b) = der f (a,b) . inl
derr  f (a,b) = der f (a,b) . inr
\end{code}
\end{lemma}
Note also that |f . (a,) = curry f a| and |f . (,b) = curry' f b|, where
\begin{code}
curry   f a b = f (a,b)
curry'  f b a = f (a,b)
\end{code}

As an example of how this decomposition of |der f| helps construct derivatives, suppose that |f| is \emph{bilinear}, which is to say that |f| is linear in each argument, while holding the other constant.
More formally |bilinearity| of |f| means that |f . (a,)| and |f . (b,)| (equivalently, |curry f a| and |curry' f b|) are both linear for all |a| and |b|.
\begin{corollary}\corLabel{deriv-bilinear}
If |f :: a :* b -> c| is bilinear then $$
|der f (a,b) == f . (,b) !!! f . (a,)|
$$
\end{corollary}
\begin{proof}~
\begin{code}
    der f (a,b)
==  derl f (a,b) !!! derr f (a,b)          -- \lemRef{deriv-pair-domain}
==  der (f . (,b)) a !!! der (f . (a,)) b  -- |derl| and |derr| definitions
==  f . (,b) !!! f . (a,)                  -- linearity
\end{code}
\end{proof}
For instance, the derivative of uncurried multiplication is given by the Leibniz product rule:
\begin{code}
    der (uncurry (*)) (a,b)
==  uncurry (*) . (,b) !!! uncurry (*) . (a,)
==  (NOP * b) !!! (a * NOP)
== \ (da,db) -> da * b + a * db
\end{code}
% which is sometimes written ``|d (u v) = u dv + v du|''.

More generally, consider differentiating interacts with uncurrying:
\begin{corollary}[\provedIn{deriv-uncurry}]\corLabel{deriv-uncurry}
$$| der (uncurry g) (a,b) == at b . der g a !!! der (g a) b |$$
\end{corollary}

As a special case, let |g| be curried multiplication:
\begin{code}
    der (uncurry (*))
==  at b . der (*) a !!! der (a *) b
==  at b . (*) !!! (a *)
==  (NOP * b) !!! (a * NOP)
\end{code}
which agrees with the calculation above.

For cartesian closure, we'll need the derivative of another function with a pair-valued domain:
\begin{code}
eval :: (a -> b) :* a -> b
eval (f,a) = f a  -- on functions
\end{code}
(Since |eval| is neither linear nor bilinear, \thmRef{deriv-linear} and \corRef{deriv-bilinear} are inapplicable.)
We'll need one more linear map operation, which is curried, reverse function application:\footnote{Linearity of |at a| follows from the usual definition of addition and scaling on functions.}
\begin{code}
at :: a -> (a -> b) :-* b
at a df = df a
\end{code}
\begin{corollary}[\provedIn{deriv-eval}] \corLabel{deriv-eval}
$$ |der eval (f,a) == at a !!! der f a| $$
\end{corollary}

\subsectionl{Function-Valued Codomains}

It will also be useful to calculate derivatives of functions with higher-order codomains.\notefoot{The previous section and this one provide ``adjoint'' techniques in a sense that currying is an adjunction from functions from products to functions to functions.
Is there something else interesting to say here?}
We'll need another linear map operation, which is the indexed variant of |(&&&)| (and a specialization of Haskell's |flip| function):
\begin{code}
forkF :: (b -> a :-* c) -> (a :-* b -> c)
forkF h = \ da b -> h b da
\end{code}
\begin{lemma}[\provedIn{deriv-function-codomain}]\lemLabel{deriv-function-codomain}
Given a function |g :: a -> b -> c|,
$$|der g a = forkF (\ b -> der (at b . g) a)|.$$
\end{lemma}

%% Curried functions differentiate as follows:
\begin{corollary}[\provedIn{deriv-curry}]\corLabel{deriv-curry}
$$ |der (curry f) a == forkF (derl f . (a,))| $$
\end{corollary}


\sectionl{Cartesian Closure, first attempt}

While |D| is a category and a \emph{cartesian} category at that, as specified by |adh| being a cartesian functor, another question naturally arises.
Can |adh| also be a cartesian \emph{closed} functor?
In other words, are there definitions of |eval|, |curry|, and |uncurry| on |D| such that
\begin{code}
curry (adh f) == adh (curry f)
uncurry (adh g) == adh (uncurry g)
eval == adh eval
\end{code}
As usual, we'll want to solve each homomorphism equation for its single unknown, which is a categorical operation on |D| (on the LHS).

\subsectionl{Curry}

Start with |curry|, simplifying the LHS:
\begin{code}
    curry (adh f)
==  curry (D (f &&& der f)) -- |adh| definition
\end{code}
Then the RHS:
\begin{code}
    adh (curry f)
==  D (curry f &&& der (curry f))                                    -- |adh| definition
==  D (\ a -> (curry f a, der (curry f) a))                          -- |(&&&)| on functions
==  D (\ a -> ((\ b -> f (a,b)), forkF (derl f . (a,))))             -- \corRef{deriv-curry}
==  D (\ a -> ((\ b -> f (a,b)), forkF (\ b -> derl f (a,b))))       -- |(.)| on functions
==  D (\ a -> ((\ b -> f (a,b)), forkF (\ b -> der f (a,b) . inl)))  -- \proofRef{deriv-pair-domain}
\end{code}
The last form uses |f| and |der f|, which can be extracted from |adh f = D (f &&& der f)|:
Thus a sufficient condition for our homomorphic specification (|curry (adh f) == adh (curry f)|) is
\begin{code}
curry (D ff') = D (\ a -> ((\ b -> f (a,b)), forkF (\ b -> f' (a,b) . inl)))
  where (f,f') = unfork ff'
\end{code}
The |unfork| function is half of an isomorphism that holds for all cartesian categories:
\begin{code}
fork :: Cartesian k => (a `k` c) :* (a `k` d) -> (a `k` (c :* d))
fork = uncurry (&&&)

unfork :: Cartesian k => (a `k` (c :* d)) -> (a `k` c) :* (a `k` d)
unfork h = (exl . h, exr . h)
\end{code}
\begin{lemma}\lemLabel{fork-iso-linear}
The pair of functions |fork| and |unfork| form an isomorphism in all cartesian categories and a linear isomorphism in the category of vector spaces and linear maps.\notefoot{To do: name this category early (say ``$\textbf{Vec}_{\!s}$'' for a semiring $s$) and refer to by name where needed.}
\proofEx
\end{lemma}

Another such linear isomorphism can be found in cocartesian categories.
The following types are specialized to biproduct categories (such as linear maps):
\begin{code}
join :: Cocartesian k => (a `k` c) :* (a `k` d) -> (a `k` (c :* d))
join = uncurry (!!!)

unjoin :: Cocartesian k => (a `k` (c :* d)) -> (a `k` c) :* (a `k` d)
unjoin h = (h . inl, h . inr)
\end{code}
\begin{lemma}\lemLabel{join-iso-linear}
The pair of functions |join| and |unjoin| form an isomorphism in all cocartesian categories and a linear isomorphism in the category of vector spaces and linear maps.
\proofEx
\end{lemma}
These two isomorphism pairs were used by \cite{Elliott-2018-ad-icfp} to construct a correct-by-construction implementation of reverse-mode AD, by merely altering the representation of linear maps used in the simple, general AD algorithm.

Another useful operation is the \emph{uncurried} version of the monoidal |(***)|:
\begin{code}
cross :: Monoidal k => (a `k` c) :* (b `k` d) -> ((a :* b) `k` (c :* d))
cross = uncurry (***)
\end{code}
\begin{lemma}\lemLabel{cross-linear}
In the category of vector spaces and linear maps, the |cross| function is linear.
\proofEx
\end{lemma}

Although |fork| and |unfork| form an isomorphism and hence preserve information, |unfork| can result in a loss of efficiency, due to computation that can be (and often is) in common to a function |f| and its derivative |der f|.
Indeed, the definition of |unfork h| above shows that |h| gets replicated.
It's unclear how to avoid this redundancy problem in practice with currying when |D| is used to represent computably differentiable functions.
Personal experience with compiling to categories \cite{Elliott-2017-compiling-to-categories} suggests that most uses of |curry| generated during translation from the $\lambda$ calculus (e.g., Haskell) are in fact transformed away at compile time using various equational CCC laws.
Still, it does seem an important question to explore.

Intriguingly, curried functions can also help eliminate redundant computation suggested by uncurried counterparts functions.
Given a function |g :: a -> b -> c|, it is sometimes convenient to ``partially apply'' |g| to an argument |u| and then apply the resulting |g u :: b -> c| to many different |v :: b|.
In some cases, a considerable amount of work can be done based solely on |u|, saving residual work to be done for different |b| values.
In such situations, |uncurry g| loses this performance advantage.

\subsectionl{Uncurry}

Next, let's tackle |uncurry|, whose defining homomorphism is
\begin{code}
uncurry (adh g) == adh (uncurry g)
\end{code}
Simplify the LHS:
\begin{code}
    uncurry (adh g)
==  uncurry (D (g &&& der g))  -- |adh| definition
\end{code}
Then the RHS:
\begin{code}
    adh (uncurry g)
==  D (uncurry g &&& der (uncurry g))                        -- |adh| definition
==  D (\ (a,b) -> (uncurry g (a,b), der (uncurry g) (a,b)))  -- |(&&&)| definition
==  D (\ (a,b) -> (g a b, der (uncurry g) (a,b)))            -- |uncurry| on functions
==  D (\ (a,b) -> (g a b, at b . der g a !!! der (g a) b))   -- \corRef{deriv-uncurry}
\end{code}
Now we have a problem with solving the defining homomorphism above.
Although we can extract |g| and |der g| from |adh g|, we cannot extract |der (g a)|.
Or rather we can, but not computably.

\subsectionl{Eval}

We don't need to work out both |uncurry| and |eval|, since each can be defined in terms of the other:\out{\footnote{The pattern |g *** id| is also called ``|first g|'', because it applies |g| to the first element of a pair while leaving the second element unchanged.}}
\begin{code}
eval = uncurry id
uncurry g  = eval . (g *** id)
           = eval . first g
\end{code}
Since we got stuck on |uncurry|, however, let's try |eval| as well to see if we learn anything new.

The corresponding homomorphism equation has a particularly simple form:
\begin{code}
eval = adh eval
\end{code}
It might appear that we have succeeded at the start, taking the equation to be a definition for |eval|.
Recall, however, that |adh| is noncomputable, being defined via |der| (differentiation itself).
Simplifying the RHS,
\begin{code}
    adh eval
==  D (eval &&& der eval)                        -- |adh| definition
==  D (\ (f,a) -> (eval (f,a), der eval (f,a)))  -- |(&&&)| on functions
==  D (\ (f,a) -> (f a, der eval (f,a)))         -- |eval| on functions
==  D (\ (f,a) -> (f a, at a !!! der f a))       -- \corRef{deriv-eval}
\end{code}
As with uncurrying (\secref{Uncurry}), the final form is well-defined but is not a computable recipe, leaving us in a pickle.
Next, let's look for some wiggle room.


\sectionl{Object Mapping}

The choice of category-associated products and exponentials is a degree of freedom not exercised in the development of AD in \cite{Elliott-2018-ad-icfp} (or above) and is tied closely to another such choice available in the general notion of \emph{cartesian closed functor} in category theory.
In general, a functor has two aspects:
\begin{itemize}
\item a mapping from arrows to arrows, and
\item a mapping from objects to objects.
\end{itemize}
The functor |adh| defined (noncomputably) above implicitly chooses an \emph{identity object mapping}, as evident in its type signature |adh :: (a -> b) -> D a b|.
The type of |adh| plus the requirement that it be a cartesian \emph{closed} functor implies that the object mapping aspect of |adh| is the identity.
More generally, however, we can define an object mapping |O :: Type -> Type| for a new functor |ado|:\notefoot{Experiment with different notation for |O a|, e.g., ``$\bar{a}$''.}
\begin{code}
ado :: (a -> b) -> D (O a) (O b)
\end{code}
Each cartesian category |k| has its own notion of categorical product |Prod k a b| (satisfying a universality property), and similarly for cocartesian categories (with categorical products and coproducts coinciding for biproduct categories).
Likewise, each cartesian \emph{closed} category |k| has its own notion of \emph{exponential} objects |Exp k a b|.

The generalized interface for cartesian closed categories with per-category exponentials is as follows:\footnote{These operations support higher-order programming and arise during translation from a typed lambda calculus (e.g., Haskell) to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}.}
%  infixrQ 1 (ExpOp k)
\begin{code}
class Cartesian k => CartesianClosed k where
  type (ExpOp k) :: Type -> Type -> Type
  curry    :: ((Prod k a b) `k` c) -> (a `k` (Exp k b c))
  uncurry  :: (a `k` (Exp k b c)) -> ((Prod k a b) `k` c)
  eval     :: (Prod k ((Exp k a b)) a) `k` b
\end{code}
where |Exp k a b| is a type of ``exponential objects'' (first class functions/arrows) from |a| to |b| for the category |k|.

The property of being a closed cartesian functor requires |O| to preserve categorical products and exponentials, i.e.,
\begin{code}
O (a  :*  b) == Prod D  (O a)  (O b)
O (a  ->  b) == Exp  D  (O a)  (O b)
\end{code}

The usual notion of cartesian products are working fine, so we'll continue to choose |Prod D a b = a :* b|.
\out{The ability to choose |Exp D a b|, however, may solve the computability trouble we ran into with |uncurry| and |eval| in \secreftwo{Uncurry}{Eval}.}
While |adh| being a closed cartesian functor (CCF) from |(->)| to |D| implies an noncomputable |eval| and |uncurry| (\secreftwo{Uncurry}{Eval}), our goal is to define |ExpOp D| and |ado| such that |ado| is a CCF with computable operations.

Consider again the homomorphic specification for |eval| (part of the CCF definition): |eval = ado eval|.
The RHS |eval| (on functions) has type |(a -> b) :* a -> b|, while the LHS |eval| (on |D|) has type
\begin{code}
    D (O ((a -> b) :* a)) (O b)
==  D (O (a -> b) :* O a) (O b)
==  D ((Exp D (O a) (O b)) :* O a) (O b)
\end{code}
The difficulty with our attempt at |eval| in \secref{Eval} was that we were given a (computable) function |f|, but we also needed its (noncomputable) derivative |der f|.
Similarly, with |uncurry| in \secref{Uncurry}, we were given |g :: a -> b -> c|, and we needed not only |g a| but also |der (g a)|.
In both cases the exponential object was a function, but we also needed its (computable) derivative.

This analysis suggests that we include a derivative in the exponential object, simply by choosing |ExpOp D| to be |D| itself.
Additionally, map scalars to themselves and cartesian products to cartesian products:
\begin{code}
O R == R
O (a  :*  b) == Prod D  (O a)  (O b)  == O a :* O b
O (a  ->  b) == Exp  D  (O a)  (O b)  == D (O a) (O b)
\end{code}
\nc\toO{\Varid{o}}
%% \nc\toO{\Varid{obj}}
%format toO = "\toO"
%format unO = "\inv{\toO}"
We will need to convert between |a| and |O a|, which we can do with a family of \emph{linear isomorphisms}\footnote{The implicit requirements for all |HasO| instances are thus that |toO . unO == id|, |unO . toO == id|, and |to| and |unO| are linear.} indexed by |a|:\notefoot{It may be more elegant to combine the functions |toO| and |unO| into a single \emph{isomorphism}.}
\begin{code}
class HasO t where
  type O t
  toO  :: t -> O t
  unO  :: O t -> t
\end{code}
For scalar types |a| and the unit type, |O a == a|, the isomorphism is trivial:
\begin{code}
instance HasO R where
  type O R = R
  toO  = id
  unO  = id

instance HasO () where
  type O () = ()
  toO  = id
  unO  = id
\end{code}
For products, convert components independently:\footnote{Recall that |(f *** g) (a,b)| = |(f a, g b)|, so |toO (a,b) = (toO *** toO) (a,b) = (toO a, toO b)|, and similarly for |unO|.}
\begin{code}
instance (HasO a, HasO b) => HasO (a :* b) where
  type O (a :* b) = O a :* O b
  toO  = toO  ***  toO
  unO  = unO  ***  unO
\end{code}
The new functor |ado| converts its given |a -> b| to |O a -> O b| and then applies the |adh| functor:\notefoot{Consider dropping the |(+=>)| definition and uses here.}
%format wrapO = wrap"_{\!o}"
%format wrapO = "\subo{"wrap"}"
%% %format wrapO = wrap
%format unwrapO = "\inv{"wrapO"}"
\begin{code}
(+=>) :: (p' -> p) -> (q -> q') -> ((p -> q) -> (p' -> q'))
f +=> h = \ g -> h . g . f

wrapO :: (a -> b) -> (O a -> O b)
wrapO = unO +=> toO

unwrapO :: (O a -> O b) -> (a -> b)
unwrapO = toO +=> unO

ado :: (a -> b) -> D (O a) (O b)
ado = adh . wrapO

unado :: D (O a) (O b) -> (a -> b)
unado = unwrapO . unadh
\end{code}
\begin{lemma}[\provedIn{wrapO-iso}]\lemLabel{wrapO-iso}
|wrapO| and |unwrapO| form a linear isomorphism.
\end{lemma}
\begin{lemma}[\provedIn{ado-iso}]\lemLabel{ado-iso}
|ado| and |unado| form a linear isomorphism.
\end{lemma}
\begin{lemma}[\provedIn{wrapO-cartesian}]\lemLabel{wrapO-cartesian}
|wrapO| is a cartesian functor.
\end{lemma}

%% \note{To do: reconsider theorems vs lemmas vs corollaries. I think more lemmas.}

The cartesian category operations already defined on |D| \citep{Elliott-2018-ad-icfp} are solutions to homomorphism equations saying that |adh| is a cartesian functor.
Thanks to the simple, regular structure of |toO| and |unO|,
\begin{theorem}\thmLabel{ado-cartesian}
|ado| is a cartesian functor.
\end{theorem}
Proof: |adh| is a cartesian functor \citep{Elliott-2018-ad-icfp}, as is |wrapO| (\lemRef{wrapO-cartesian}), so |ado = adh . wrapO| is also.

What about exponentials and cartesian \emph{closure}?
As mentioned above, |O (a -> b) == Exp D (O a) (O b) == D (O a) (O b)|, which suggests using |ado| and |unado| for |toO| and |unO|:
\begin{code}
instance (HasO a, HasO b) => HasO (a -> b) where
  type O (a -> b) = D (O a) (O b)
  toO  = ado
  unO  = unado
\end{code}

A useful consequence:
\begin{lemma}[\provedIn{wrapO-curry}]\lemLabel{wrapO-curry}
$$|wrapO (curry f) == adh . curry (wrapO f)|$$
\end{lemma}
\begin{corollary}\corLabel{curry-wrapO}
$$|curry (wrapO f) == unadh . wrapO (curry f)|$$
\end{corollary}
\begin{proof}
Left-compose |unadh| with both sides of \lemRef{wrapO-curry}; then simplify and reverse the resulting equation.
\end{proof}

Let's now try to solve the CCF equations for |ado|.
This time begin with |eval|:
\begin{lemma}[\provedIn{ado-eval}] \lemLabel{ado-eval}
With the following (effective) definition of |eval| on |D|, |eval == ado eval|:
\begin{code}
eval = D (\ (D h,a) -> let (b,f') = h a in (b, at a . unadh !!! f'))
\end{code}
\end{lemma}
For |uncurry|, use the standard definition |uncurry g = eval . first g|.

%format fw = "\subo{f}"
The definition of |curry| in \secref{Curry} worked fine, but we'll need to check again, as we did with the cartesian category operations (\thmRef{ado-cartesian}).
The homomorphism equation is |curry (ado f) == ado (curry f)|, to be solved for the unknown LHS |curry| (on |D|), with |f :: a :* b -> c|.
First let |fw = wrapO f|.
Simplify the LHS:
\begin{code}
    curry (ado f)
==  curry (adh (wrapO f))                  -- |ado| definition
==  curry (adh fw)                         -- |fw| definition
==  curry (D (fw &&& der fw))              -- |adh| definition
\end{code}
Then the RHS:\notefoot{State, prove, and use a lemma about |adh (g . f) a| for linear |g| and another for linear |f|.
Maybe also |ado (g . f) a| for linear |g| or |f|.}
\begin{lemma}[\provedIn{ado-curry}]\lemLabel{ado-curry}~
\begin{code}
ado (curry f) ==
  D  (\ a ->  (  D (\ b -> (fw (a,b), derr fw (a,b)))
              ,  \da -> D (\ b -> (derl fw (a,b) da, at da . derr (derl fw) (a,b)))))
\end{code}
where |fw = wrapO f|.
\end{lemma}
The RHS uses |fw (a,b)| and |der fw (a,b)| (via its components |derl fw (a,b)| and |derr fw (a,b)|), but it also uses a \emph{second} partial derivative |derr (derl fw) (a,b)|, which is not available from the |curry| argument |D (fw &&& der fw)|.

\sectionl{Where Are We?}

Let's now reflect on what we've learned so far:
\begin{itemize}

\item The cartesian functor (CF) |adh :: (a -> b) -> D a b| also forms a cartesian \emph{closed} functor (CCF) with suitable definitions of |curry|, |uncurry|, and |eval|, but not computably (\secref{Cartesian Closure, first attempt}).
More specifically, |curry| is computable, but |uncurry| and |eval| are not, since they need to synthesize derivatives of regular computable functions.

\item General categorical functors can remap objects (here, types) as well as morphisms (here, functions).
Exploiting this degree of freedom, define |ado :: (a -> b) -> D (O a) (O b)|, where |O :: Type -> Type| replaces regular functions with computably differentiable functions, i.e., |O (u -> v) = D (O u) (O v)|.
This new function is defined in terms of the old one, |ado = adh . wrapO|, and indeed |ado| is a CF as well.
In the absence of higher-order functions, |O| is the identity mapping, and |ado| coincides with |adh|.

\item Computably satisfying the required homomorphism properties of |ado| for |uncurry| and |eval| becomes easy, since the operations are \emph{given} the required derivatives rather than having to synthesize them.
Unfortunately, now |curry| becomes noncomputable because it has to synthesize partial \emph{second} derivatives.

\end{itemize}

\sectionl{Higher-Order Derivatives}

Where can we go from here?
An obvious next step is to add second order derivatives to the representation of computably differentiable functions.
It seem likely, however, that the CCF specification would reveal that |curry| needs at least third order derivatives, and so on.
In other words, differentiation of higher-order functions requires all higher-order derivatives of functions.

In order to construct higher-order derivatives, it will help to examine the linearity properties of our familiar categorical vocabulary, which turns out to be mostly linear with just a bit of bilinearity.
As noted in \cite{Elliott-2018-ad-icfp}, the categorical operation |id|; the cartesian operations |exl|, |exr|, |dup|; and the cocartesian operations |inl|, |inr|, and |jam| are all linear.
\lemRefTwo{fork-iso-linear}{join-iso-linear} have already noted that the functions |fork| and |join| (uncurried versions of |(&&&)| and |(!!!)| defined in \secref{Curry}) are linear (as well as isomorphisms).
Next, let |comp| be uncurried composition:\notefoot{Maybe define |comp| only for linear maps.}
\begin{code}
comp :: Category k => (b `k` c) :* (a `k` b) -> (a `k` c)
comp = uncurry (.)
\end{code}

\begin{lemma}[\provedIn{comp-bilinear}]\lemLabel{comp-bilinear}
On linear maps, |comp| is bilinear.
\end{lemma}

%% The bilinearity of |comp| gives it several useful properties:
\begin{lemma}[\provedIn{bilinear-props}]\lemLabel{bilinear-props}
Given any bilinear function |h|:
\begin{enumerate}
\item |curry h a| is linear for all |a|.
\item |curry' h b| is linear for all |b|.
\item |curry h| and |curry' h| are linear.
\item |der h| is linear.
\end{enumerate}
\end{lemma}

\begin{corollary}\corLabel{comp-props}
On linear maps,
\begin{enumerate}
\item |(g . NOP)| is linear for all |g|.
\item |(NOP . f)| is linear for all |f|.
\item |(.)| and |flip (.)| are linear.
\item |der comp| is linear.
\end{enumerate}
\end{corollary}
%if False
\begin{lemma}[\provedIn{compose-linear}]\lemLabel{compose-linear}
Function composition has the following properties:
\begin{enumerate}
\item |(NOP . f)| is linear for all functions |f|.
\item |(g . NOP)| is linear for all linear functions |g|.
\item |(.)| and |flip (.)| are linear.
\item |comp| on linear maps is bilinear.
\item |der comp (g,f) = (NOP . f) !!! (g . NOP)|
\end{enumerate}
\end{lemma}
%endif

These properties will help re-express \thmRefTwo{deriv-compose}{deriv-cross} and related facts in a form more amenable to constructing higher derivatives:
\nc\lemLabelPF[1]{\label{deriv-pointfree-#1}}
\nc\lemRefPF[1]{\lemRef{deriv-pointfree}\ref{deriv-pointfree-#1}}
\begin{lemma}[\provedIn{deriv-pointfree}]\lemLabel{deriv-pointfree}~
\begin{enumerate}
\item \lemLabelPF{compose}
  |der (g . f) == comp . (der g . f &&& der f)|.

\item \lemLabelPF{cross}
  |der (f *** g) == cross . (der f *** der g)|.

\item \lemLabelPF{fork}
  |der (f &&& g) == fork . (der f &&& der g)|.

\item \lemLabelPF{linear}
  For a \emph{linear} function |f|, |der f == const f|.

\item \lemLabelPF{pair-domain}
  For any function |f :: a :* b -> c|, |der f == join . (derl f *** derr f)|.

\item \lemLabelPF{bilinear}
  For a \emph{bilinear} function |f :: a :* b -> c|, |der f == join . (curry' f *** curry f) . swap|.

\item \lemLabelPF{comp}
  On linear maps, |der comp == join . (flip (.) *** (.)) . swap|.
\end{enumerate}
\end{lemma}

%% %format dern (n) f = f"^{("n")}"
%format dern (n) = der"^{"n"}"

Let us now consider the task of constructing \emph{all} orders of derivatives.
The |D| category encapsulates a function |f| and its first derivative, i.e., the zeroth and first derivatives of |f|, which we might write as ``|adh f = dern 0 f &&& dern 1 f|''.
Our new category will encapsulate \emph{all} derivatives of |f|, i.e.,\footnote{Take |&&&| to be \emph{right}-associative.}
\begin{code}
ders f = dern 0 f &&& dern 1 f &&& dern 2 f &&& cdots
\end{code}
where
\begin{code}
dern 0      f = f
dern (n+1)  f = dern n (der f)
\end{code}
Then
\begin{code}
    ders f
==  dern 0 f &&& dern 1 f &&& dern 2 f &&& dern 3 f &&& cdots
==  f &&& dern 1 f &&& dern 2 f &&& dern 3 f &&& cdots
==  f &&& dern 0 (der f) &&& dern 1 (der f) &&& dern 2 (der f) &&& cdots
==  f &&& ders (der f)
\end{code}
which we can take as a recursive definition of |ders|.
Define a corresponding type of infinitely differentiable functions:\footnote{For notational simplicity, we'll drop the |newtype| isomorphisms.}
\begin{code}
type Ds a b = a -> T a b

type T a b = b :* T a (a :-* b)

ders :: (a -> b) -> Ds a b
ders f = f &&& ders (der f)
\end{code}
We will want to find cartesian category operations for |Ds| such that |ders| is a cartesian functor (CF), which will be coinductively assumed at several points below.

Start with the constant-zero function\footnote{As usual, types are restricted to vector spaces over a common field, which we can take to be |R|}: |zero :: a -> b|:
\begin{code}
    ders zero
==  zero &&& ders (der zero)  -- |ders| definition
==  zero &&& ders zero        -- |der zero == const zero == zero|
==  zero &&& zero             -- coinduction
==  zero                      -- Zero on pairs
\end{code}
Then constant functions more generally:
\begin{code}
    ders (const b)
==  const b &&& ders (der (const b))  -- |ders| definition
==  const b &&& ders zero             -- |der (const b) == zero|
==  const b &&& zero                  -- above
\end{code}
Next, linear functions |f|:
\begin{code}
    ders f
==  f &&& ders (der f)      -- |ders| definition
==  f &&& ders (const f)    -- |f| linearity
==  f &&& const f &&& zero  -- above
\end{code}
We will have several uses of this formula, so name it:
\begin{code}
linear :: (a :-* b) -> Ds a b
linear f = f &&& const f &&& zero
\end{code}
For instance, the following definitions of |id|, |exl| and |exr| satisfy the associated homomorphism (cartesian functor) properties:
\begin{code}
id   == linear id
exl  == linear exl
exr  == linear exr
\end{code}

\noindent
Next, \emph{bilinear} functions |g|:
\begin{code}
    ders g
==  g &&& ders (der g)                                   -- |ders| definition
==  g &&& linear (der g)                                 -- derivative of bilinear is linear
==  g &&& linear (join . (curry' g *** curry g) . swap)  -- \lemRefPF{bilinear}
\end{code}

\noindent
Specialize to uncurried linear map composition:
\begin{code}
    ders comp
==  comp &&& linear (join . (curry' comp *** curry comp) . swap)  -- above
==  comp &&& linear (join . (flip (.) *** (.)) . swap)            -- |comp| definition
\end{code}
Name |ders comp| for future use:
\begin{code}
comp' :: Ds ((b :-* c) :* (a :-* b)) (a :-* c)
comp' = comp &&& linear (join . (flip (.) *** (.)) . swap)
\end{code}
Then sequential compositions:
\begin{code}
    ders (g . f)
==  g . f &&& ders (der (g . f))                                    -- |ders| definition
==  g . f &&& ders (comp . (der g . f &&& der f))                   -- \lemRefPF{compose}
==  g . f &&& ders comp . (ders (der g) . ders f &&& ders (der f))  -- coinduction
==  g . f &&& comp' . (ders (der g) . ders f &&& ders (der f))      -- above
\end{code}
Note that all of the components here (|g|, |f|, |ders (der g)|, |ders f|, and |ders (der f)|) are available in |ders g| and |ders f|, so we have a computable recipe for |(.)| on |Ds|.
\note{To do: fill in the details.}

Finally, |f &&& g|:
\begin{code}
    ders (f &&& g)
==  (f &&& g) &&& ders (der (f &&& g))                           -- |ders| definition
==  (f &&& g) &&& ders (fork . (der f &&& der g))                -- \lemRefPF{fork}
==  (f &&& g) &&& ders fork . (ders (der f) &&& ders (der g))    -- coinduction
==  (f &&& g) &&& linear fork . (ders (der f) &&& ders (der g))  -- |fork| linearity (\lemRef{fork-iso-linear})
\end{code}
Again, the components here (|f|, |g|, |ders (der f)|, and |ders (der g)|) are all available from |ders f| and |ders g|, so we have a computable recipe for |(&&&)| on |Ds|.
\note{To do: fill in the details.}

\workingHere



\sectionl{Avoiding redundant computation}

The |adh| functor was carefully chosen to enable elimination of redundant computation between a function and its derivative.
The potential for redundancy is apparent in the chain rule (\thmRef{deriv-compose}):
$$|der (g . f) a == der g (f a) . der f a|$$
This theorem reveals that computation of |(g . f) a| and |der (g . f) a| at both involve computing |f a|.
Since sequential composition is a very commonly used building block of computations, it is thus typical for functions and their derivatives to involve common work.
%format ad0 = der QQ"_{\scriptscriptstyle 0}\!\!^+\!"
This fact motivates the choice |adh f = f &&& der f| over |ad0 f = (f,der f)| \citep[Section 3.1]{Elliott-2018-ad-icfp}.
While both options can give rise to compositional (functorial) AD, |ad0| precludes sharing of work, while |adh| enables such sharing, with just a bit of care:
$$|D gh . D fh == D (\ a -> let { (b,f') = fh a ; (c,g') = gh b } in (c, g' . f'))|$$
\nc\lemLabelCompFork[1]{\label{cross-fork-#1}}
\nc\lemRefCompFork[1]{\lemRef{cross-fork}\ref{cross-fork-#1}}
%format assocR = assoc"_{\!R}"
We can calculate this definition in a categorical/pointfree form using \lemRefPF{compose}:\notefoot{Define and use a variant of |adh| that omits |D|. Then introduce |D| in ``We can thus define ...''.}\footnote{The |assocR| operation in monoidal categories is defined for functions as |assocR ((a,b),c) = (a,(b,c))|.}
\begin{code}
    adh (g . f)
==  D (g . f &&& der (g . f))                                       -- |adh| definition
==  D (g . f &&& comp . (der g . f &&& der f))                      -- \lemRefPF{compose}
==  D (second comp . (g . f &&& (der g . f &&& der f)))             -- \lemRefCompFork{second} below
==  D (second comp . assocR . ((g . f &&& der g . f) &&& der f))    -- \note{justify this step}
==  D (second comp . assocR . ((g &&& der g) . f &&& der f))        -- \citet[Section 1.5.1]{Gibbons2002Calculating}.
==  D (second comp . assocR . (unD (adh g) . f &&& der f))          -- |adh| definition
==  D (second comp . assocR . first (unD (adh g)) . (f &&& der f))  -- \lemRefCompFork{first} below
==  D (second comp . assocR . first (unD (adh g)) . unD (adh f))    -- |adh| definition
\end{code}
We can thus define
\begin{code}
D gh . D fh  = D (second comp . assocR . first (unD (D gh)) . unD (D fh))
             = D (second comp . assocR . first gh . fh)
\end{code}
with the consequence that |adh g . adh f == adh (g . f)|.
In this form, |fh| and |gh| each appear once, so as long as |D fh| and |D gh| are nonredundant, |D gh . D fh| will be nonredundant as well.
Inlining the definitions of |comp| and of |second|, |assocR|, and |first| for functions and then simplifying yields the pointful definition above.

\begin{lemma}\lemLabel{cross-fork}
The following properties hold for |(&&&)|:
\begin{enumerate}
\item \lemLabelCompFork{cross}
  $|(h *** k) . (f &&& g) == h . f &&& k . g|$
\item \lemLabelCompFork{first}
  $|first h . (f &&& g) == h . f &&& g|$
\item \lemLabelCompFork{second}
  $|second k . (f &&& g) == f &&& k . g|$
\end{enumerate}
\end{lemma}
\noindent
Proof: For \emph{a}, see \citet[Section 1.5.1]{Gibbons2002Calculating}.
Then \emph{b} and \emph{c} follow as corollaries from the definitions |first h = h *** id| and |second k = id *** k|.

%if False

\sectionl{Deep Zipping}

Consider the following alternative definition to |ders| from \secref{Higher-Order Derivatives}:
\begin{code}
ders' f = f &&& der (ders' f)

    ders' f
==  f &&& der (ders' f)
==  f &&& der (f &&& der (ders' f))
==  f &&& der f &&&& der (der (ders' f))

ders' f  = f &&& der (f &&& der (f &&& ...))
         = f &&& der f &&&& der (der (f &&& ...))


der (f &&& g) = fork . (der f &&& der g) = der f &&&& der g

f :: a -> b

ders f :: a -> T a b

der f :: a -> a :-* b

ders (der f) ::

ders f :: a -> T a b

der (ders f) :: a -> a :-* T a b




der f &&& der (der f)

f :: a -> b
der f :: a -> a :-* b
der (der f) :: a -> a :-* a :-* b

der f &&& der (der f) :: a -> (a :-* b) :* (a :-* a :-* b)
fork . (der f &&& der (der f)) :: a -> a :-* b :* (a :-* b)

\end{code}

%endif

\workingHere

The calculation of |adh (g . f)| above is somewhat tedious, and it's unclear how to extend it to higher derivatives.
Some of the complexity comes from routing |der f| (i.e., |exr . adh f|) around |adh g| to compose derivatives (|der g (f a) . der f a|).
The motivation for this routing arises from an asymmetry in |D|, namely that it maps just a primal value to a primal \emph{and} derivative:
%format prime = "^\prime"
%format adhp = adh prime
%% %format adhp = "\hat{\dot{"der"}}"
\begin{code}
adhp : (a -> b) -> (a -> b :* (a :-* b))
adhp f = f &&& der f
\end{code}
%% %format adt = twiddle(der)
%format adt = "\tilde{\der}"
%format adtp = adt prime
Suppose instead that we thread the primal/derivative pairs \emph{in} as well as out.\notefoot{To do: find a (non-effective) \emph{definition} for |adtp|. Well, I know how to define |adtp| in this case:
\begin{code}
adtp f (a,q') = (b, f' . q') where (b,f') = adhp f a
\end{code}
How to extend to higher derivatives?}\footnote{This formulation is similar to the use of dual numbers in forward-mode AD \needcite{}.}
\begin{code}
adtp : (a -> b) -> forall z. NOP a :* (z :-* a) -> b :* (z :-* b)
forall f q. NOP adtp f . adhp q == adhp (f . q)  -- specification
\end{code}
Moreover, |adtp| suffices to compute |adhp|:
\begin{code}
    adhp f
==  adhp (f . id)
==  adtp f . adhp id
==  adtp f . (id &&& der id)
==  adtp f . (\ z -> (z,id))
\end{code}

Now consider sequential composition:
\begin{code}
    adtp (g . f) . adhp q
==  adhp ((g . f) . q)          -- specification of |adtp|
==  adhp (g . (f . q))          -- associativity of |(.)|
==  adtp g . adhp (f . q)       -- specification of |adtp|
==  adtp g . (adtp f . adhp q)  -- specification of |adtp|
==  (adtp g . adtp f) . adhp q  -- associativity of |(.)|
\end{code}
Hence |adtp (g . f) == adtp g . adtp f| by the following lemma.\notefoot{Move lemma and proof to the appendix.}
\begin{lemma}\lemLabel{uncurry-epi}
For any |f :: a -> b -> c|, if |uncurry f| is surjective and |forall x :: a . SP g . f x == g' . f x|, then |g == g'|.
\end{lemma}
\begin{proof}~
\begin{code}
    g . uncurry f
==  \ (x,y) -> g (uncurry f (x,y))   -- |eta| conversion
==  \ (x,y) -> g (f x y)             -- |uncurry| on functions
==  uncurry (\ x y -> g (f x y))     -- |uncurry| on functions
==  uncurry (\ x y -> (g . f x) y)   -- |(.)| on functions
==  uncurry (\ x -> g . f x)         -- |eta| conversion
==  uncurry (\ x -> g' . f x)        -- assumption
==  uncurry (\ x y -> (g' . f x) y)  -- |eta| conversion
==  uncurry (\ x y -> g' (f x y))    -- |(.)| on functions    
==  \ (x,y) -> g' (f x y)            -- |uncurry| on functions
==  \ (x,y) -> g' (uncurry f (x,y))  -- |uncurry| on functions
==  g' . uncurry f                   -- |eta| conversion
\end{code}
Since |uncurry f| is surjective, |g == g'|.
\end{proof}

Likewise, consider |id|:
\begin{code}
    adtp id . adhp q
==  adhp (id . q)
==  adhp q
==  id . adhp q
\end{code}
By \lemRef{uncurry-epi}, |adtp id = id|.

\workingHere

%format &&&& = "\mathbin{\blacktriangle}"
Parallel composition:
\begin{code}
    adtp (f &&& g) . adhp q
==  adhp ((f &&& g) . q)                  -- |adtp| specification
==  adhp (f . q &&& g . q)                -- Cartesian law
==  adhp (f . q) &&&& adhp (g . q)        -- \note{for suitable |(&&&&)|, probably as with |(&&&)| for |D|}
==  adtp f . adhp q &&&& adtp g . adhp q  -- |adtp| specification
==  (adtp f &&&& adtp g) . adhp q         -- \note{To prove about |(&&&&)|}
\end{code}

\sectionl{What's Next?}

\note{Yet to come:
\begin{itemize}
\item Avoid redundant computation in |Ds|.
 Doing so is fairly easy in |D| (zeroth and first derivatives), but I don't yet see how in |Ds| (all derivatives).
\item Spell out the |Category| and |Cartesian| instances that result from solving the cartesian functor equations as in \secref{Higher-Order Derivatives}.
\item Cartesian \emph{closure} (|curry| and |eval|/|uncurry|) for |Ds|, exploiting higher-order derivatives.
\item Variation of |ders|: |ders' f = f &&& der (ders' f)|.
\end{itemize}
}


\sectionl{Related Work}

The most closely related work I'm aware of is by \cite{Vytiniotis-2019-differentiable-curry}\footnote{I am in the middle of an in-depth conversation with authors.}, who also define an algorithm around the language of cartesian closed categories.
There appear to be some significant shortcomings, however, at least when considered as an extension to \cite{Elliott-2018-ad-icfp}:
\begin{itemize}
\item
  Although the work is referred to as ``differentiable programming'', it appears to lack a specification and proof that match this claim, i.e., one defined by the mathematical operation of differentiation.
  As such, it's unclear to me whether the algorithm is about differentiation or something else.
  In contrast, the specification at the center of \cite{Elliott-2018-ad-icfp} (and the extensions described above) is just (Fréchet) differentiation itself, combined with the original function as needed by the chain rule, or rather the requirement that the function-with-derivative satisfies a standard collection of homomorphism properties.
  Correctness of the algorithm was defined as faithfulness to this simple specification, and the algorithm is systematically derived from this specification and hence is correct by construction.
\item
  Functions are already well-defined as a vector space, and thus linear maps (including derivatives) are as well, but the authors chose a different notion.
  They write
  \begin{quotation} \noindent
  [...] what should be the tangent space of a function type?
  Perhaps surprisingly, a function type itself is not the right answer.
  We provide two possible implementations for function tangents and differentiable currying, and explain the tradeoffs.
  \end{quotation}
  There is no explanation, however, of what makes their answers ``right'' and the unsurprising answer wrong.
It is unclear what it could possibly mean for their answer to be right, since the usual notion of derivative of a function |f :: a -> b| between vector spaces has type |a -> a :-* b| for all vector spaces |a| and |b|, \emph{including function types}.
This observation seems to contradict the claim that the tangent space for a function types is not a function type.
\item
  The algorithm presented is limited to reverse mode rather than a general AD algorithm as in \cite{Elliott-2018-ad-icfp} and the work described above.
\end{itemize}

Another related paper is \cite{Brunel2019Backprop}.
The authors write (in Section 1)
\begin{quotation} \noindent
However, Elliot's approach is still restricted to first-order programs (i.e., computational graphs): as far as we understand, the functor D is cartesian but not cartesian closed, so the higher-order primitives ($\lambda$-abstraction and application) lack a satisfactory treatment. This is implicit in Sect. 4.4 of \cite{Elliott-2018-ad-icfp}, where the author states that he only uses biproduct categories: it is well-known that non-trivial cartesian closed biproduct categories do not exist.
\end{quotation}
The confusion here---which was mistakenly encouraged by \cite{Elliott-2018-ad-icfp}---is the idea that the category of differentiable functions itself is (or need be) a biproduct category.
Rather, all that was needed is that the various representations of \emph{linear maps} (derivatives) are biproduct categories.
This requirement is easily satisfied by construction, since these representations are all calculated from their denotation (linear functions, itself a biproduct category) via simple cocartesian functors.

\bibliography{bib}

\appendix

\sectionl{Proofs}

\subsection{\lemRef{deriv-pair-domain}}\proofLabel{deriv-pair-domain}

Suppose we have a function |f :: a :* b -> c|, and we want to compute its derivative at a point in its (pair-valued) domain.
Because linear maps (derivatives) form a cocartesian category,\footnote{The cocartesian law |h = h . inl !!! h . inr| is dual to the cartesian law |h = exl . h &&& exr . h| \citep{Gibbons2002Calculating}.}
\begin{code}
der f (a,b) == der f (a,b) . inl !!! der f (a,b) . inr
\end{code}
Noting that (for linear maps) |inl da = (da,0)| and |inr db = (0,db)|, we can see that the ``partial derivatives'' (|der f (a,b) . inl| and |der f (a,b) . inr|) allow only one half of a pair to change.

Next, note that |der f (a,b) . inl = der (f . (,b)) a|, by the following equational reasoning:
\begin{code}
    der (f . (,b)) a
==  der f ((,b) a) . der (,b) a                      -- chain rule (\thmRef{deriv-compose})
==  der f (a,b) . der (,b) a                         -- |(,b)| definition
==  der f (a,b) . der (inl + const (0,b)) a          -- |inl| on functions, and meaning of |(,b)|
==  der f (a,b) . (der inl a + der (const (0,b)) a)  -- linearity of |(+)|
==  der f (a,b) . der inl a                          -- |der (const z) a == 0|
==  der f (a,b) . inl                                -- linearity of |inl|; \thmRef{deriv-linear}
\end{code}
Likewise, |der f (a,b) . inr = der (f . (a,)) b|.

\subsection{\corRef{deriv-uncurry}}\proofLabel{deriv-uncurry}

\begin{code}
    der (uncurry g) (a,b)
==  derl (uncurry g) (a,b) !!! derr (uncurry g) (a,b)      -- \lemRef{deriv-pair-domain}
==  der (uncurry g . (,b)) a !!! der (uncurry g . (a,)) b  -- |derl| and |derr| definitions
==  der (\ a' -> uncurry g (a',b)) a !!!                   -- $\eta$ expansion and simplification
    der (\ b' -> uncurry g (a,b')) b
==  der (\ a' -> g a' b) a !!! der (\ b' -> g a b') b      -- |uncurry| on functions
==  der (at b . g) a !!! der (g a) b                       -- |at| definition and $\eta$ reduction
==  der (at b) (g a) . der g a !!! der (g a) b             -- chain rule (\thmRef{deriv-compose})
==  at b . der g a !!! der (g a) b                         -- linearity of |at|
\end{code}

\subsection{\corRef{deriv-eval}}\proofLabel{deriv-eval}

\begin{code}
    der eval (f,a)
==  derl eval (f,a) !!! derr eval (f,a)          -- \lemRef{deriv-pair-domain}
==  der (eval . (,a)) f !!! der (eval . (f,)) a  -- |derl| and |derr| alternative definitions
==  der (at a) f !!! der f          a            -- |eval| on functions; |at| definition
==  at a !!! der f a                             -- linearity of |at a|
==  \ (df,dx) -> df a + der f a dx               -- |(!!!) on linear maps|; |at| definition
\end{code}

Alternatively, calculate |der eval| via |uncurry|:
\begin{code}
    der eval (f,a)
==  der (uncurry id) (f,a)            -- |eval = uncurry id|
==  at a . der id a !!! der (id f) a  -- \corRef{deriv-uncurry}
==  at a . id !!! der f a             -- |id| linearity
==  at a !!! der f a                  -- |id| as identity
\end{code}

\subsection{\lemRef{deriv-function-codomain}}\proofLabel{deriv-function-codomain}

\begin{code}
    forkF (\ b -> der (at b . g) a)
==  \ da b -> der (at b . g) a da              -- |forkF| definition
==  \ da b -> (der (at b) (g a) . der g a) da  -- chain rule (\thmRef{deriv-compose})
==  \ da b -> (at b . der g a) da              -- |at b| linearity
==  \ da b -> at b (der g a da)                -- |(.)| on functions
==  \ da b -> der g a da b                     -- |at| definition
==  der g a                                    -- $\eta$ reduction (twice)
\end{code}

\subsection{\corRef{deriv-curry}}\proofLabel{deriv-curry}

\begin{code}
    der (curry f) a
==  forkF (\ b -> der (at b . curry f)) a           -- \lemRef{deriv-function-codomain}
==  forkF (\ b -> der (\ a -> at b (curry f a))) a  -- |(.)| on functions
==  forkF (\ b -> der (\ a -> curry f a b)) a       -- |at| definition
==  forkF (\ b -> der (\ a -> f (a,b))) a           -- |curry| on functions
==  forkF (\ b -> der (f . (,b))) a                 -- |(, b)| definition
==  forkF (\ b -> derl f (a,b))                     -- |derl| definition
==  forkF (derl f . (a,))                           -- |(a,)| definition
\end{code}


\subsection{\lemRef{wrapO-iso}}\proofLabel{wrapO-iso}

The functions |wrapO| and |unwrapO| form an isomorphism:
\begin{code}
    unwrapO (wrapO f)
==  unwrapO (toO . f . unO)        -- |wrapO| definition
==  unO . (toO . f . unO) . toO    -- |unwrapO| definition
==  (unO . toO) . f . (unO . toO)  -- |(.)| associativity 
==  id . f . id                    -- |unO . toO == id|
==  f                              -- |id| is identity for |(.)|

    wrapO (unwrapO h)
==  wrapO (unO . h . toO)          -- |unwrapO| definition
==  toO . (unO . h . toO) . unO    -- |wrapO| definition
==  (toO . unO) . h . (toO . unO)  -- |(.)| associativity
==  id . h . id                    -- |toO . unO == id|
==  h                              -- |id| is identity for |(.)|
\end{code}

Linearity of |wrapO| and |unwrapO| follows from two facts:
\begin{itemize}
\item |(NOP . f)| is linear for all |f|.
\item |(g . NOP)| is linear for all \emph{linear} |g|.
\end{itemize}
Proof: exercise.

\subsection{\lemRef{ado-iso}}\proofLabel{ado-iso}

The functions |ado| and |unado| form an isomorphism:
\begin{code}
    unado . ado
==  unwrapO . unadh . adh . wrapO  -- |unado| and |ado| definitions
==  unwrapO . wrapO                -- |unadh . adh == id|
==  id                             -- |unwrapO . wrapO == id|

    ado . unado
==  adh . wrapO . unwrapO . unado  -- |ado| and |unado| definitions
==  adh . unado                    -- |unwrapO . wrapO == id|
==  id                             -- |unadh . adh == id|
\end{code}

Linearity of |ado| and |unado| follows from linearity of |adh| and |unadh| and \lemRef{wrapO-iso}.

\subsection{\lemRef{wrapO-cartesian}}\proofLabel{wrapO-cartesian}

The proof that |wrapO| is a cartesian functor mainly exploit the regular structure of |toO| and |unO|:

\begin{code}
    wrapO id
==  toO . id . unO                             -- |wrapO| definition
==  toO . unO                                  -- |id| is identity for |(.)|
==  id                                         -- |toO . unO == id|

    wrapO (g . f)
==  toO . g . f . unO                          -- |wrapO| definition
==  toO . g . unO . toO . f . unO              -- |unO . toO == id|
==  (toO . g . unO) . (toO . f . unO)          -- |adh| is a functor
==  wrapO g . wrapO f                          -- |wrapO| definition

    wrapO (f *** g)                            -- |ado| definition
==  toO . (f *** g) . unO                      -- |wrapO| definition
==  (toO *** toO) . (f *** g) . (unO *** unO)  -- |toO| on products
==  toO . f . unO *** toO . g . unO            -- monoidal category law
==  toO . f . unO *** toO . g . unO            -- |adh| is a monoidal functor
==  wrapO f *** wrapO g                        -- |wrapO| definition

    wrapO exl                                  -- |ado| definition
==  toO . exl . unO                            -- |wrapO| definition
==  toO . exl . (unO *** unO)                  -- |unO| on products
==  toO . unO . exl                            -- |exl . (f *** g) == f . exl| for cartesian categories
==  exl                                        -- |toO . unO == id|

    wrapO exr                                  -- |ado| definition
==  toO . exr . unO                            -- |wrapO| definition
==  toO . exr . (unO *** unO)                  -- |unO| on products
==  toO . unO . exr                            -- |exr . (f *** g) == g . exr| for cartesian categories
==  exr                                        -- |toO . unO == id|

    wrapO dup                                  -- |ado| definition
==  toO . dup . unO                            -- |wrapO| definition
==  toO . (unO *** unO) . dup                  -- |dup . f == (f *** f) . dup| for cartesian categories
==  toO . unO . dup                            -- |unO| on products
==  dup                                        -- |toO . unO == id|
\end{code}

\subsection{\lemRef{wrapO-curry}}\proofLabel{wrapO-curry}

\begin{code}
    wrapO (curry f)
==  toO . curry f . unO          -- |wrapO| definition
==  ado . curry f . unO          -- |toO| on functions
==  adh . wrapO . curry f . unO  -- |ado| definition
==  adh . curry (wrapO f)        -- below
\end{code}

For this last step,
\begin{code}
    wrapO . curry f . unO
==  \ a -> wrapO (curry f (unO a))          -- $\eta$ expansion
==  \ a -> toO . curry f (unO a) . unO      -- |wrapO| definition
==  \ a b -> toO (curry f (unO a) (unO b))  -- $\eta$ expansion
==  \ a b -> toO (f (unO a, unO b))         -- |curry| on functions
==  \ a b -> toO (f (unO (a,b)))            -- |unO| on pairs
==  \ a b -> wrapO f (a,b)                  -- |wrapO| definition
==  curry (wrapO f)                         -- |curry| on functions
\end{code}

Equivalently, |curry (wrapO f) == unadh . wrapO (curry f)|.\notefoot{Maybe this form will help simplify another proof.}

\subsection{|ado| and |eval|}\proofLabel{ado-eval}

The homomorphism equation is |eval == ado eval|.
Simplifying the RHS,
\begin{code}
    ado eval
==  adh (wrapO eval)                                                        -- |ado| definition
==  adh (toO . eval . unO)                                                  -- |wrapO| definition
==  adh (toO . eval . (unado *** unO))                                      -- |unO| on |(a -> b) :* a|
==  adh (\ (fh,a) -> (toO . eval . (unado *** unO)) (fh,a))                 -- |eta| expansion
==  adh (\ (fh,a) -> toO (eval (unado fh, unO a)))                          -- |(.)| and |(***)| on functions
==  adh (\ (fh,a) -> toO (unado fh (unO a)))                                -- |eval| on functions
==  adh (\ (fh,a) -> toO (unwrapO (unadh fh) (unO a)))                      -- |unado| definition
==  adh (\ (fh,a) -> toO ((unO . unadh fh . toO) (unO a))                   -- |unwrapO| definition
==  adh (\ (fh,a) -> toO (unO (unadh fh (toO (unO a)))))                    -- |(.)| on functions
==  adh (\ (fh,a) -> unadh fh a)                                            -- |toO . unO == id|
==  adh (uncurry unadh)                                                     -- |uncurry| on functions
==  D (\ (fh,a) -> (uncurry unadh (fh,a), der (uncurry unadh) (fh,a)))      -- |adh| definition
==  D (\ (fh,a) -> (unadh fh a, der (uncurry unadh) (fh,a)))                -- |uncurry| on functions
==  D (\ (fh,a) -> (unadh fh a, at a . der unadh fh !!! der (unadh fh) a))  -- \proofRef{deriv-uncurry}
==  D (\ (fh,a) -> (unadh fh a, at a . unadh !!! der (unadh fh) a))         -- |unadh| linearity
\end{code}
%if False
\begin{code}
==  D (\ (D h,a) -> (unadh (D h) a, at a . unadh !!! der (unadh (D h)) a))
==  D (\ (D h,a) -> let (b,f') = (unadh (D h) a,der (unadh (D h)) a) in(b, at a . unadh !!! f'))
==  D (\ (D h,a) -> let (b,f') = h a in (b, at a . unadh !!! f'))
\end{code}
%endif
% Now we are in a position to eliminate the noncomputable |der| operation.
Now note that
\begin{code}
    fh
==  adh (unadh fh)                   -- |adh . unadh == id|
==  D (unadh fh &&& der (unadh fh))  -- |adh| definition
\end{code}
Letting |D h = fh|, we have
\begin{code}
h a  == (unadh fh &&& der (unadh fh)) a
     == (unadh fh a, der (unadh fh) a)
\end{code}

A bit of refactoring then replaces |unadh fh a| and (the noncomputable) |der (unadh fh a)|, yielding a \emph{computable} form:
\begin{code}
    ado eval
==  ...
==  D (\ (D h,a) -> let (b,f') = h a in (b, at a . unadh !!! f'))
\end{code}

Since this calculation was fairly involved, let's get a sanity check on the types in the final form:
\begin{code}

(  D h,  a)  :: O ((a -> b) :* a)
             :: D (O a) (O b) :* O a
   D h       :: D (O a) (O b)
         a   :: O a
     h       :: O a -> O b :* (O a :-* O b)
     h a     :: O b :* (O a :-* O b)

(  b,  f')   :: O b :* (O a :-* O b)
   b         :: O b
       f'    :: O a :-* O b

                                             unadh      :: D (O a) (O b) :-* (O a -> O b)
                                at a                    :: (O a -> O b) :-* O b
                                at a .  unadh           :: D (O a) (O b) :-* O b
                                at a .  unadh !!! f'    :: D (O a) (O b) :* O a :-* O b
                           (b,  at a .  unadh !!! f')   :: O b :* (D (O a) (O b) :* O a :-* O b)
     \ (D h,a) -> ... in   (b,  at a .  unadh !!! f')   :: O ((a -> b) :* a) -> O b :* (D (O a) (O b) :* O a :-* O b)
D (  \ (D h,a) -> ... in   (b,  at a .  unadh !!! f'))  :: D (O ((a -> b) :* a)) (O b)

     eval  :: (a -> b) :* a -> b
ado  eval  :: D (O ((a -> b) :* a)) (O b)
\end{code}

\subsection{\lemRef{ado-curry}}\proofLabel{ado-curry}

Letting |fw = wrapO f|,
\begin{code}
    ado (curry f)
==  adh (wrapO (curry f))                                      -- |ado| definition
==  adh (adh . curry (wrapO f))                                -- \lemRef{wrapO-curry}
==  adh (adh . curry fw)                                       -- |fw| definition
==  D ((adh . curry fw) &&& der (adh . curry fw))              -- |adh| definition
==  D (\ a -> adh (curry fw a), der (adh . curry fw) a)        -- |(&&&)| definition
==  D (\ a -> adh (curry fw a), adh . der (curry fw) a)        -- chain rule; linearity of |adh|
==  D (\ a -> adh (curry fw a), adh . forkF (derl fw . (a,)))  -- \thmRef{deriv-cross}
\end{code}
Now, separately simplify the two main parts of this last form.
\begin{code}
    adh (curry fw a)
==  D (\ b -> (fw (a,b), derr fw (a,b)))       -- |adh| definition and \lemRef{deriv-pair-domain}
\end{code}
\begin{code}
    adh . forkF (derl fw . (a,))
==  adh . (\ da b -> (derl fw . (a,)) b da)                                         -- |forkF| definition
==  adh . (\ da b -> derl fw (a,b) da)                                              -- |(.)| on functions
==  \da -> adh (\ b -> derl fw (a,b) da)                                            -- |(.)| on functions
==  \da -> adh (\ b -> at da (derl fw (a,b)))                                       -- |at| definition
==  \da -> adh (at da . derl fw . (a,))                                             -- |(.)| on functions
==  \da -> D (\ b -> ((at da . derl fw . (a,)) b, der (at da . derl fw . (a,)) b))  -- |adh| definition 
==  \da -> D (\ b -> (derl fw (a,b) da, der (at da . derl fw . (a,)) b))            -- |(.)| on functions
\end{code}
Now simplify the remaining differentiated composition:
\begin{code}
    der (at da . derl fw . (a,)) b
==  at da . der (derl fw . (a,)) b            -- chain rule; linearity of |at da|
==  at da . derr (derl fw) (a,b)              -- \lemRef{deriv-pair-domain}
\end{code}
Putting the pieces back together,
\begin{code}
ado (curry f) ==
  D  (\ a ->  (D (\ b -> (fw (a,b), derr fw (a,b)))
     ,  \da -> D (\ b -> (derl fw (a,b) da, at da . derr (derl fw) (a,b)))))
\end{code}

\subsection{\lemRef{comp-bilinear}}\proofLabel{comp-bilinear}

To show that |comp = uncurry (.)| is bilinear, we can show that it is linear in each argument, which is to say |curry comp g = (g . NOP)| and |curry' comp f = (NOP . f)| are linear for all |g| and |f|.

First, |(NOP . f)| is linear for \emph{any} function |f| (not just linear):
\begin{code}
    (NOP . f) (g + g')
==  (g + g') . f                          -- left section definition
==  \ a -> (g + g') (f a)                 -- $\eta$ expansion
==  \ a -> g (f a) + g' (f a)             -- addition on functions
==  (\ a -> g (f a)) + (\ a -> g' (f a))  -- addition on functions
==  (g . f) + (g' . f)                    -- |(.)| on functions
==  (NOP . f) g + (NOP . f) g'            -- left section definition

    (NOP . f) (s *. g)
==  (s *. g) . f                          -- left section definition
==  \ a -> (s *. g) (f a)                 -- |(.)| on functions
==  \ a -> s *. g (f a)                   -- scaling on functions
==  s *. (\ a -> g (f a))                 -- scaling on functions
==  s *. (g . f)                          -- |(.)| on functions
==  s *. (NOP . f) g                      -- left section definition
\end{code}

Second, |(g . NOP)| is linear for any \emph{linear} functions |g|:
\begin{code}
    (g . NOP) (f + f')
==  g . (f + f')                          -- right section definition
==  \ a -> g ((f + f') a)                 -- $\eta$ expansion
==  \ a -> g (f a + f' a)                 -- addition on functions
==  \ a -> g (f a) + g (f' a)             -- linearity of |g|
==  (\ a -> g (f a)) + (\ a -> g (f' a))  -- addition on functions
==  (g . f) + (g . f')                    -- |(.)| on functions
==  (g . NOP) f + (g . NOP) f'            -- right section definition

    (g . NOP) (s *. f)
==  g . (s *. f)                          -- right section definition
==  \ a -> g ((s *. f) a)                 -- |(.)| on functions
==  \ a -> g (s *. f a)                   -- scaling on functions
==  \ a -> s *. g (f a)                   -- linearity of |g|
==  s *. (g . f)                          -- scaling on functions
==  s *. (g . NOP) f                      -- right section definition
\end{code}

\subsection{\lemRef{bilinear-props}}\proofLabel{bilinear-props}

Given any bilinear function |h|,

\begin{enumerate}

\item |curry h a| is linear for all linear functions |g|:
\begin{code}
    curry h a (b + b')
==  h (a,b + b')                -- |curry| on functions
==  h (a,b) + h (a,b')          -- bilinearity of |h|
==  curry h a b + curry h a b'  -- |curry| on functions

    curry h a (s *. b)
==  h (a,s *. b)                -- |curry| on functions
==  s *. h (a,b)                -- bilinearity of |h|
\end{code}

\item |curry' h b| is linear for all functions |b|:
Proof similar to |curry h a|.

\item |curry h| and |curry' h| are linear:
\begin{code}
    curry h (a + a')
==  \ b -> curry h (a + a') b             -- $\eta$ expansion
==  \ b -> h (a+a',b)                     -- |curry| on functions
==  \ b -> h (a,b) + h (a',b)             -- bilinearity of |h|
==  (\ b -> h (a,b)) + (\ b -> h (a',b))  -- addition on functions
==  curry h a + curry h a'                -- |curry| on functions

    curry h (s *. a)
==  \ b -> curry h (s *. a) b             -- $\eta$ expansion
==  \ b -> h (s *. a',b)                  -- |curry| on functions
==  \ b -> s *. h (a,b)                   -- bilinearity of |h|
==  s *. (\ b -> h (a,b))                 -- scaling on functions
==  s *. curry h a                        -- |curry| on functions
\end{code}

Similarly for |curry' h|.

\item |der h| is linear:
\begin{code}
    der h ((a,b) + (a',b'))
==  der h (a+a',b+b')                                         -- |(+)| on functions
==  h . (, b+b') !!! h . (a+a')                               -- \corRef{deriv-bilinear}
==  \ (da,db) -> h (da,b+b') + h (a+a',db)                    -- |(!!!)| on functions
==  \ (da,db) -> h (da,b) + h (da,b') + h (a,db) + h (a',db)  -- bilinearity of |h|
==  \ (da,db) -> h (da,b) + h (a,db) + h (da,b') + h (a',db)  -- commutativity of |(+)|
==  (\ (da,db) -> h (da,b   ) + h (a   ,db)) +
    (\ (da,db) -> h (da,b'  ) + h (a'  ,db))                  -- |(+)| on functions
==  der h (a,b) + der h (a',b')                               -- \corRef{deriv-bilinear}
\end{code}

Similarly for scaling.

\end{enumerate}

\subsection{\lemRef{deriv-pointfree}}\proofLabel{deriv-pointfree}

\begin{enumerate}

\item Sequential composition:
\begin{code}
    der (g . f)
==  \ a -> der (g . f) a                         -- $\eta$ expansion
==  \ a -> der g (f a) . der f a                 -- chain rule (\thmRef{deriv-compose})
==  \ a -> (der g . f) a . der f a               -- |(.)| on functions
==  \ a -> (.) ((der g . f) a) (der f a)         -- alternative notation
==  \ a -> uncurry (.) ((der g . f) a, der f a)  -- |uncurry| on functions
==  \ a -> comp ((der g . f) a, der f a)         -- |comp| definition
==  comp . (\ a -> ((der g . f) a, der f a))     -- |(.)| on functions
==  comp . (der g . f &&& der f)                 -- |(&&&)| definition
\end{code}

\item Cross:
\begin{code}
    der (f *** g)
==  \ (a,b) -> der (f *** g) (a,b)               -- $\eta$ expansion
==  \ (a,b) -> der f a *** der g b               -- cross rule (\thmRef{deriv-cross})
==  \ (a,b) -> uncurry (***) (der f a, der g b)  -- |uncurry| on functions
==  \ (a,b) -> cross (der f a, der g b)          -- |cross| definition
==  \ (a,b) -> cross ((der f *** der g) (a,b))   -- |(***)| on functions
==  cross . (der f *** der g)                    -- |(.)| on functions
\end{code}

\item Fork:
\begin{code}
    der (f &&& g)
==  der ((f *** g) . dup)                     -- cartesian law
==  \ a -> der ((f *** g) . dup) a            -- $\eta$ expansion
==  \ a -> der (f *** g) (dup a) . der dup a  -- chain rule (\thmRef{deriv-compose})
==  \ a -> der (f *** g) (a,a) . der dup a    -- |dup| for functions
==  \ a -> der f a *** der g a . der dup a    -- cross rule (\thmRef{deriv-cross})
==  \ a -> der f a *** der g a . dup          -- |dup| linearity
==  \ a -> der f a &&& der g a                -- cartesian law
==  fork . (der f &&& der g)                  -- |fork| definition
\end{code}

\item A linear function |f|,
\begin{code}
    der f
==  \ a -> der f a  -- $\eta$ expansion
==  \ a -> f        -- \thmRef{deriv-linear}
==  const f         -- |const| definition
\end{code}

\item Any function |f :: a :* b -> c|,
\begin{code}
    der f
==  \ (a,b) -> der f (a,b)                    -- $\eta$ expansion
==  \ (a,b) -> derl f (a,b) !!! derr f (a,b)  -- \lemRef{deriv-pair-domain}
==  join . (derl f &&& derr f)                -- |join| definition
\end{code}

\item A \emph{bilinear} function |f :: a :* b -> c|,
\begin{code}
    der f
==  \ (a,b) -> der f (a,b)                                 -- $\eta$ expansion
==  \ (a,b) -> f . (, b) !!! f . (a,)                      -- \corRef{deriv-bilinear}
==  \ (a,b) -> curry' f b !!! curry f a                    -- section definitions
==  \ (a,b) -> join (curry' f b, curry f a)                -- |join = uncurry (!!!)|
==  \ (a,b) -> join ((curry' f *** curry f) (b,a))         -- |(***)| on functions
==  \ (a,b) -> join ((curry' f *** curry f) (swap (a,b)))  -- |swap| on functions
==  join . (curry' f *** curry f) . swap                   -- |(.)| on functions
\end{code}

\item Uncurried composition on linear maps,
\begin{code}
    der comp
==  join . (curry' comp *** curry comp) . swap  -- previous (|comp| is bilinear)
==  join . (flip (.) *** (.)) . swap            -- |comp| definition
\end{code}

\end{enumerate}

\end{document}
