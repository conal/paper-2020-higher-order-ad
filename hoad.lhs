% -*- latex -*-

%% While editing/previewing, use 12pt or 14pt and tiny margin.
\documentclass[12pt,twoside]{extarticle}  % fleqn,14pt
\usepackage[margin=0.2in]{geometry}  % 0.12in, 0.9in

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

\usepackage[round]{natbib}  % square
\bibliographystyle{plainnat}

\title{\tit \emph{\\[1.5ex] \Large (early draft---comments invited\notefoot{Add GitHub link here for latest version, and welcome issues and even pull requests.})}}

\date{\today}

\setlength{\blanklineskip}{2ex} % blank lines in code environment

\nc\proofRef[1]{Appendix \ref{proof:#1}}
\nc\provedIn[1]{\textnormal{Proved in \proofRef{#1}}}
\nc\proofLabel[1]{\label{proof:#1}}

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

Begin with the category of computably differentiable functions from \cite[Section 4.1]{Elliott-2018-ad-icfp}:
\begin{code}
newtype D a b = D (a -> b :* (a :-* b))
\end{code}
where |a :-* b| is the type of linear maps from |a| to |b|.
The function around which the automatic differentiation (AD) algorithm is organized simply ``zips'' together a function |f :: a -> b| and its derivative |der f :: a -> a :-* b|:\footnote{This paper generally uses Haskell notation but deviates slightly by using a single colon rather than double colon for type signatures. \note{Experimental.}}\footnote{The infix operators for function types (``|->|'') and linear maps (``|:-*|'') both associate to the right and have equal, very low precedence.
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
\vspace{-3ex}
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

\subsectionl{Linearity and invertibility}

%if False
As is well-known,
\begin{theorem}
Differentiation itself (i.e., |der|) is linear.
\end{theorem}

\note{|adh| as well. Move inversion and |unadh| here also. Also |fork| and |unfork|}
%endif

\subsectionl{Pair-Valued Domains}

One half of the |curry|/|uncurry| isomorphism involves functions of pair-valued domains.
The notion of partial derivatives is helpful for differentiating such functions.\notefoot{I'm leaning toward eliminating |derl| and |derr| in favor of their meanings.
Whenever I use the names below, I then immediate inline them.}\footnote{Recall that, on linear maps, |(f !!! g) (a,b) = f a + g b|, |inl a = (a,0)|, and |inr b = (0,b)|}
\begin{theorem}[\provedIn{thm:deriv-pair-domain}]\thmLabel{deriv-pair-domain}
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
\end{theorem}
Note also that |f . (a,) = curry f a| and |f . (,b) = curry' f b|, where
\begin{code}
curry   f a b = f (a,b)
curry'  f b a = f (a,b)
\end{code}

As an example of how this decomposition of |der f| helps construct derivatives, suppose that |f| is \emph{bilinear}, which is to say that |f| is linear in each argument, while holding the other constant.
More formally |bilinearity| of |f| means that |f . (a,)| and |f . (b,)| are both linear for all |a| and |b|.
\begin{corollary}\corLabel{deriv-bilinear}
If |f :: a :* b -> c| is bilinear then $$
|der f (a,b) == f . (,b) !!! f . (a,)|
$$
\end{corollary}
\begin{proof}~
\begin{code}
    der f (a,b)
==  derl f (a,b) !!! derr f (a,b)          -- \thmRef{deriv-pair-domain}
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
\begin{corollary}[\provedIn{cor:deriv-uncurry}]\corLabel{deriv-uncurry}
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
eval f a = f a  -- on functions
\end{code}
(Note that |eval| is neither linear nor bilinear, so \thmRef{deriv-linear} and \corRef{deriv-bilinear} are both inapplicable.)
We'll need one more linear map operation, which is curried, reverse function application:\footnote{Linearity of |at a| follows from the usual definition of addition and scaling on functions.}
\begin{code}
at :: a -> (a -> b) :-* b
at a df = df a
\end{code}
\begin{corollary}[\provedIn{cor:deriv-eval}] \corLabel{deriv-eval}
$$ |der eval (f,a) == at a !!! der f a| $$
\end{corollary}

\subsectionl{Function-Valued Codomains}

It will also be useful to calculate derivatives of functions with higher-order codomains.\notefoot{The previous section and this one provide ``adjoint'' techniques in a sense that currying is an adjunction from functions from products to functions to functions.
Is there something else interesting to say here?}
We'll need anoter linear map operation, which is the indexed variant of |(&&&)|:
\begin{code}
forkF :: (b -> a :-* c) -> (a :-* b -> c)
forkF h = \ da b -> h b da
\end{code}
\begin{theorem}[\provedIn{thm:deriv-function-codomain}]\thmLabel{deriv-function-codomain}
Given a function |g :: a -> b -> c|,
$$|der g a = forkF (\ b -> der (at b . g) a)|.$$
\end{theorem}

%% Curried functions differentiate as follows:
\begin{corollary}[\provedIn{cor:deriv-curry}]\corLabel{deriv-curry}
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
==  D (\ a -> ((\ b -> f (a,b)), forkF (derl f . (a,))))             -- |curry| and |(a,)|; \corRef{deriv-curry}
==  D (\ a -> ((\ b -> f (a,b)), forkF (\ b -> derl f (a,b))))       -- |(.)| on functions
==  D (\ a -> ((\ b -> f (a,b)), forkF (\ b -> der f (a,b) . inl)))  -- \proofRef{thm:deriv-pair-domain}
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
\begin{lemma}
The pair of functions |fork| and |unfork| form a linear isomorphism.
\end{lemma}
Proof: Exercise.

Another such linear isomorphism can be found in cocartesian categories.
The following types are specialized to biproduct categories (such as linear maps):
\begin{code}
join :: Cocartesian k => (a `k` c) :* (a `k` d) -> (a `k` (c :* d))
join = uncurry (!!!)

unjoin :: Cocartesian k => (a `k` (c :* d)) -> (a `k` c) :* (a `k` d)
unjoin h = (h . inl, h . inr)
\end{code}
\begin{lemma}
The pair of functions |join| and |unjoin| form a linear isomorphism.
\end{lemma}
Proof: Exercise.

These two isomorphism pairs were used by \cite{Elliott-2018-ad-icfp} to construct a correct-by-construction implementation of reverse-mode AD, by merely altering the representation of linear maps used in the simple, general AD algorithm.

Although |fork|/|unfork| form an isomorphism and hence preserve information, |unfork| can result in a loss of efficiency, due to computation that can be (and often is) shared between a function |f| and its derivative |der f|.
Indeed, the definition of |unfork h| above shows that |h| gets replicated.
It's unclear how to avoid this redundancy problem in practice with currying when |D| is used to represent computably differentiable functions.
My own experience with compiling to categories \cite{Elliott-2017-compiling-to-categories} suggests that most uses of |curry| generated during translation from the $\lambda$ calculus (e.g., Haskell) are in fact transformed away at compile time using various equational CCC laws.
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
==  D (uncurry g &&& der (uncurry g))                       -- |adh| definition
==  D (\ (a,b) -> (uncurry g (a,b), der (uncurry g) (a,b))  -- |(&&&)| definition
==  D (\ (a,b) -> (g a b, der (uncurry g) (a,b))            -- |uncurry| on functions
==  D (\ (a,b) -> (g a b, at b . der g a !!! der (g a) b))  -- \corRef{deriv-uncurry}
\end{code}
Now we have a problem with solving the defining homomorphism above .
Although we can extract |g| and |der g| from |adh g|, we cannot extract |der (g a)|.
Or rather we can, but not computably.

\subsectionl{Eval}

We don't need to work out both |uncurry| and |eval|, since each can be defined in terms of the other:\out{\footnote{The pattern |g *** id| is also called ``|first g|'', because it applies |g| to the first element of a pair while leaving the second element unchanged.}}
\begin{code}
eval = uncurry id
uncurry g  = eval . (g *** id)
           = eval . first g
\end{code}
Since we got stuck on |uncurry|, let's try |eval| as well to see if we learn anything new.

The corresponding homomorphism equation has a particularly simple form:
\begin{code}
eval = adh eval
\end{code}
It might appear that we are done at the start, taking the equation to be a definition for |eval|.
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
Let's look for some wiggle room.


\sectionl{Object Mapping}

The choice of category-associated products and exponentials is a degree of freedom not exercised in the development of AD in \cite{Elliott-2018-ad-icfp} (or above) and is tied closely to another such choice available in the general notion of \emph{cartesian closed functor} in category theory.
In general, a functor has two aspects:
\begin{itemize}
\item a mapping from arrows to arrows, and
\item a mapping from objects to objects.
\end{itemize}
The functor |adh| defined (noncomputably) above implicitly chooses an \emph{identity object mapping}, as evident in its type signature |adh :: (a -> b) -> D a b|.
The type of |adh| plus the requirement that it be a cartesian \emph{closed} functor implies that the object mapping aspect of |adh| is the identity.
More generally, we can define an object mapping |O :: Type -> Type| for a new functor |ado|:\notefoot{Experiment with different notation for |O a|, e.g., ``$\bar{a}$''.}
\begin{code}
ado :: (a -> b) -> D (O a) (O b)
\end{code}
Each cartesian category |k| has its own notion of categorical product |Prod k a b| (satisfying a universality property), and similarly for cocartesian categories (with categorical products and coproducts coinciding for biproduct categories).
Likewise, each cartesian \emph{closed} category |k| has its own notion of \emph{exponential} objects |Exp k a b|.

The generalized interface for cartesian closed categories with per-category exponentials is as follows:\footnote{These operations support higher-order programming and arise during translation from a typed lambda calculus (e.g., Haskell) to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}.}
\begin{code}
class Cartesian k => CartesianClosed k where
  infixrQ 1 (ExpOp k)
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

Consider again the homomorphic specification for |curry| (part of the CCF definition): |eval = ado eval|.
The RHS |eval| (on functions) has type |(a -> b) :* a -> b|, while the RHS |eval| (on |D|) has type
\begin{code}
    D (O ((a -> b) :* a)) (O b)
==  D (O (a -> b) :* O a) (O b)
==  D ((Exp D (O a) (O b)) :* O a) (O b)
\end{code}
The difficulty with our attempt at |eval| in \secref{Eval} was that we were given a function |f|, but we also needed its derivative |der f|.
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
We will need to convert between |a| and |O a| , which we can do with a family of \emph{isomorphisms}\footnote{An implicit requirement for all |HasO| instances is thus that |toO . unO == id| and |unO . toO == id|.} indexed by |a|:\notefoot{It may be more elegant to combine the functions |toO| and |unO| into a single \emph{isomorphism}.}
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
\begin{theorem}[\provedIn{wrapO-iso}]\thmLabel{wrapO-iso}
|wrapO| and |unwrapO| form a linear isomorphism.
\end{theorem}
\begin{theorem}[\provedIn{ado-iso}]\thmLabel{ado-iso}
|ado| and |unado| form a linear isomorphism.
\end{theorem}
\begin{theorem}[\provedIn{wrapO-cartesian}]\thmLabel{wrapO-cartesian}
|wrapO| is a cartesian functor.
\end{theorem}

\note{To do: reconsider theorems vs lemmas vs corollaries. I think more lemmas.}

The cartesian category operations already defined on |D| \citep{Elliott-2018-ad-icfp} are solutions to homomorphism equations saying that |adh| is a cartesian functor.
Thanks to the simple, regular structure of |toO| and |unO|, |ado| is a cartesian functor as well:
\begin{theorem}\thmLabel{ado-cartesian}
|ado| is a cartesian functor.
\end{theorem}
Proof: |ado| is a cartesian functor \citep{Elliott-2018-ad-icfp}, as is |wrapO| (\thmRef{wrapO-cartesian}).

What about exponentials and cartesian \emph{closure}?
As mentioned above, |O (a -> b) == Exp D (O a) (O b) == D (O a) (O b)|, which suggests using |ado| and |unado| for |toO| and |unO|:
\begin{code}
instance (HasO a, HasO b) => HasO (a -> b) where
  type O (a -> b) = D (O a) (O b)
  toO  = ado
  unO  = unado
\end{code}

A useful consequence:
\begin{theorem}[\provedIn{wrapO-curry}]\thmLabel{wrapO-curry}
$$|wrapO (curry f) == adh . curry (wrapO f)|$$
\end{theorem}
\begin{corollary}\corLabel{curry-wrapO}
$$|curry (wrapO f) == unadh . wrapO (curry f)|$$
\end{corollary}
\begin{proof}
Left-compose |unadh| with both sides of \thmRef{wrapO-curry}; then simplify and reverse the resulting equation.
\end{proof}

Let's now try to solve the CCF equations for |ado|.
This time begin with |eval|:
\begin{theorem}[\provedIn{thm:ado-eval}] \thmLabel{ado-eval}
With the following (effective) definition of |eval| on |D|, |eval == ado eval|:
\begin{code}
eval = D (\ (D h,a) -> let (b,f') = h a in (b, at a . unadh !!! f'))
\end{code}
\end{theorem}
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
\begin{code}
    ado (curry f)
==  adh (wrapO (curry f))                                      -- |ado| definition
==  adh (adh . curry (wrapO f))                                -- \thmRef{wrapO-curry}
==  adh (adh . curry fw)                                       -- |fw| definition
==  D ((adh . curry fw) &&& der (adh . curry fw))              -- |adh| definition
==  D (\ a -> adh (curry fw a), der (adh . curry fw) a)        -- |(&&&)| definition
==  D (\ a -> adh (curry fw a), adh . der (curry fw) a)        -- chain rule; linearity of |adh|
==  D (\ a -> adh (curry fw a), adh . forkF (derl fw . (a,)))  -- \corRef{deriv-curry}
\end{code}

Now, separately simplify the two main parts of this last form.
\begin{code}
    adh (curry fw a)
==  D (\ b -> (fw (a,b), derr fw (a,b)))       -- |adh| definition and \thmRef{deriv-pair-domain}
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
==  at da . derr (derl fw) (a,b)              -- \thmRef{deriv-pair-domain}
\end{code}
Putting the pieces back together, we get a simplified specification for |curry| on |D|:
\begin{code}
    curry (D (fw &&& der fw))
==  D  (\ a ->  (D (\ b -> (fw (a,b), derr fw (a,b)))
                ,  \da -> D (\ b -> (derl fw (a,b) da, at da . derr (derl fw) (a,b)))))
\end{code}
The RHS uses |fw (a,b)| and |der fw (a,b)| (via its components |derl fw (a,b)| and |derr fw (a,b)|), but it also uses a \emph{second} partial derivative |derr (derl fw) (a,b)|, which is not available from the |curry| argument |D (fw &&& der fw)|.

\sectionl{Where Are We?}

Let's now reflect on what we've learned so far:
\begin{itemize}

\item The cartesian functor (CF) |adh :: (a -> b) -> D a b| also forms a cartesian \emph{closed} functor (CCF) with suitable definitions of |curry|, |uncurry|, and |eval|, but not computably (\secref{Cartesian Closure, first attempt}).
More specifically, |curry| is computable, but |uncurry| and |eval| are not, since they need to synthesize derivatives of regular computable functions.

\item General categorical functors can remap objects (here, types) as well as morphisms (here, functions).
Exploiting this degree of freedom, define |ado :: (a -> b) -> D (O a) (O b)|, where |O :: Type -> Type| that replaces regular functions with computably differentiable functions, i.e., |O (u -> v) = D (O u) (O v)|.
This new function is defined in terms of the old one, |ado = adh . wrapO|, and indeed |ado| is a CF as well.
In the absence of higher-order functions, |O| is the identity mapping, and |ado| coincides with |adh|.

\item Computably satisfying the required homomorphism properties of |ado| for |uncurry| and |eval| becomes easy, since the operations are \emph{given} the required derivatives rather than having to synthesize them.
Unfortunately, now |curry| becomes noncomputable because it has to synthesize partial \emph{second} derivatives.

\end{itemize}



\sectionl{Related Work}

\note{Working here.}

\begin{itemize}
\item \cite{Vytiniotis-2019-differentiable-curry}
\item \cite{Brunel2019Backprop}
\end{itemize}

\bibliography{bib}

\appendix

\sectionl{Proofs}

\subsection{\thmRef{deriv-pair-domain}}\proofLabel{thm:deriv-pair-domain}

Suppose we have a function |f :: a :* b -> c|, and we want to compute its derivative at a point in its (pair-valued) domain.
Because linear maps (derivatives) form a cocartesian category,\footnote{The cocartesian law |h = h . inl !!! h . inr| is dual to the cartesian law |h = exl . h &&& exr . h| \citep{Gibbons2002Calculating}.}
\begin{code}
der f (a,b) == der f (a,b) . inl !!! der f (a,b) . inr
\end{code}
Noting that |inl da = (da,0)| and |inr db = (0,db)|, we can see that the partial derivatives allow only one half of a pair to change.

Next, note that |der f (a,b) . inl = der (f . (,b)) a|, by the following equational reasoning:
\begin{code}
    der (f . (,b)) a
==  der f ((,b) a) . der (,b) a                      -- chain rule
==  der f (a,b) . der (,b) a                         -- |(,b)| definition
==  der f (a,b) . der (inl + const (0,b)) a          -- |inl| on functions, and meaning of |(,b)|
==  der f (a,b) . (der inl a + der (const (0,b)) a)  -- linearity of |(+)|
==  der f (a,b) . der inl a                          -- |der (const z) a == 0|
==  der f (a,b) . inl                                -- linearity of |inl|; \thmRef{deriv-linear}
\end{code}
Likewise, |der f (a,b) . inr = der (f . (a,)) b|.

\subsection{\corRef{deriv-uncurry}}\proofLabel{cor:deriv-uncurry}

\begin{code}
    der (uncurry g) (a,b)
==  derl (uncurry g) (a,b) !!! derr (uncurry g) (a,b)      -- \thmRef{deriv-pair-domain}
==  der (uncurry g . (,b)) a !!! der (uncurry g . (a,)) b  -- |derl| and |derr| definitions
==  der (\ a' -> uncurry g (a',b)) a !!!                   -- $\eta$ expand and simplification
    der (\ b' -> uncurry g (a,b')) b
==  der (\ a' -> g a' b) a !!! der (\ b' -> g a b') b      -- |uncurry| on functions
==  der (at b . g) a !!! der (g a) b                       -- |at| definition and $\eta$ reduction
==  der (at b) (g a) . der g a !!! der (g a) b             -- chain rule
==  at b . der g a !!! der (g a) b                         -- linearity of |at|
\end{code}

\subsection{\corRef{deriv-eval}}\proofLabel{cor:deriv-eval}

\begin{code}
    der eval (f,a)
==  derl eval (f,a) !!! derr eval (f,a)          -- method of partial derivatives
==  der (eval . (,a)) f !!! der (eval . (f,)) a  -- |derl| and |derr| alternative definitions
==  der (at a) f !!! der f          a            -- |eval| on functions
==  at a !!! der f a                             -- linearity of |at a|
==  \ (df,dx) -> df a + der f a dx               -- |(!!!) on linear maps|; |at| definition
\end{code}

Alternatively, calculate |der eval| via |uncurry|:
\begin{code}
    der eval (f,a)
==  der (uncurry id) (f,a)            -- |eval = uncurry id|
==  at a . der id a !!! der (id f) a  -- \corRef{deriv-uncurry}
==  at a . id !!! der f a             -- |id| is linear
==  at a !!! der f a                  -- |id| as identity
\end{code}

\subsection{\thmRef{deriv-function-codomain}}\proofLabel{thm:deriv-function-codomain}

\begin{code}
    forkF (\ b -> der (at b . g) a)
==  \ da b -> der (at b . g) a da              -- |forkF| definition
==  \ da b -> (der (at b) (g a) . der g a) da  -- chain rule
==  \ da b -> (at b . der g a) da              -- |at b| is linear
==  \ da b -> at b (der g a da)                -- |(.)| on functions
==  \ da b -> der g a da b                     -- |at| definition
==  der g a                                    -- $\eta$ reduction (twice)
\end{code}

\subsection{\corRef{deriv-curry}}\proofLabel{cor:deriv-curry}

%% The proof is a simple application of \thmRef{deriv-function-codomain}:
\begin{code}
    der (curry f) a
==  forkF (\ b -> der (at b . curry f)) a           -- \thmRef{deriv-function-codomain}
==  forkF (\ b -> der (\ a -> at b (curry f a))) a  -- |(.)| on functions
==  forkF (\ b -> der (\ a -> curry f a b)) a       -- |at| definition
==  forkF (\ b -> der (\ a -> f (a,b))) a           -- |curry| on functions
==  forkF (\ b -> der (f . (,b))) a                 -- |(, b)| definition
==  forkF (\ b -> derl f (a,b))                     -- |derl| definition
==  forkF (derl f . (a,))                           -- |(a,)| definition
\end{code}


\subsection{\thmRef{wrapO-iso}}\proofLabel{wrapO-iso}

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

\subsection{\thmRef{ado-iso}}\proofLabel{ado-iso}

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

Linearity of |ado| and |unado| follows from linearity of |adh| and |unadh| and \thmRef{wrapO-iso}.

\subsection{\thmRef{wrapO-cartesian}}\proofLabel{wrapO-cartesian}

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

\subsection{\thmRef{wrapO-curry}}\proofLabel{wrapO-curry}

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

\subsection{|ado| and |eval|}\proofLabel{thm:ado-eval}

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
==  D (\ (fh,a) -> (unadh fh a, at a . der unadh fh !!! der (unadh fh) a))  -- \proofRef{cor:deriv-uncurry}
==  D (\ (fh,a) -> (unadh fh a, at a . unadh !!! der (unadh fh) a))         -- |unadh| is linear
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

\end{document}
