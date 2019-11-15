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

\nc\proofLabel[1]{\label{proof:#1}}
\nc\proofRef[1]{Appendix \ref{proof:#1}}

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

This general AD algorithm is justified by three main theorems:
\begin{quotation}
\vspace{-3ex}
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
|der (g . f) a == der g (f a) . der f a|.
\end{theorem}
\begin{theorem}[cross rule] \thmLabel{cross}
|der (f *** g) (a,b) == der f a *** der g b|.
\end{theorem}
\begin{theorem}[linear rule] \thmLabel{linear}
For all linear functions |f|, |der f a == f|.
\end{theorem}
\end{quotation}
\noindent
In addition to these three theorems, we need a collection of facts about the derivatives of various mathematical operations, e.g., |adh sin x = scale (cos x)|, where |scale :: a -> a :-* a| is uncurried scalar multiplication (so |scale s| is linear for all |s|).

\sectionl{Partial Derivatives}

In tackling the question of cartesian closure for AD, it will be helpful to develop a simple, categorical viewpoint of \emph{partial derivatives}, which serve as a useful tool for differentiating functions having non-scalar domains.

Suppose we have a function |f :: a :* b -> c|, and we want to compute its derivative at a point in its (pair-valued) domain.
Because linear maps (derivatives) form a cocartesian category,\footnote{The cocartesian law |h = h . inl ### h . inr| is dual to the cartesian law |h = exl . h &&& exr . h| \citep{Gibbons2002Calculating}.}
\begin{code}
der f (a,b) == der f (a,b) . inl ||| der f (a,b) . inr
\end{code}
The arguments to |(###)| here are the (``first'' and ``second'', or ``left'' and ``right'') ``partial derivatives'' of |f| at |(a,b)|.
Noting that |inl da = (da,0)| and |inr db = (0,db)|, we can see that the partial derivatives allow only one half of a pair to change.
It will be handy to give names to these arguments, as well as alternative forms, which follow from a bit of equational reasoning (\proofRef{partial-alt}).\footnote{The notation ``|(a,)|'' and ``|(b,)|'' refers to right and left ``sections'' of pairing: |(,b) a == (a,) b == (a,b)|.}
\begin{code}
derl :: (a :* b -> c) -> a :* b -> (a :-* c)
derl f (a,b)  = der f (a,b) . inl
              = der (f . (,b)) a

derr :: (a :* b -> c) -> a :* b -> (b :-* c)
derr f (a,b)  = der f (a,b) . inr
              = der (f . (a,)) b
\end{code}
Then
\begin{code}
der f (a,b) == derl f (a,b) ||| derr f (a,b)
\end{code}

\note{To do: Rewrite as a theorem with proof in the appendix.
Do I need the |inl|/|inr| version at all, or can I seclude it in the proof?}

As an example of how this decomposition of |der f| helps construct derivatives, suppose that |f| is \emph{bilinear}, which is to say that |f| is linear in each argument, while holding the other constant.
More formally |bilinearity| of |f| means that |f . (a,)| and |f . (b,)| are both linear for all |a :: a| and |b :: b|.
Therefore,
\begin{code}
    der f (a,b)
==  derl f (a,b) ||| derr f (a,b)          -- above
==  der (f . (,b)) a ||| der (f . (a,)) b  -- above
==  f . (,b) ||| f . (a,)                  -- linearity
\end{code}
For instance, the derivative of uncurried multiplication is given by the Leibniz product rule:
\begin{code}
    der (uncurry (*)) (a,b)
==  uncurry (*) . (,b) ||| uncurry (*) . (a,)
==  (NOP * b) ||| (a * NOP)
== \ (da,db) -> da * b + a * db
\end{code}
% which is sometimes written ``|d (u v) = u dv + v du|''.


\sectionl{Higher-Order Codomains}

It will also be useful to calculate derivatives of functions with higher-order codomains.\notefoot{The previous section and this one provide ``adjoint'' techniques in a sense that they currying is an adjunction from functions from products to functions to functions.
Is there something else interesting to say here?}
We'll need another linear map operation, which is reverse function application:\footnote{Linearity of |applyTo a| follows from the usual definition of addition and scaling on functions.}
\begin{code}
applyTo :: a -> (a -> b) :-* b
applyTo a df = df a
\end{code}
\begin{theorem}\thmLabel{higher-order-codomain}
Given a function |g :: a -> b -> c|,
$$|der g a = \ da b -> der (applyTo b . g) a da|.$$
\end{theorem}
Proof:\notefoot{Maybe move this proof to the appendix.}\footnote{As a sanity check, the RHS (``|\ da b -> ...|'') is indeed linear, because |der (applyTo b . g) a| linear is (being a derivative) and noting the usual interpretation of scaling and addition on functions.}
\begin{code}
    \ da b -> der (applyTo b . g) a da
==  \ da b -> (der (applyTo b) (g a) . der g a) da  -- chain rule
==  \ da b -> (applyTo b . der g a) da              -- |applyTo b| is linear
==  \ da b -> applyTo b (der g a da)                -- |(.)| on functions
==  \ da b -> der g a da b                          -- |applyTo| definition
==  der g a                                         -- $\eta$ reduction (twice)
\end{code}


\sectionl{Cartesian Closed?}

While |D| is a category and a \emph{cartesian} category at that, as specified by |adh| being a cartesian functor, another question naturally arises.
Can |adh| also be a cartesian \emph{closed} functor?
In other words, are there definitions of |eval|, |curry|, and |uncurry| on |D| such that
\begin{code}
curry (adh f) = adh (curry f)
uncurry (adh g) = adh (uncurry g)
eval = adh eval
\end{code}

As usual, we'll want to solve each homomorphism equation for its single unknown, which is a categorical operation on |D| (on the LHS).
Start with |curry|, simplifying the LHS:
\begin{code}
    curry (adh f)
==  curry (D (f &&& der f))
\end{code}
Then the RHS:
\begin{code}
    adh (curry f)
==  D (curry f &&& der (curry f))
\end{code}
where (on functions) |curry f = \ a b -> f (a,b)|.

\workingHere \note{Postpone the class/interface discussion to \secref{Object Mapping}}. \vspace{5ex}

These three operations come  with every \emph{cartesian closed} category:
\begin{code}
class Cartesian k => CartesianClosed k where
  type ExpOp k :: Type -> Type -> Type
  eval     :: (Prod k ((Exp k a b)) a) `k` b
  curry    :: ((Prod k a b) `k` c) -> (a `k` (Exp k b c))
  uncurry  :: (a `k` (Exp k b c)) -> ((Prod k a b) `k` c)
\end{code}
where |Exp k a b| is a type of ``exponential objects'' (first class functions/arrows) from |a| to |b| for the category |k|.
These operations support higher-order programming and arise during translation from a typed lambda calculus (e.g., Haskell) to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}.

Similarly, monoidal and cartesian categories have category-associated categorical \emph{products}:
%format MonoidalP = Monoidal
%format MonoidalC = Monoidal'
\begin{code}
class Category k => MonoidalP k where
  type ProdOp k :: Type -> Type -> Type
  (***) :: (a `k` c) -> (b `k` d) -> ((Prod k a b) `k` (Prod k c d))  -- product bifunctor

class Monoidal k => Cartesian k where
  exl  :: (Prod k a b) `k` a
  exr  :: (Prod k a b) `k` b
  dup  :: a `k` (Prod k a a)
\end{code}
A particularly important related operation:
\begin{code}
(&&&) :: Cartesian k => (a `k` c) -> (a `k` d) -> (a `k` (Prod k c d))
f &&& g = (f *** g) . dup
\end{code}
For functions and linear maps, the categorical product is the usual cartesian product, and product operations defined as follows:\footnote{These method definitions are written as if linear maps were represented by functions that happen to be linear. Other representations will be useful as well, with method definitions specified again via simple, regular homomorphism equations \citep{Elliott-2018-ad-icfp}.}
\begin{code}
(f *** g) (a,b) = (f a, g b)
exl  (a,b) = a
exr  (a,b) = b
dup a = (a,a)
\end{code}
Hence
\begin{code}
    (f &&& g) a
==  ((f *** g) . dup) a
==  (f *** g) (dup a)
==  (f *** g) (a,a)
==  (f a, g a)
\end{code}

Dually, we have monoidal and cocartesian categories with associated categorical ``coproducts'':
\begin{code}
class Category k => MonoidalC k where
  type CoprodOp k :: Type -> Type -> Type
  (+++) :: (a `k` c) -> (b `k` d) -> ((Coprod k a b) `k` (Coprod k c d))  -- coproduct bifunctor

class MonoidalC k => Cocartesian k where
  inl  :: a `k` (Coprod k a b)
  inr  :: b `k` (Coprod k a b)
  jam  :: (Coprod k a a) `k` a

(|||) :: Cocartesian k => (c `k` a) -> (d `k` a) -> ((Coprod k c d) `k` a)
f ||| g = jam . (f +++ g)
\end{code}
In this paper we will be working in the setting of \emph{biproducts}, where products and coproducts coincide.\footnote{More precisely, linear maps (in all representations) form a biproduct category, but we will not use coproducts with functions or (computably) differentiable functions.
Coproducts are useful in defining a simple, dualized linear map category that yields reverse mode AD when used with the single, general AD algorithm.}
The corresponding bifunctor operations |(:+)| and |(:*)| thus also coincide:
\begin{code}
class MonoidalP k => Cocartesian k where
  inl  :: a `k` (Prod k a b)
  inr  :: b `k` (Prod k a b)
  jam  :: (Prod k a a) `k` a

(|||) :: (c `k` a) -> (d `k` a) -> ((Prod k c d) `k` a)
f ||| g = jam . (f *** g)
\end{code}
For the category of linear maps and vector spaces (or semimodules) over a specified scalar type,
\begin{code}
inl  a = (a,0)
inr  b = (0,b)
jam (a,a') = a + a'
\end{code}
from which it follows that
\begin{code}
    (f ||| g) (c,d)
==  jam ((f *** g) (c,d))
==  jam (f c, c d)
==  f c + g d
\end{code}

Just as the |Category| and |Cartesian| instances for |D| arose from solving corresponding homomorphism equations about |adh|, let's now try the same with |CartesianClosed|.
First note that we do not really have to define all three methods, since |eval| and |uncurry| can each be defined in terms of the other:\out{\footnote{The pattern |g *** id| is also called ``|first g|'', because it applies |g| to the first element of a pair while leaving the second element unchanged.}}
\begin{code}
eval = uncurry id
uncurry g  = eval . (g *** id)
           = eval . first g
\end{code}
Since |eval| looks simpler, start there.
The corresponding homomorphism equation has a particularly simple form:
\begin{code}
eval = adh eval
\end{code}
It might appear that we are done at the start, taking the equation to be a definition for |eval|.
Recall, however, that |adh| is noncomputable, being defined via |der| (differentiation itself).
Let us press forward undeterred, opening up the definition of |adh| to see if we can transform away the (noncomputable) |der|:
\begin{code}
    adh eval
==  D (eval &&& der eval)                        -- definition of |adh|
==  D (\ (f,a) -> (eval (f,a), der eval (f,a))   -- |(&&&) on functions|
==  D (\ (f,a) -> (f a, der eval (f,a))          -- |eval| on functions
\end{code}
Now we do not need the general |der|, but rather the specific |der eval|.
If |eval| were linear, we could apply \thmRef{linear}, but alas it is not.
No matter, as we can instead use the technique of partial derivatives.
\begin{code}
    der eval (f,a)
==  derl eval (f,a) ||| derr eval (f,a)          -- method of partial derivatives
==  der (eval . (,a)) f ||| der (eval . (f,)) a  -- |derl| and |derr| alternative definitions
==  der (applyTo a) f ||| der f          a       -- |eval| on functions
==  applyTo a ||| der f a                        -- linearity of |applyTo a|
==  \ (df,dx) -> df a + der f a dx               -- |(###) on linear maps|; |applyTo| definition
\end{code}
Now we can complete the calculation of |eval| for |D|:
\begin{code}
    eval
==  adh eval
==  D (eval &&& der eval)                       -- definition of |adh|
==  D (\ (f,a) -> (eval (f,a), der eval (f,a))  -- |(&&&) on functions|
==  D (\ (f,a) -> (f a, der eval (f,a))         -- |eval| on functions
==  D (\ (f,a) -> (f a, applyTo a ||| der f a)  -- above
\end{code}
Although this final form is well-defined, it uses the noncomputable |der| and so is not a computable recipe, leaving us in a pickle.
Let's look for some wiggle room.

\sectionl{Object Mapping}

The choice of category-associated products and exponentials is a degree of freedom not exercised in the development of AD in \cite{Elliott-2018-ad-icfp} and one that is tied closely to another such choice available in the general notion of \emph{cartesian closed functor} in category theory.
In general, a functor has two aspects:
\begin{itemize}
\item a mapping from arrows to arrows, and
\item a mapping from objects to objects.
\end{itemize}
The functor |adh| defined (noncomputably) above implicitly chooses an \emph{identity object mapping}, as evident in its type signature |adh :: (a -> b) -> D a b|.\out{\notefoot{Rewrite this section more clearly.}}

Recall the types of |eval| and |adh|:
\begin{code}
eval :: CartesianClosed k => (Prod k ((Exp k a b)) a) `k` b
\end{code}
This type of |adh| plus the requirement that it be a cartesian \emph{closed} functor implies that the object mapping aspect of |adh| is the identity, and in particular |Exp D u v = u -> v|.
It is this final conclusion that puts us in the pickle noted above, namely the need to compute the noncomputable.
We can, however, make this impossible task trivial by building the needed derivative into |Exp D u v|, say by choosing |Exp D u v = D u v|.
In this case, we must alter |adh| as well.
Letting |O| be the object mapping aspect of the new functor |ado|,\notefoot{Experiment with different notation for |O a|, e.g., ``$\bar{a}$''.}
\begin{code}
ado :: (a -> b) -> D (O a) (O b)
\end{code}
The property of being a closed cartesian functor requires |O| to preserve categorical products and exponentials, i.e.,
\begin{code}
O (a  :*  b) == Prod D  (O a)  (O b)
O (a  ->  b) == Exp  D  (O a)  (O b)
\end{code}
To make |eval| on |D| computable, choose |Exp D u v = D u v| as mentioned above.
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
The new functor |ado| converts its given |a -> b| to |O a -> O b| and then applies the |adh| functor:
%format wrapO = wrap"_o"
%format wrapO = wrap
%format unwrapO = "\inv{"wrapO"}"
\begin{code}
wrapO :: (a -> b) -> (O a -> O b)
wrapO f = toO . f . unO

unwrapO :: (O a -> O b) -> (a -> b)
unwrapO h = unO . h . toO

ado :: (a -> b) -> D (O a) (O b)
ado = adh . wrapO

unado :: D (O a) (O b) -> (a -> b)
unado = unwrapO . unadh
\end{code}
A bit of equational reasoning shows that |wrapO| and |unwrapO| form an isomorphism (\proofRef{wrapO-iso}), as do |ado| and |unado| (\proofRef{ado-iso}).

The cartesian category operations already defined |D| \citep{Elliott-2018-ad-icfp} are solutions to homomorphism equations saying that |adh| is a cartesian functor.
Thanks to the simple, regular structure of |toO| and |unO|, |ado| is a cartesian functor as well (\proofRef{ado-cartesian}).

% \sectionl{A cartesian closed category of computably differentiable functions}
\sectionl{Cartesian Closed}

What about exponentials and cartesian \emph{closure}?
As mentioned above, we'll choose |Exp D u v = D u v|.
Requiring |ado| to be a cartesian closed \emph{functor} necessitates that |O (a -> b) == Exp D (O a) (O b) == D (O a) (O b)|, which suggests using |ado| and |unado| for |toO| and |unO|:
\begin{code}
instance (HasO a, HasO b) => HasO (a -> b) where
  type O (a -> b) = D (O a) (O b)
  toO  = ado
  unO  = unado
\end{code}
For a |CartesianClosed| instance, we'll need to define |curry| and |eval|.
We don't have definitions from \cite{Elliott-2018-ad-icfp} to imitate and verify this time, so we have to discover new ones by solving the homomorphism equations.
For |eval|, the homomorphism equation is |eval == ado eval|, which is already a definition but not a computable one (since |ado| involves |adh|, which involves |der|, i.e., differentiation).
The choice of |toO = ado| enables a computable solution (\proofRef{ado-eval}):
\begin{code}
eval = D (\ (D h,a) -> let (b,f') = h a in (b, applyTo a . unadh ||| f'))
\end{code}

Next consider currying.
The homomorphism equation is |curry (ado f) == ado (curry f)|, to be solved for the unknown LHS |curry| (on |D|).

\note{Maybe introduce |curry| earlier before |eval|.
I think it's computable with |adh|, though |eval| isn't.
I can get some of the preparatory work done.}


\sectionl{Related Work}

\note{Working here.}

\begin{itemize}
\item \cite{Vytiniotis-2019-differentiable-curry}
\item \cite{Brunel2019Backprop}
\end{itemize}

\bibliography{bib}

\appendix

\sectionl{Proofs}

\subsection{Partial Derivatives}\proofLabel{partial-alt}

\begin{code}
    der (f . (,b)) a
==  der f ((,b) a) . der (,b) a                      -- chain rule
==  der f (a,b) . der (,b) a                         -- |(,b)| definition
==  der f (a,b) . der (inl + const (0,b)) a          -- |inl| on functions, and meaning of |(,b)|
==  der f (a,b) . (der inl a + der (const (0,b)) a)  -- linearity of |(+)|
==  der f (a,b) . der inl a                          -- |der (const z) a == 0|
==  der f (a,b) . inl                                -- linearity of |inl|; \thmRef{linear}
\end{code}
Likewise for |der (f . (a,)) b|.

\subsection{Differentiation and Uncurrying}\proofLabel{der-uncurry}

\begin{code}
    der (uncurry h) (a,b)
==  der (eval . first h) (a,b)                         -- CCC law (above)
==  der eval (first h (a,b)) . der (first h) (a,b)     -- chain rule
==  der eval (h a,b) . der (first h) (a,b)             -- |first| for functions
==  (applyTo b ||| der (h a) b) . der (first h) (a,b)  -- |der eval| (above)
==  (applyTo b ||| der (h a) b) . first (der h a)      -- below
==  (applyTo b ||| der (h a) b) . (der h a *** id)     -- |first| definition
==  applyTo b . der h a ||| der (h a) b                -- |(f ### g) . (p *** q) == f . p ### g . q|

    der (first h) (a,b)
==  der (h *** id) (a,b)                               -- |first| definition
==  der h a *** der id b                               -- \thmRef{cross}
==  der h a *** id                                     -- |id| is linear
==  first (der h a)                                    -- |first| definition
\end{code}

\subsection{|wrapO| as Isomorphism}\proofLabel{wrapO-iso}

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

\subsection{|ado| as Isomorphism}\proofLabel{ado-iso}

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

\subsection{|ado| as Cartesian Functor}\proofLabel{ado-cartesian}

Given the cartesian category operations already defined on |D|, |ado| is also a cartesian functor \citep{Elliott-2018-ad-icfp}.
The proofs mainly exploit the regular structure of |toO| and |unO|:

\begin{code}
    ado id
==  adh (wrapO id)                                   -- |ado| definition
==  adh (toO . id . unO)                             -- |wrapO| definition
==  adh (toO . unO)                                  -- |id| is identity for |(.)|
==  adh id                                           -- |toO . unO == id|
==  id                                               -- |adh| is funtor

    ado (g . f)
==  adh (wrapO (g . f))                              -- |ado| definition
==  adh (toO . g . f . unO)                          -- |wrapO| definition
==  adh (toO . g . unO . toO . f . unO)              -- |unO . toO == id|
==  adh (toO . g . unO) . adh (toO . f . unO)        -- |adh| is a functor
==  adh (wrapO g) . adh (wrapO f)                    -- |wrapO| definition
==  ado g . ado f                                    -- |ado| definition

    ado (f *** g)
==  adh (wrapO (f *** g))                            -- |ado| definition
==  adh (toO . (f *** g) . unO)                      -- |wrapO| definition
==  adh ((toO *** toO) . (f *** g) . (unO *** unO))  -- |toO| on products
==  adh (toO . f . unO *** toO . g . unO)            -- monoidal category law
==  adh (toO . f . unO) *** adh (toO . g . unO)      -- |adh| is a monoidal functor
==  adh (wrapO f) *** adh (wrapO g)                  -- |wrapO| definition
==  ado f *** ado g                                  -- |ado| definition

    ado exl
==  adh (wrapO exl)                                  -- |ado| definition
==  adh (toO . exl . unO)                            -- |wrapO| definition
==  adh (toO . exl . (unO *** unO))                  -- |unO| on products
==  adh (toO . unO . exl)                            -- |exl . (f *** g) == f . exl| for cartesian categories
==  adh exl                                          -- |toO . unO == id|
==  exl                                              -- |adh| is a cartesian functor                          

    ado exr
==  adh (wrapO exr)                                  -- |ado| definition
==  adh (toO . exr . unO)                            -- |wrapO| definition
==  adh (toO . exr . (unO *** unO))                  -- |unO| on products
==  adh (toO . unO . exr)                            -- |exr . (f *** g) == g . exr| for cartesian categories
==  adh exr                                          -- |toO . unO == id|
==  exr                                              -- |adh| is a cartesian functor

    ado dup
==  adh (wrapO dup)                                  -- |ado| definition
==  adh (toO . dup . unO)                            -- |wrapO| definition
==  adh (toO . (unO *** unO) . dup)                  -- |dup . f == (f *** f) . dup| for cartesian categories
==  adh (toO . unO . dup)                            -- |unO| on products
==  adh dup                                          -- |toO . unO == id|
==  dup                                              -- |adh| is a cartesian functor
\end{code}

\subsection{|ado| and |eval|}\proofLabel{ado-eval}

The homomorphism equation is |eval == ado eval|.
Simplifying the RHS,
\begin{code}
    ado eval
==  adh (wrapO eval)                                                             -- |ado| definition
==  adh (toO . eval . unO)                                                       -- |wrapO| definition
==  adh (toO . eval . (unado *** unO))                                           -- |unO| on |(a -> b) :* a|
==  adh (\ (fh,a) -> (toO . eval . (unado *** unO)) (fh,a))                      -- |eta| expansion
==  adh (\ (fh,a) -> toO (eval (unado fh, unO a)))                               -- |(.)| and |(***)| on functions
==  adh (\ (fh,a) -> toO (unado fh (unO a)))                                     -- |eval| on functions
==  adh (\ (fh,a) -> toO (unwrapO (unadh fh) (unO a)))                           -- |unado| definition
==  adh (\ (fh,a) -> toO ((unO . unadh fh . toO) (unO a))                        -- |unwrapO| definition
==  adh (\ (fh,a) -> toO (unO (unadh fh (toO (unO a)))))                         -- |(.)| on functions
==  adh (\ (fh,a) -> unadh fh a)                                                 -- |toO . unO == id|
==  adh (uncurry unadh)                                                          -- |uncurry| on functions
==  D (\ (fh,a) -> (uncurry unadh (fh,a), der (uncurry unadh) (fh,a)))           -- |adh| definition
==  D (\ (fh,a) -> (unadh fh a, der (uncurry unadh) (fh,a)))                     -- |uncurry| on functions
==  D (\ (fh,a) -> (unadh fh a, applyTo a . der unadh fh ||| der (unadh fh) a))  -- \proofRef{der-uncurry}
==  D (\ (fh,a) -> (unadh fh a, applyTo a . unadh ||| der (unadh fh) a))         -- |unadh| is linear
\end{code}
%if False
\begin{code}
==  D (\ (D h,a) -> (unadh (D h) a, applyTo a . unadh ||| der (unadh (D h)) a))
==  D (\ (D h,a) -> let (b,f') = (unadh (D h) a,der (unadh (D h)) a) in(b, applyTo a . unadh ||| f'))
==  D (\ (D h,a) -> let (b,f') = h a in (b, applyTo a . unadh ||| f'))
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
==  D (\ (D h,a) -> let (b,f') = h a in (b, applyTo a . unadh ||| f'))
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

                                             unadh           :: D (O a) (O b) :-* (O a -> O b)
                                applyTo a                    :: (O a -> O b) :-* O b
                                applyTo a .  unadh           :: D (O a) (O b) :-* O b
                                applyTo a .  unadh ||| f'    :: D (O a) (O b) :* O a :-* O b
                           (b,  applyTo a .  unadh ||| f')   :: O b :* (D (O a) (O b) :* O a :-* O b)
     \ (D h,a) -> ... in   (b,  applyTo a .  unadh ||| f')   :: O ((a -> b) :* a) -> O b :* (D (O a) (O b) :* O a :-* O b)
D (  \ (D h,a) -> ... in   (b,  applyTo a .  unadh ||| f'))  :: D (O ((a -> b) :* a)) (O b)

     eval  :: (a -> b) :* a -> b
ado  eval  :: D (O ((a -> b) :* a)) (O b)

\end{code}

\end{document}
