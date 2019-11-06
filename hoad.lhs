% -*- latex -*-

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12pt,twoside]{article}  % fleqn,
\usepackage[margin=0.12in]{geometry}  % 0.12in, 0.9in

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
adh :: (a -> b) -> D a b
adh f  = D (\ a -> (f a, der f a))
       = D (f &&& der f)
\end{code}
Note that this definition is not computable, since |der| is not \citep{PourEl1978Diff, PourEl1983Comp}.
The whole specification of AD is then simply that |adh| is a homomorphism with respect to a standard compositional vocabulary of functions, namely that of cartesian categories, plus a collection of numeric primitives like (uncurried) addition and multiplication, |sin| and |cos|, etc.
An example of such an equation is |adh g . adh f == adh (g . f)|, in which the only unknown is the meaning of the LHS |(.)|, i.e., sequential composition in the category |D|.
Solving the collection of such homomorphism equations yields correct-by-construction AD.

%format unadh = "\inv{"adh"}"
The function |adh| is invertible, i.e., |unadh . adh == id|, where |unadh| simply drops the derivative:
\begin{code}
unadh :: D a b -> (a -> b)
unadh (D h) = exl . h
\end{code}
Indeed, |unadh| is a left inverse of |adh|:
\begin{code}
    unadh (adh f)
==  unadh (D (f &&& der f))  -- |adh| definition
==  exl . (f &&& der f)      -- |unadh| definition
==  f                        -- |exl . (g &&& h) == g| in cartesian categories
\end{code}

AD is often described as coming in forward and backward ``modes''.
For many practical applications (including deep learning and other high-dimensional optimization problems), reverse mode is much more efficient than forward mode.
As typically presented, reverse mode is also much more complicated, but this difference appears to be due only to unfortunate choices in how to think about and implement AD.
Instead, a single, simple algorithm works for both reverse as well as forward (and other) modes, with the only difference being that reverse mode uses a different linear map representation resulting from a simple classic trick \citep{Elliott-2018-ad-icfp}.

This general AD algorithm is justified by three main theorems:
\begin{quotation}
\vspace{-3ex}
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
$|der (g . f) a == der g (f a) . der f a|$.
\end{theorem}
\begin{theorem}[cross rule] \thmLabel{cross}
$|der (f *** g) (a,b) == der f a *** der g b|$.
\end{theorem}
\begin{theorem}[linear rule] \thmLabel{linear}
For all linear functions |f|, |der f a == f|.
\end{theorem}
\end{quotation}
\noindent
In addition to these three theorems, we need a collection of facts about the derivatives of various mathematical operations, e.g., |adh sin x = scale (cos x)|, where |scale :: a -> a :-* a| is uncurried scalar multiplication (so |scale s| is linear for all |s|).

\sectionl{Cartesian closed?}

While |D| is a category and a \emph{cartesian} category at that, as specified by |adh| being a cartesian functor, another question naturally arises.
Can |adh| also a \emph{closed} cartesian functor?
In other words, are there definitions of |eval|, |curry|, and |uncurry| on |D| such that
\begin{code}
eval = adh eval
curry (adh f) = adh (curry f)
uncurry (adh g) = adh (uncurry f)
\end{code}
These three operations come from the following interface:
\begin{code}
class Cartesian k => CartesianClosed k where
  type ExpOp k :: Type -> Type -> Type
  eval :: (Prod k ((Exp k a b)) a) `k` b
  curry :: ((Prod k a b) `k` c) -> (a `k` (Exp k b c))
  uncurry :: (a `k` (Exp k b c)) -> ((Prod k a b) `k` c)
\end{code}
where |Exp k a b| is a type of ``exponential objects'' (first class functions/arrows) from |a| to |b| for the category |k|.
These operations support higher-order programming and arise during translation from a typed lambda calculus (e.g., Haskell) to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}.

Similarly, monoidal and cartesian categories have category-associated categorical \emph{products}:
\begin{code}
class Category k => MonoidalP k where
  type ProdOp k :: Type -> Type -> Type
  (***) :: (a `k` c) -> (b `k` d) -> ((Prod k a b) `k` (Prod k c d))

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
For functions and linear maps, the categorical product is the usual cartesian product and
\begin{code}
(f *** g) (a,b) = (f a, g b)
exl (a,b) = a
exr (a,b) = b
dup a = (a,a)
\end{code}
Hence
\begin{code}
    (f &&& g) a
==  (f *** g) (dup a)
==  (f *** g) (a,a)
==  (f a, g a)
\end{code}

Dually, we have monoidal and cocartesian categories with associated categories ``coproducts'':
\begin{code}
class Category k => MonoidalC k where
  type CoprodOp k :: Type -> Type -> Type
  (+++) :: (a `k` c) -> (b `k` d) -> ((Coprod k a b) `k` (Coprod k c d))

class Cocartesian k where
  inl :: a `k` (Coprod k a b)
  inr :: b `k` (Coprod k a b)
  jam :: (Coprod k a a) `k` a

(|||) :: (c `k` a) -> (d `k` a) -> ((Coprod k c d) `k` a)
f ||| g = jam . (f +++ g)
\end{code}
In this paper we will be working in the setting of \emph{biproducts}, where products and coproducts coincide, hence
\begin{code}
class Cocartesian k where
  inl :: a `k` (Prod k a b)
  inr :: b `k` (Prod k a b)
  jam :: (Prod k a a) `k` a

(|||) :: (c `k` a) -> (d `k` a) -> ((Prod k c d) `k` a)
f ||| g = jam . (f *** g)
\end{code}
For functions over additive monoids (including vector spaces and semimodules) and linear maps,
\begin{code}
inl a = (a,0)
inr b = (0,b)
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
First note that we do not really have to define all three methods, since |eval| and |uncurry| can each be defined in terms of the other:\footnote{The pattern |g *** id| is also called ``|first g|'', because it applies |g| to the first element of a pair while leaving the second element unchanged.}
\begin{code}
eval = uncurry id
uncurry g = eval . (g *** id)
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
==  der (f . (,b)) a ||| der (f . (a,)) b  -- \mynote{To do: justify}
==  derl f (a,b) ||| derr f (a,b)
\end{code}
where by convenient definition, |derl| and |derr| denote ``partial derivatives'' in which one half of a pair-valued argument is allowed to vary while the other half is held constant, i.e.,\footnote{The Haskell notation ``|(a,)|'' and ``|(b,)|'' refers to right and left ``sections'' of pairing: |(,b) a == (a,) b == (a,b)|.}
\begin{code}
derl :: (a :* b -> c) -> a :* b -> (a :-* c)
derl f (a,b) = der (f . (,b)) a

derr :: (a :* b -> c) -> a :* b -> (b :-* c)
derr f (a,b) = der (f . (a,)) b
\end{code}
Apply the technique of partial derivatives to |eval|.
%format ## = "\mathbin{\$}"
%if False
First, define reverse function application:
\begin{code}
applyTo :: a -> (a -> b) -> b
applyTo a f = f a
\end{code}
%else
Using ``|(##)|'' for infix function application,
%endif
\begin{code}
    der eval (f,a)
==  derl eval (f,a) ||| derr eval (f,a)    -- method of partial derivatives
==  der (## NOP a) f ||| der (f NOP ##) a  -- |derl| and |derr| definitions; |eval| on functions
==  der (## NOP a) f ||| der f a           -- |(f NOP ##) == f|
==  (## NOP a) ||| der f a                 -- \thmRef{linear}, since |(## NOP a) is linear|
==  \ (df,dx) -> df a + der f a dx         -- |(###) on linear maps|
\end{code}

Now we can complete the calculation of |eval| for |D|:
\begin{code}
    eval
==  adh eval
==  D (eval &&& der eval)                        -- definition of |adh|
==  D (\ (f,a) -> (eval (f,a), der eval (f,a))   -- |(&&&) on functions|
==  D (\ (f,a) -> (f a, der eval (f,a))          -- |eval| on functions
==  D (\ (f,a) -> (f a, (## NOP a) ||| der f a)  -- above
\end{code}
Although this final form is well-defined, it is not a computable recipe, since |der| is not computable, which leaves us in a pickle.
The problem specification inherited from \cite{Elliott-2018-ad-icfp} leaves us in a pickle.
Let's look for some wiggle room.

\sectionl{Object mapping}

The choice of category-associated products and exponentials is a degree of freedom not exercised in the development of AD in \cite{Elliott-2018-ad-icfp} and one that is tied closely to another such choice available in the general notion of \emph{cartesian closed functor} in category theory.
In general, a functor has two aspects:
\begin{itemize}
\item a mapping from arrows to arrows, and
\item a mapping from objects to objects.
\end{itemize}
The functor |adh| defined (noncomputably) above implicitly chooses an \emph{identity object mapping}, as evident in its type |(a -> b) -> D a b|.

Recall the types of |eval| and |adh|:
\begin{code}
eval :: CartesianClosed k => (Prod k ((Exp k a b)) a) `k` b
\end{code}
This type of |adh| plus the requirement that it be a cartesian \emph{closed} functor implies that the object mapping aspect of |adh| be the identity, and in particular |Exp D u v = u -> v|.
It is this final conclusion that puts us in the pickle noted above, namely the need to compute the noncomputable.
We can make this impossible task trivial by building the needed derivative into |Exp D u v|, say by choosing |Exp D u v = D u v|.
In this case, we must alter |adh| so as not to require an identity object mapping.
Letting |O| be the object mapping aspect of the new functor |ado|,
\begin{code}
ado :: (a -> b) -> D (O a) (O b)
\end{code}
%format unado = "\inv{"ado"}"
The property of being a closed cartesian functor requires |O| to preserve categorical products and exponentials, i.e.,
\begin{code}
O (a  :*  b) == Prod D  (O a)  (O b)
O (a  ->  b) == Exp  D  (O a)  (O b)
\end{code}
To make |eval| on |D| computable, we can choose |Exp D u v = D u v| as mentioned above.
Additionally map scalars to themselves and cartesian products to cartesian products:
\begin{code}
O R == R
O (a  :*  b) == Prod D  (O a)  (O b)  == O a :* O b
O (a  ->  b) == Exp  D  (O a)  (O b)  == D (O a) (O b)
\end{code}
%format toO = "\Varid{o}"
%format unO = "\inv{\Varid{o}}"
We will need to convert between |a| and |O a| , which we can do with a family of isomorphisms indexed by |a|:\notefoot{It may be more elegant to combine the functions |toO| and |unO| into a single \emph{isomorphism}.}
\begin{code}
class HasO (k :: * -> * -> *) t where
  type O k t
  toO  :: t -> O k t
  unO  :: O k t -> t
\end{code}
Moreover, |toO| and |unO| will form an isomorphism.
For some types |a|, |O a == a|, so the isomorphism is trivial:
\begin{code}
instance HasO k R where
  type O k R = R
  toO  = id
  unO  = id

instance HasO k () where
  type O k () = ()
  toO  = id
  unO  = id
\end{code}
For products, convert components independently:
\begin{code}
instance (HasO k a, HasO k b) => HasO k (a :* b) where
  type O k (a :* b) = O k a :* O k b
  toO  = toO  ***  toO
  unO  = unO  ***  unO
\end{code}
The new functor |ado| converts its given |a -> b| to |O a -> O b| and then applies the |adh| functor.
\begin{code}
ado :: (a -> b) -> D (O a) (O b)
ado f  = adh (toO . f . unO)
       = let g = toO . f . unO in D (\ a -> (g a, der g a))
       = let g = toO . f . unO in D (g &&& der g)

unado :: D (O a) (O b) -> (a -> b)
unado h = unO . unadh h . toO
\end{code}
Note that indeed |unado . ado == id|:
\begin{code}
    unado (ado f)
==  unado (adh (toO . f . unO))              -- |ado| definition
==  unO . unadh (adh (toO . f . unO)) . toO  -- |unado| definition
==  unO . (toO . f . unO) . toO              -- |unadh . adh == id|
==  (unO . toO) . f . (unO . toO)            -- associativity of |(.)|
==  id . f . id                              -- |unO . toO == id|
==  f                                        -- |id| is the left \& right identity for |(.)|
\end{code}
Use |ado| for exponentials in |D|:
\begin{code}
instance (HasO k a, HasO k b) => HasO k (a -> b) where
  type O k (a -> b) = D (O a) (O b)
  toO  = ado
  unO  = unado
\end{code}

\sectionl{Related Work}

\bibliography{bib}

\end{document}

