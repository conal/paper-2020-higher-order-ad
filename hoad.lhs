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
\fancyhead[LO]{Calculating compilers categorically}
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

\title{Calculating compilers categorically \\ \emph{\large (early draft---comments invited)}}
%% Cheap and cheerful stack machines
%% Compiling to a stack machine easily and correctly
%% Calculating compilers categorically
%% Calculating compilers cheaply and cheerfully
%% Calculating compilers cheaply, cheerfully, and categorically
%% Calculating a cheap and cheerful compiler
%% Cheap and cheerful correct compilation

%% \date{\today}

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

This note revisits the classic exercise of compiling a programming language to a stack-based virtual machine.
The main innovation is to factor the exercise into two phases: translation into standard algebraic vocabulary, and a stack-oriented interpretation of that vocabulary.
The first phase is independent of stack machines and has already been justified and implemented in a much more general setting.
The second phase captures the essential nature of stack-based computation, is independent of the source language, and is calculated from a very simple specification.

The first translation phase converts a typed functional language (here, Haskell) to the vocabulary of categories \citep{Elliott-2017-compiling-to-categories}.
All that remains is to specify and calculate a category of stack computations, which is quite easily done as demonstrated below.
Other examples of this compiling-to-categories technique include generation of massively parallel implementations on GPUs and FPGAs, incremental evaluation, interval analysis, and automatic differentiation \citep{Elliott-2017-compiling-to-categories,Elliott-2018-ad-icfp}.

\end{abstract}

\sectionl{Stack functions}

A stack machine for functional computation is like a mathematical function |f :: a -> b|, but it can also use additional storage to help compute |f|, as long as it does so in a stack discipline.\footnote{This paper uses ``stack machine'' to refers to data stacks, not control stacks.}
A simple formalization of this informal description is that the machine computes |first f|, where\footnote{In this paper, |(:*)| (cartesian product) has higher syntactic precedence than |(->)| (functions), so the type of |first| is equivalent to |(a -> b) -> forall z. ((a :* z) -> (b :* z))|.}
\begin{code}
first :: (a -> b) -> forall z. (a :* z -> b :* z)
first f (a,z) = (f a, z)
\end{code}
We are representing the stack as a pair, with |a| on top at the start of the computation, |f a| on top at the end of the computation, and |z| as the rest of the stack at the start and finish.
In-between the start and end, the stack may grow and shrink, but in the end the \emph{only} stack change is on top.
Note also that |first f| can do nothing with |z| other than preserve it.\notefoot{Find and cite some reasonably clear descriptions of this stack discipline.}

The purpose of a stack in language implementation is as a place to save intermediate results until they are ready to be consumed, after a given sub-computation completes.
For instance, suppose we want to apply the function |\ x -> (x + 2) * (x - 3)|.
Assuming right-to-left evaluation, a stack machine would evaluate |x - 3|, leaving the result |v| on the stack, then |x + 2|, leaving its result |u| on the stack above |v|, and then replace the top two stack elements |u| and |v| with |u * v|.

Let's now further formalize this notion of stack computation as a data type of ``stack functions'', having a simple relationship with regular functions:%
\notefoot{Is there a free theorem saying that any function of type |forall z. a :* z -> b :* z| \emph{must} be equivalent to |first f| for some |f :: a -> b|?
If so, then |stackFun| is an isomorphism, which may be useful.}
\begin{code}
newtype StackFun a b = SF (forall z. a :* z -> b :* z)

stackFun :: (a -> b) -> StackFun a b
stackFun f = SF (first f)
\end{code}
Conversely, we can evaluate a stack function into a regular function, initializing the stack to contain |a| and |()|\out{ (i.e., |a| on top and no information below)}, evaluating the contained stack operations, and discarding the final |()|:\footnote{The |()| type contains only a single value (other than |undefined|), which is also called ``|()|''. As such, it takes no space to represent.}
\begin{code}
evalStackFun :: StackFun a b -> (a -> b)
evalStackFun (SF f) a = b where (b,()) = f (a,())
\end{code}
We can also formulate |evalStackFun| in more general terms:
\begin{code}
evalStackFun (SF f) = rcounit . f . runit
\end{code}
The new operations belong to a categorical interface:\\
\begin{minipage}[b]{0.31\textwidth}
\begin{code}
class UnitCat k where
  lunit    :: a `k` (() :* a)
  lcounit  :: (() :* a) `k` a
  runit    :: a `k` (a :* ())
  rcounit  :: (a :* ()) `k` a
\end{code}
\end{minipage}
\codesep{0.9in}
\begin{minipage}[b]{0.3\textwidth} \mathindent2em
\begin{code}
instance UnitCat (->) where
  lunit a = ((),a)
  lcounit ((),a) = a
  runit a = (a,())
  rcounit (a,()) = a
\end{code}
\end{minipage}
%
\begin{lemma}[\provedIn{evalStackFun-as-left-inverse}] \lemLabel{evalStackFun-as-left-inverse}
|evalStackFun| is a left inverse for |stackFun|, i.e., |evalStackFun . stackFun == id|.
\end{lemma}

\begin{lemma}[\provedIn{stackFun-surjective}] \lemLabel{stackFun-surjective}
|stackFun| is surjective, i.e., \emph{every} |h :: StackFun a b| has the form |SF (first f)| for some |f :: a -> b|.
\end{lemma}

\begin{lemma}[\provedIn{stackFun-injective}] \lemLabel{stackFun-injective}
|stackFun| is injective, i.e., \emph{every} |stackFun f == stackFun f' ==> f == f'| for all |f, f' :: a -> b|.
\end{lemma}

\begin{corollary} \lemLabel{evalStackFun-as-inverse}
|evalStackFun| is the full (\emph{left} and right) inverse for |stackFun|.
\end{corollary}
\begin{proof}
Since |stackFun| is surjective and injective, it has a full (two-sided) inverse, which is necessarily unique.
Moreover, whenever a category morphism has both left and right inverses, those inverses must be equal \citep[Lemma 2.1]{ncatlab-retract}.
\end{proof}

\out{\mynote{I'd like to show that |evalStackFun| is a homomorphism w.r.t all of the same classes as its right inverse |stackFun|. How? Does the left inverse property suffice? I think it suffices to show that |stackFun| is surjective or that |evalStackFun| is a full inverse. The value of a a homomorphic |evalStackFun| is that then we need only show that |progFun| is as well, in order to establish that |evalProg = evalStackFun . progFun| is as well.}}

The definition of |stackFun| above serves as a simple \emph{specification}\out{, while the actual implementation will use an explicit data stack}.
Instead of starting with a \emph{function} |f| as suggested by |stackFun|, we will start with a recipe for |f| and systematically construct an analogous recipe for |stackFun f|.
Specifically, start with a formulation of |f| in the vocabulary of categories  \citep{MacLane1998categories,Lawvere:2009:Conceptual,Awodey2006CT}, and require that |stackFun| preserves the algebraic structure of that vocabulary.
While inconvenient to program in this vocabulary directly, we can instead automatically convert from Haskell programs \citep{Elliott-2017-compiling-to-categories}.
This approach to calculating correct implementations has also been used for automatic differentiation \citep{Elliott-2018-ad-icfp}.
A benefit is that we need only implement a few type class instances rather than manipulate any syntactic representation.

\subsectionl{Sequential composition}

The first requirement is that |stackFun| preserve the structure of |Category|, which is to say that it a category homomorphism (also called a ``functor'').
The |Category| interface:
\begin{code}
class Category k where
  id   :: a `k` a
  (.)  :: (b `k` c) -> (a `k` b) -> (a `k` c)
\end{code}
The corresponding structure preservation (homomorphism) properties:
\begin{code}
id = stackFun id
stackFun g . stackFun f = stackFun (g . f)
\end{code}
The identity and composition operations on the LHS are for |StackFun|, while the ones on the right are for |(->)| (i.e., regular functions).
Solving these equations for the LHS operations results in a correct instance of |Category| for |StackFun|.

The |id| equation is trivial to satisfy, since it is already in solved form, so we can use it directly as an implementation.
Instead, simplify the equation as follows:
\begin{code}
   stackFun id
=  {- definition of |stackFun| -}
   SF (first id)
=  {- property of |first| and |id| -}
   id = SF id
\end{code}

The |(.)| equation requires a little more work.
First simplify the LHS:
\begin{code}
   stackFun g . stackFun f
=  {- definition of |stackFun| -}
   SF (first g) . SF (first f)
\end{code}
Then the RHS:
\begin{code}
   stackFun (g . f)
=  {- definition of |stackFun| -}
   SF (first (g . f))
=  {- property of |first| and |(.)| -}
   SF (first g . first f)
\end{code}
The simplified specification:
\begin{code}
SF (first g) . SF (first f) == SF (first g . first f)
\end{code}
Strengthen this equation by generalizing from |first g| and |first f| to arbitrary functions (also called ``|g|'' and ``|f|'' and having the same types as |first g| and |first f|):
\begin{code}
SF g . SF f == SF (g . f)
\end{code}
This generalized/strengthened condition is in solved form, so we can satisfy it simply by definition, yielding sufficient definitions for both category operations:
\begin{code}
instance Category StackFun where
  id = SF id
  SF g . SF f = SF (g . f)
\end{code}
In words, the identity stack function is the identity function on stacks, and the composition of stack functions is the composition of functions on stacks.

%format lassocP = lassoc
%format rassocP = rassoc
%format swapP = swap
%format AssociativePCat = AssociativeCat
%format BraidedPCat = BraidedCat
Two other categorical classes can be trivially handled in the same manner as |id| above:\notefoot{Maybe drop these two.}
\begin{code}
class AssociativePCat k where
  rassocP :: ((a :* b) :* c) `k` (a :* (b :* c))
  lassocP :: (a :* (b :* c)) `k` ((a :* b) :* c)

class BraidedPCat k where
  swapP :: (a :*  b) `k` (b :* a)
\end{code}
The associated homomorphism equations are in solved form and can serve as definitions:
\begin{code}
instance AssociativePCat StackFun where
  rassocP = stackFun rassocP
  lassocP = stackFun lassocP

instance BraidedPCat StackFun where
  swapP = stackFun swapP
\end{code}


\subsectionl{Parallel composition (products)}

%format MonoidalPCat = MonoidalP
In general, the purpose of a stack is to sequentialize computations.
Since we've only considered sequential composition so far, we've done nothing interesting with the stack.
Nonsequential computation comes from parallel composition, as embodied in the ``cross'' operation in the |MonoidalPCat| interface:
\begin{code}
class MonoidalPCat k where
  (***) :: (a `k` c) -> (b `k` d) -> ((a :* b) `k` (c :* d))
\end{code}
There are two special forms that are sometimes more convenient (one of which we've already seen in a more specialized context):
\begin{code}
first :: MonoidalPCat k => (a `k` c) -> ((a :* b) `k` (c :* b))
first f = f *** id

second :: MonoidalPCat k => (b `k` d) -> ((a :* b) `k` (a :* d))
second g = id *** g
\end{code}
The following law holds for all monoidal categories \citep[Section 1.5.1]{Gibbons2002:Calculating}:
\begin{code}
(f *** g) . (p *** q) == (f . p) *** (g . q)
\end{code}
Taking |g = id| and |p = id|, and renaming |q| to ``|g|'', we get
\begin{code}
first f . second g == f *** g
\end{code}
Similarly,
\begin{code}
second g . first f == f *** g
\end{code}
We can also define |second| in terms of |first| (or vice versa):\notefoot{What would it take to prove this claim in general?}
\begin{code}
second g = swap . first g . swap
\end{code}
Thanks to these relationships, any two of |(***)|, |first|, and |second| can be defined in terms of the other.
For our purpose, it will be convenient to calculate a definition of |first| on |StackFun|, and then define |(***)| as follows:
\begin{code}
f *** g  = first f . second g
         = first f . swap . first g . swap
\end{code}
We thus need only define |first|, which we can do by solving the corresponding homomorphism property, i.e.,
\begin{code}
first (stackFun f) == stackFun (first f)
\end{code}
Equivalently (filling in the definition of |stackFun|),
\begin{code}
first (SF (first f)) == SF (first (first f))
\end{code}
What do we do with |first (first f)|?\notefoot{Also noted by \citet[Section 1.1]{Paterson03arrowsand} and by \citet[Definition 2]{Alimarine06thereand}. \mynote{Is there a category theory reference for this property in, say, monoidal categories?}}
Let's examine the types involved:
\begin{code}
                f   :: a -> c
         first  f   :: a :* b -> c :* b
first (  first  f)  :: forall z. (a :* b) :* z -> (c :* b) :* z
\end{code}
To reshape this computation into a stack function, temporarily move |b| aside by re-associating:
\begin{code}
   first (first f)
=  {- definition of |first| on |(->)| -}
   \ ((a,b),z) -> ((f a,b),z)
=  {- definition of |lassocP|, |rassocP|, and |first| on |(->)| -}
   lassocP . first f . rassocP
\end{code}
Our required homomorphism equation for |first| is thus equivalent to the following:\footnote{%
It may be tempting to invoke the definition of of |(.)| on |StackFun|, and rewrite the RHS to |SF lassocP . SF (first f) . SF rassoc|. Exercise: what goes wrong?}
\begin{code}
first (SF (first f)) == SF (lassocP . first f . rassoc)
\end{code}
Generalizing from |first f|, we get the following sufficient condition:
\begin{code}
first (SF f) == SF (lassocP . f . rassoc)
\end{code}
Since this generalized equation is in solved form, we can use it as a definition, expressing |second| and |(***)| in terms of it:
\begin{code}
instance MonoidalPCat StackFun where
  first (SF f) = SF (lassocP . f . rassocP)
  second g = swap . first g . swap
  f *** g = first f . second g
\end{code}
This sequentialized computation corresponds to right-to-left evaluation of arguments.
We can get left-to-right evaluation by reformulating parallel composition as |f *** g = first f . second g|.

To understand the operational implications of this |MonoidalPCat| instance, let's see how parallel composition unfolds on a stack machine:

\begin{code}
    stackFun f *** stackFun g
==  {- definition of |stackFun| -}
    SF (first f) *** SF (first g)
==  {- definition of |(***)| on |StackFun| -}
    first (SF (first f)) . second (SF (first g))
==  {- definition of |second| on |StackFun| -}
    first (SF (first f)) . swap . first (SF (first g)) . swap
==  {- definitions of |first| and |swap| on |StackFun| -}
    SF (lassocP . first f . rassocP) . stackFun swap . SF (lassocP . first g . rassocP) . stackFun swap
==  {- definition of |stackFun| -}
    SF (lassocP . first f . rassocP) . SF (first swap) . SF (lassocP . first g . rassocP) . SF (first swap)
==  {- definition of |(.)| on |StackFun| -}
    SF (lassocP . first f . rassocP . first swap . lassocP . first g . rassocP . first swap)
\end{code}
%if False
The first (right-most) stack-manipulation sequence:
\begin{code}
    rassocP . first swap
    \ ((a,b),z) -> (rassocP . first swap) ((a,b),z)
==  {- $\eta$-expansion -}
    \ ((a,b),z) -> rassocP (first swap ((a,b),z))
==  {- definition of |(.)| on |(->)| -}
    \ ((a,b),z) -> rassocP (swap (a,b),z)
==  {- definition of |swap| on |(->)| -}
    \ ((a,b),z) -> rassocP ((b,a),z)
==  {- definition of |rassocP| -}
    \ ((a,b),z) -> (b,(a,z))
\end{code}
Likewise, the second (left-most) stack-manipulation sequence:
\begin{code}
rassocP . first swap . lassoc == \ (d,(a,z)) -> (a,(d,z))
\end{code}
The first sequence begins with |(a,b)| at the top of the stack, moves |a| out of the way onto the rest of the stack, and leaves |b| behind for |g| to consume.
The second sequence begins with |g b| on the top of the stack and |(a,z)| below, moves |g b| to the rest of the stack, leaving |a| on top for |f| to consume.
Following |f|, the stack contains |(f a, (g b, z))|, which gets re-associated to  |((f a, g b), z)|, i.e.,\out{ |((f *** g) (a,b), z)|, which is equivalent to} |first (f *** g) ((a,b),z)| as desired.
%endif
Step-by-step, the stack evolves as follows:
%format --> = "\ \longmapsto\ "
\begin{code}
                 ((a,b)          ,z)
first swap  -->  ((b,a)          ,z)
rassocP     -->  (b              ,(a,z))
first g     -->  (g b            ,(a,z))
lassocP     -->  ((g b,a)        ,z)
first swap  -->  ((a, g b)       ,z)
rassocP     -->  (a              ,(g b,z))
first f     -->  (f a            ,(g b,z))
lassocP     -->  ((f a, g b)     ,z)
\end{code}
Operationally, |first g| and |first f| stand for stack-transformation sub-sequences.
Note that this final stack state is equal to |first (f *** g) ((a,b),z)| as needed.
We have, however, flattened (under the |SF| constructor) into \emph{purely sequential} compositions of functions of three forms:
\begin{itemize}\itemsep0ex
\item |first p| for simple functions |p|, 
\item |rassocP|, and
\item |lassocP|.
\end{itemize}
Moreover, the latter two always come in balanced pairs.

\subsectionl{Duplicating and destroying information}

%format ProductCat = Cartesian
The vocabulary above gives no way to duplicate or destroy information, but there is a standard interface for doing so:\notefoot{I've been experimenting with having |ProductCat| be independent of |MonoidalPCat|. A more conventional choice is to have the former require the latter. I think the clean split enables a generalization later on.}
\begin{code}
class ProductCat k where
  exl  :: (a :* b) `k` a
  exr  :: (a :* b) `k` b
  dup  :: a `k` (a :* a)
\end{code} 
Again, the required homomorphism properties are already in solved form, so we can immediately write them down a sufficient instance:
\begin{code}
instance ProductCat StackFun where
  exl  = stackFun exl
  exr  = stackFun exr
  dup  = stackFun dup
\end{code}
These three operations are used in the translation from $\lambda$-calculus (e.g., Haskell) to categorical form.
The two projections (|exl| and |exr|) arise from translation of pattern-matching on pairs, while duplication is used for translation of pair formation and application expressions, in the guise of the ``fork'' operation \citep[Section 3]{Elliott-2017-compiling-to-categories}:
\begin{code}
(&&&) :: (a `k` c) -> (a `k` d) -> (a `k` (c :* d))
f &&& g = (f *** g) . dup
\end{code}

\subsectionl{Conditional composition (coproducts)}

%format MonoidalSCat = MonoidalS
Just as we have |MonoidalPCat| and |ProductCat| for products (defined above), there are also \emph{dual} counterparts that work on coproducts (sums) instead of products:\footnote{
There are two special forms dual to |first| and |second|:
\begin{code}
left :: MonoidalSCat k => (a `k` c) -> ((a :+ b) `k` (c :+ b))
left f = f +++ id

right :: MonoidalSCat k => (b `k` d) -> ((a :+ b) `k` (a :+ d))
right g = id +++ g
\end{code}
}
\begin{code}
class MonoidalSCat k where
  (+++) :: (a `k` c) -> (b `k` d) -> ((a :+ b) `k` (c :+ d))
\end{code}

There is also a dual interface to |ProductCat|:
%format CoproductCat = Cocartesian
\begin{code}
class CoproductCat k where
  inl  :: a `k` (a :+ b)
  inr  :: b `k` (a :+ b)
  jam  :: (a :+ a) `k` a
\end{code}
The homomorphism properties are easily satisfied:
\begin{code}
instance ProductCat StackFun where
  inl  = stackFun inl
  inr  = stackFun inr
  jam  = stackFun jam
\end{code}

Just as the |(&&&)| (``fork'') operation for producing products is defined via |(***)| and |dup|, so is the |(###)| (``join'') operation for consuming coproducts/sums defined via |(+++)| and |jam|:
\begin{code}
(|||) :: (a `k` c) -> (b `k` c) -> ((a :+ b) `k` c)
f ||| g = jam . (f +++ g)
\end{code}
\mynote{Consider skipping |(***)| and |(+++)| in favor of |(&&&)| and |(###)|, which is consistent with the CtoC paper \citep{Elliott-2017-compiling-to-categories}.}

%format DistribCat = Distributive
Categorical products and coproducts are related in \emph{distributive} categories \citep[Section 1.5.5]{Gibbons2002:Calculating}:%
\footnote{
There's also a right-distributing counterpart:
\begin{code}
distr :: ((u :+ v) :* b) `k` ((u :* b) :+ (v :* b))
distr = (swapP +++ swapP) . distl . swapP
\end{code}
Inverses can be defined without |DistribCat| \citep[Section 1.5.5]{Gibbons2002:Calculating}:
\begin{code}
undistl :: (MonoidalPCat k, MonoidalSCat k, CoproductCat k) => ((a :* u) :+ (a :* v)) `k` (a :* (u :+ v))
undistl = second inl ||| second inr

undistr :: (MonoidalPCat k, MonoidalSCat k, CoproductCat k) => ((u :* b) :+ (v :* b)) `k` ((u :+ v) :* b)
undistr = first inl ||| first inr
\end{code}
\vspace{-2ex}
}
\begin{code}
class (ProductCat k, CoproductCat k) => DistribCat k where
  distl :: (a :* (u :+ v)) `k` ((a :* u) :+ (a :* v))
\end{code}
The |(###)| and |distl| operations suffice to translate multi-constructor |case| expressions to categorical form \citep[Section 8]{Elliott-2017-compiling-to-categories}.
The instance for stack functions is again trivial:
\begin{code}
instance DistribCat StackFun where
  distl = stackFun distl
\end{code}

With the |MonoidalSCat| and |DistribCat| instances for |(->)|, we can define a correct |MonoidalSCat| instance for |StackFun|:

\begin{theorem}[\provedIn{stackFun-MonoidalSCat}]\thmLabel{stackFun-MonoidalSCat}
Given the instance definition above, |stackFun| is a |MonoidalSCat| homomorphism.
\begin{code}
instance MonoidalSCat StackFun where
  SF f +++ SF g = SF (undistr . (f +++ g) . distr)
\end{code}
\end{theorem}
\vspace{-4ex}

\subsectionl{Closed categories}

\mynote{In progress. I don't know whether |StackFun| is closed. In any case, probably move to after \secref{Stack programs}.}

\sectionl{Stack programs}

The definitions of |StackFun| and its type class instances above capture the essence of stack computation, while allowing evaluation as functions (via |evalStackFun|).
For optimization and code generation, however, we will need to inspect the structure of a computation, which is impossible with |StackFun| due to its representation as a function.
To remedy this situation, let's now make the notion of stack computation explicit as a data type having a precise relationship with the function representation.
%\notefoot{Introduce |Op| and |evalOp| before |StackOp| and |evalStackOp|.}

As a first step, define a data type of reified primitive functions, along with an evaluator:
\begin{code}
data Prim :: * -> * -> * NOP where
  Exl  :: Prim (a :* b) a
  Exr  :: Prim (a :* b) b
  Dup  :: Prim a (a :* a)
  ...
  Negate :: Num a => Prim a a
  Add, Sub, Mul :: Num a => Prim (a :* a) a
  ...

evalPrim :: Prim a b -> (a -> b)
evalPrim Exl     = exl
evalPrim Exr     = exr
evalPrim Dup     = dup
               ...
evalPrim Negate  = negateC
evalPrim Add     = addC
evalPrim Sub     = subC
evalPrim Mul     = mulC
               ...
\end{code}

%format Pure = Prim
%% %format Push = RotR
%% %format Pop  = RotL
A stack program is a sequence of instructions, most of which correspond to primitive functions that replace the top of the stack without using the rest, and the others that re-associate:\notefoot{Maybe rename the constructors to something like |FirstSO|, |RassocSO|, and |LassocSO|. Look for prettier alternatives.}
\begin{code}
data StackOp :: * -> * -> * NOP where
  Pure  :: Prim a b -> StackOp (a :* z) (b :* z)
  Push  :: StackOp ((a :* b) :* z) (a :* (b :* z))
  Pop   :: StackOp (a :* (b :* z)) ((a :* b) :* z)
\end{code}
Stack operations have a simple interpretation as functions:\footnote{The operations |negateC|, |addC|, etc are the categorical versions of |negate|, (+), etc, uncurried where needed. We use the categorical versions here for easier generalization later.}
\begin{code}
evalStackOp :: StackOp u v -> (u -> v)
evalStackOp (Pure f)  = first (evalPrim f)
evalStackOp Push      = rassocP
evalStackOp Pop       = lassocP
\end{code}

%format :< = "\triangleleft"
We will form chains (linear sequences) of stack operations, each feeding its result to the next:\notefoot{Maybe I should change |StackOps| to preserve the composition structure.
The calculations would be simpler, and the implementation more efficient.}
\begin{code}
infixr 5 :<
data StackOps :: * -> * -> * NOP where
  Nil   :: StackOps a a
  (:<)  :: StackOp a b -> StackOps b c -> StackOps a c
NOP
evalStackOps :: StackOps u v -> (u -> v)
evalStackOps Nil           = id
evalStackOps (op :< rest)  = evalStackOps rest . evalStackOp op
\end{code}

%format ++* = ++
We'll want to compose these chains sequentially:
\begin{code}
infixr 5 ++*
(++*) :: StackOps a b -> StackOps b c -> StackOps a c
Nil          ++* ops' = ops'
(op :< ops)  ++* ops' = op :< (ops ++* ops')
\end{code}

\begin{lemma} \lemLabel{StackOps-cat}
|Nil| and |(++*)| implement identity and composition on functions in the following sense:
\begin{code}
id == evalStackOps Nil

evalStackOps g . evalStackOps f == evalStackOps (f ++* g)
\end{code}
\end{lemma}
\vspace{-3.5ex}
%% TODO: customize vertical spacing around lemmas instead.
\begin{proof}
The first property is immediate from the definition of |evalStackOps|.
The second follows by structural induction on |g|.
\end{proof}

A complete stack program is a chain of stack operations that can change only the top of the stack: 
\begin{code}
data StackProg a b = SP { unSP :: forall z. StackOps (a :* z) (b :* z) }
\end{code}
To compile a stack program, convert it to a stack function:
\begin{code}
progFun :: StackProg a b -> StackFun a b
progFun (SP ops) = SF (evalStackOps ops)
\end{code}
We can also convert all the way to a regular function:
\begin{code}
evalProg :: StackProg a b -> (a -> b)
evalProg = evalStackFun . progFun
\end{code}

This |evalProg| definition constitutes an interpreter for stack programs.
Our quest, however, is the reverse.
Given a function |f|, we want to construct a purely sequential, stack-manipulating program |p| such that |evalProg p == f|.
As stated, this goal is impossible, since functions are not inspectable.
Moreover, for a given function |f| there may be no program |p| that satisfy this requirement, or there may be many such programs.
Although we cannot invert |evalProg| as written, we can transform this specification into a correct and effective implementation.
As in \secref{Stack functions}, we can calculate instances of |Category| etc for |StackProg| resulting in \figref{progFun}.

%% As usual, we can derive instances for our new category by homomorphic specification:
\begin{theorem}[\provedIn{progFun}]\thmLabel{progFun}
Given the definitions in \figref{progFun}, |progFun| is a homomorphism with respect to each instantiated class.
\end{theorem}
\begin{figure}
\begin{center}
\begin{code}
instance Category StackProg where
  id = SP Nil
  SP g . SP f = SP (f ++* g)

instance MonoidalPCat StackFun where
  first (SP ops) = SP (Push :< ops ++* Pop :< Nil)
  second g = swap . first g . swap
  f *** g = first f . second g

primProg :: Prim a b -> StackProg a b
primProg p = SP (Pure p :< Nil)

instance ProductCat StackProg where
  exl  = primProg Exl
  exr  = primProg Exr
  dup  = primProg Dup

instance Num a => NumCat StackProg a where
  negateC = primProg Negate
  addC    = primProg Add
  subC    = primProg Sub
  mulC    = primProg Mul
          ...
\end{code}
\caption{Stack programs (specified by |progFun| as homomorphism and calculated in \proofRef{progFun})}
\figlabel{progFun}
\end{center}
\end{figure}

\begin{corollary}\thmLabel{evalProg}
Given the definitions in \figref{progFun}, |evalProg| is also a homomorphism with respect to each instantiated class.
\end{corollary}
\begin{proof}
The composition of homomorphisms (here |evalStackFun| and |progFun|) is a homomorphism (|evalProg|).
\end{proof}

\sectionl{What's next?}

\mynote{Working here.}

\begin{itemize}
\item Examples
\item Optimization
\item More with |CoproductCat|, including multi-constructor |case| expressions.
      Maybe start with conditionals.
      Hm!
      I don't think I can define |(+++)| on |StackProg|, because the representation is a linear sequence of stack operations.

\end{itemize}

\sectionl{Related work}

\begin{itemize}
\item \citep{Meijer1992Dissertation}
\item \citep{Meijer1991MoreAdvice}
\item \citep{BahrHutton2015:CCC}
\item \citep{VazouEtAl2018:TPFA}
\item \citep{McKinnaWrite2006}
\end{itemize}

\appendix

\vspace{2ex}

\sectionl{Proofs}

\subsection{\lemRef{evalStackFun-as-left-inverse}}\proofLabel{evalStackFun-as-left-inverse}

We need to show that |evalStackFun| is a left inverse for |stackFun|, i.e., for all |f|, |evalStackFun (stackFun f) == f|.
Reasoning equationally,
\begin{code}
    evalStackFun (stackFun f)
==  {- definition of |stackFun| -}
    evalStackFun (SF (first f))
==  {- second definition of |evalStackFun| -}
    rcounit . first f . runit
==  {- definition of |(.)| on functions -}
    \ a -> rcounit (first f (runit a))
==  {- definition of |rcounit| on functions -}
    \ a -> rcounit (first f (a,()))
==  {- definition of |first| on functions -}
    \ a -> rcounit (f a,())
==  {- definition of |rcounit| on functions -}
    \ a -> f a
==  {- $\eta$-reduction -}
    f
\end{code}

\subsection{\lemRef{stackFun-surjective}}\proofLabel{stackFun-surjective}

\mynote{Adapt Joachim Breitner's proof in haskell-cafe email 2018-07-23, giving him credit.}

\subsection{\lemRef{stackFun-injective}}\proofLabel{stackFun-injective}

We need to show that |stackFun| is injective, i.e., |stackFun f == stackFun f' ==> f == f'|.
Since |stackFun == SF . first|, and |SF| is |injective|, we only need show that |first| as injective:
\begin{code}
      first f == first f'
<==>  {- equality on functions (extensionality) -}
      forall x z. first f (x,z) == first f' (x,z)
<==>  {- definition of |first| -}
      forall x z. (f x,z) == (f' x,z)
<==>  {- equality on pairs -}
      forall x z. f x == f' x && z == z
<==>  {- trivial conjunct -}
      forall x. f x == f' x
<==>  {- equality on functions -}
      f == f'
\end{code}

\subsection{\thmRef{stackFun-MonoidalSCat}}\proofLabel{stackFun-MonoidalSCat}

The |MonoidalSCat| homomorphism property:
\begin{code}
stackFun f +++ stackFun g == stackFun (f +++ g)
\end{code}
Using the definition of |stackFun|,
\begin{code}
SF (first f) +++ SF (first g) == SF (first (f +++ g))
\end{code}
Simplify the RHS:
\begin{code}
    first (f +++ g)
==  {- |undistr . distr == id| -}
    (undistr . distr) . first (f +++ g) . (undistr . distr)
==  {- associativity of |(.)| -}
    undistr . (distr . first (f +++ g) . undistr) . distr
==  {- \lemRef{distr-first-plus} -}
    undistr . (first f +++ first g) . distr
\end{code}
The required |MonoidalSCat| homomorphism is thus equivalent to
\begin{code}
SF (first f) +++ SF (first g) == SF (undistr . (first f +++ first g) . distr)
\end{code}
Strengthen by generalizing from |first f| and |first g|, resulting in a sufficient definition:
\begin{code}
instance MonoidalSCat StackFun where
  SF f +++ SF g == SF (undistr . (f +++ g) . distr)
\end{code}

The needed lemma:
\begin{lemma} \lemLabel{distr-first-plus}
\begin{code}
distr . first (f +++ g) . undistr == first f +++ first g
\end{code}
\end{lemma}
\vspace{-2ex}
\begin{proof}
It will be convenient to prove an equivalent, slightly different form, eliminating |distr|:
\begin{code}
first (f +++ g) . undistr == undistr . (first f +++ first g)
\end{code}
Simplify the LHS:
\begin{code}
    first (f +++ g) . undistr
==  {- definition of |undistr| \citep[Section 1.5.5 variation]{Gibbons2002:Calculating} -}
    first (f +++ g) . (first inl ||| first inr)
==  {- |r . (p ### q) == (r . p) ### (r . q)| \citep[Section 1.5.2]{Gibbons2002:Calculating} -}
    first (f +++ g) . first inl ||| first (f +++ g) . first inr
==  {- property of |first| and |(.)| -}
    first ((f +++ g) . inl) ||| first ((f +++ g) . inr)
==  {- \citep[Section 1.5.2 variation]{Gibbons2002:Calculating} -}
    first (inl . f) ||| first (inr . g)
\end{code}
Then the RHS:
\begin{code}
   undistr . (first f +++ first g)
=  {- definition of |undistr| -}
   (first inl ||| first inr) . (first f +++ first g)
=  {- |(###)/(+++)| law \citep[Section 1.5.2]{Gibbons2002:Calculating} -}
   (first inl . first f) ||| (first inr . first g)
=  {- Property of |first| and |(.)| -}
   first (inl . f) ||| first (inr . g)
\end{code}
% \vspace{-4ex}
\end{proof}

\subsection{\thmRef{progFun}}\proofLabel{progFun}

Let's see how the definitions in \figref{progFun} follow from homomorphism properties.

\subsubsection{|Category|}

The homomorphic requirement for |id|:
\begin{code}
progFun id == id
\end{code}
Simplify the LHS:
\begin{code}
    progFun id
==  {- |SP| and |unSP| are inverses -}
    progFun (SP (unSP id))
==  {- definition of |progFun| -}
    SF (evalStackOps (unSP id))
\end{code}
Then the RHS:
\begin{code}
    id
==  {- definition of |id| on |SF| -}
    SF id
==  {- \lemRef{StackOps-cat} -}
    SF (evalStackOps Nil)
==  {- |unSP| and |SP| are inverses  -}
    SF (evalStackOps (unSP (SP Nil)))
\end{code}
The simplified |id| homomorphism requirement:
\begin{code}
     SF (evalStackOps (unSP id)) == SF (evalStackOps (unSP (SP Nil)))
<==  {- |SF . evalStackOps . unSP| is a function -}
     id == SP Nil
\end{code}

The homomorphic requirement for |(.)|:
\begin{code}
progFun (SP g . SP f) == progFun (SP g) . progFun (SP f)
\end{code}
Simplify the LHS:
\begin{code}
    progFun (SP g . SP f)
==  {- definition of |progFun| -}
    SF (evalStackOps (unSP (SP g . SP f)))
\end{code}
Then the RHS:
\begin{code}
    progFun (SP g) . progFun (SP f)
==  {- definition of |progFun| -}
    SF (evalStackOps g) . SF (evalStackOps f)
==  {- definition of |(.)| for |StackFun| -}
    SF (evalStackOps g . evalStackOps f)
==  {- \lemRef{StackOps-cat} -}
    SF (evalStackOps (f ++* g))
\end{code}
These simplified |(.)| homomorphism requirement:
\begin{code}
      SF (evalStackOps (unSP (SP g . SP f))) == SF (evalStackOps (f ++* g))
<==   {- |SF . evalStackOps| is a function -}
      unSP (SP g . SP f) == f ++* g
<==>  {- |SP| is bijective -}
      SP (unSP (SP g . SP f)) == SP (f ++* g)
<==>  {- |SP| and |unSP| are inverses -}
      SP g . SP f == SP (f ++* g)
\end{code}
These simplified homomorphic specifications are in solved form and so suffice as a correct implementation.

\subsubsection{Primitive functions}

The |primProg| function (\figref{progFun}) captures primitive functions in the following sense:
\begin{lemma} \lemLabel{primProg}
\begin{code}
progFun (primProg op) == stackFun (evalPrim op)
\end{code}
\end{lemma}
\begin{proof}
Reason equationally:
\begin{code}
    progFun (primProg op)
==  {- definition of |primProg| -}
    progFun (SP (Pure op :< Nil))
==  {- definition of |progFun| -}
    SF (evalStackOps (Pure op :< Nil))
==  {- definition of |evalStackOps| -}
    SF (evalStackOps Nil . evalStackOp (Pure op))
==  {- definitions of |evalStackOps| and |evalStackOp| -}
    SF (id . first (evalPrim op))
==  {- |Category| law -}
    SF (first (evalPrim op))
==  {- definition of |stackFun| -}
    stackFun (evalPrim op)
\end{code}
\end{proof}

As a typical use of |evalPrim|, consider the homomorphism equation |progFun exl == exl|, beginning with the RHS:
\begin{code}
    exl
==  {- |stackFun| is a |ProductCat| homomorphism -}
    stackFun exl
==  {- definition of |evalPrim| -}
    stackFun (evalPrim Exl)
==  {- \lemRef{primProg} -}
    progFun (opP Exl)
\end{code}
Our homomorphic specification is thus
\begin{code}
     progFun exl == progFun (opP Exl)
<==  {- |progFun| is a function -}
     exl == opP Exl
\end{code}

\subsubsection{|MonoidalPCat|}

The required homomorphism:
\begin{code}
progFun (first f) == first (progFun f)
\end{code}
In other words,
\begin{code}
progFun (first (SP ops)) == first (progFun (SP ops))
\end{code}
Simplify the RHS:
\begin{code}
    first (progFun (SP ops))
==  {- definition of |progFun| -}
    first (SF (evalStackOps ops))
==  {- definition of |first| on |StackFun| -}
    SF (lassocP . evalStackOps ops . rassocP)
==  {- definition of |evalStackOps|; \lemRef{StackOps-cat}  -}
    SF (evalStackOps (Push :< ops ++* Pop :< Nil))
==  {- definition of |progFun| -}
    progFun (SP (Push :< ops ++* Pop :< Nil))
\end{code}
The simplified homomorphism:
\begin{code}
progFun (first (SP ops)) == progFun (SP (Push :< ops ++* Pop :< Nil))
\end{code}
A sufficient definition:
\begin{code}
first (SP ops) = SP (Push :< ops ++* Pop :< Nil)
\end{code}

\mynote{|MonoidalSCat|. Doesn't seem possible with the current |StackProg| definition.}

\bibliography{bib}

\end{document}

