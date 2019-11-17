%% Discarded text, saved here in case some is useful later.

\begin{code}

                  fh    :: a -> b -> c
             der  fh a  :: a :-* (b -> c)
applyTo b               :: (b -> c) :-* c
applyTo b .  der  fh a  :: a :-* c

    der (applyTo b . fh) a           -- |applyTo| and |(.)|
==  der applyTo b (fh a) . der fh a  -- chain rule
==  applyTo b . der fh a             -- |applyTo b| is linear

    uncurry fh . (,a)
==  \ b -> (uncurry fh . (a,)) b
==  \ b -> uncurry fh (a,b)
==  \ b -> fh a b
==  fh a

    uncurry fh . (,b)
==  \ a -> (uncurry fh . (,b)) a
==  \ a -> uncurry fh (a,b)
==  \ a -> fh a b
==  applyTo b . fh

g :: a -> b -> c

uncurry g :: a :* b -> c

wrapO (uncurry g) :: O a :* O b -> O c

ado (uncurry g)
adh (wrapO (uncurry g))

f :: a :* b -> c

curry f :: a -> b -> c

wrapO (curry f)  :: O a -> O (b -> c)
                 :: O a -> D (O b) (O c)

    wrapO (curry f)
==  toO . curry f . unO
==  ado . curry f . unO
==  adh . wrapO . curry f . unO
==  adh . curry (toO . f . unO)
==  adh . curry (wrapO f)

    wrapO (uncurry g)
==  toO . uncurry g . unO
==  ado . uncurry g . unO
==  adh . wrapO . uncurry g . unO
==  adh . uncurry (toO . g . unO)
==  adh . uncurry (wrapO g)


D ((adh . wrapO . uncurry g . unO) &&& der (adh . wrapO . uncurry g . unO))
D (\ a -> (adh (wrapO (uncurry g (unO a))), der (adh . wrapO . uncurry g . unO) a))




    der (curry f) a
==  \ b -> derl f (a,b)


\end{code}


If |f :: a :* b -> c|, then |curry f :: a -> b -> c|, so |der (curry f) a :: a :-* b -> c|, i.e., a linear map from |a| to |b -> c|.
As |a| varies ...


\begin{code}
           
     


der (curry f) a da b = der f (a,b) (da,0)

  der (curry f) a
= \ da b -> der f (a,b) (da,0)
= forkF (\ b da -> der f (a,b) (da,0))
= forkF (\ b -> der f (a,b) . inl)
= forkF (\ b -> derl f (a,b))
= forkF (derl f . (a,))
\end{code}

Note that |der g a :: a :-* b -> c|.
A simple proof uses the chain rule in reverse:\footnote{As a sanity check, the RHS (``|\ da b -> ...|'') is indeed linear, because |der (applyTo b . g) a| linear is (being a derivative) and noting the usual interpretation of scaling and addition on functions.}
\begin{code}
    der g a
==  \ da b -> der g a da b                          -- $\eta$ expansion (twice)
==  \ da b -> applyTo b (der g a da)                -- |applyTo| definition
==  \ da b -> (applyTo b . der g a) da              -- |(.)| on functions
==  \ da b -> (der (applyTo b) (g a) . der g a) da  -- chain rule
==  \ da b -> der (applyTo b . g) a da              -- |applyTo b| is linear
\end{code}


\footnote{As a sanity check, the RHS (``|\ da b -> ...|'') is indeed linear, because |der (applyTo b . g) a| linear is (being a derivative) and noting the usual interpretation of scaling and addition on functions.}


\sectionl{Pair-valued Domains}

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


\subsection{Differentiation and Uncurrying}\proofLabel{der-uncurry}

\begin{code}
    der (uncurry h) (a,b)
==  der (eval . first h) (a,b)                      -- CCC law (above)
==  der eval (first h (a,b)) . der (first h) (a,b)  -- chain rule
==  der eval (h a,b) . der (first h) (a,b)          -- |first| for functions
==  (at b ||| der (h a) b) . der (first h) (a,b)    -- |der eval| (above)
==  (at b ||| der (h a) b) . first (der h a)        -- below
==  (at b ||| der (h a) b) . (der h a *** id)       -- |first| definition
==  at b . der h a ||| der (h a) b                  -- |(f ### g) . (p *** q) == f . p ### g . q|

    der (first h) (a,b)
==  der (h *** id) (a,b)                               -- |first| definition
==  der h a *** der id b                               -- \thmRef{cross}
==  der h a *** id                                     -- |id| is linear
==  first (der h a)                                    -- |first| definition
\end{code}



\note{Type check:}
\begin{code}
f . (a,) :: b -> c
f . (,b) :: a -> c
der (f . (,b)) a :: a :-* c
derl f (a,b) :: a :-* c
der f (a,b) . inl :: a :-* c
\ b -> der f (a,b) . inl :: b -> a :-* c
forkF (\ b -> der f (a,b) . inl) :: a :-* b -> c
\end{code}


Now we do not need the general |der|, but rather the specific |der eval|.
If |eval| were linear, we could apply \thmRef{deriv-linear}, and if |eval| were bilinear, we could apply \corRef{deriv-bilinear}, but alas, |eval| is neither.
No matter, as we can instead use the technique of partial derivatives (\secref{Pair-Valued Domains}).\notefoot{Move this calculation into a proof of a theorem stated in \secref{Pair-Valued Domains}}.
\begin{code}
    der eval (f,a)
==  derl eval (f,a) !!! derr eval (f,a)          -- method of partial derivatives
==  der (eval . (,a)) f !!! der (eval . (f,)) a  -- |derl| and |derr| alternative definitions
==  der (at a) f !!! der f          a            -- |eval| on functions
==  at a !!! der f a                             -- linearity of |at a|
==  \ (df,dx) -> df a + der f a dx               -- |(!!!) on linear maps|; |at| definition
\end{code}
Now we can complete the calculation of |eval| for |D|:
\begin{code}
    eval
==  adh eval
==  D (eval &&& der eval)                       -- definition of |adh|
==  D (\ (f,a) -> (eval (f,a), der eval (f,a))  -- |(&&&) on functions|
==  D (\ (f,a) -> (f a, der eval (f,a))         -- |eval| on functions
==  D (\ (f,a) -> (f a, at a !!! der f a))      -- above
\end{code}


Although this final form is well-defined, it uses the noncomputable |der| and so is not a computable recipe, leaving us in a pickle.
Let's look for some wiggle room.
