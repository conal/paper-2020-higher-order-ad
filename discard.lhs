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
