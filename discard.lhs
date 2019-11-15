%% Discarded text, saved here in case some is useful later.


%% Some junk

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
