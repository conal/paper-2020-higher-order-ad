{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE UndecidableInstances #-}  -- see below

-- | Free vector space as representable functors
module Vector where

import Data.Monoid (Sum(..),Product(..))
import GHC.Generics (U1(..),Par1(..),(:*:)(..),(:.:)(..))
#ifdef VectorSized
import GHC.TypeLits (KnownNat)
import Data.Vector.Sized (Vector)
#endif
import Data.Constraint ((:-)(..),Dict(..))
import Data.Functor.Rep (Representable(..),pureRep,liftR2)

import ConCat.Misc ((:*),(<~),sqr)
import qualified ConCat.Rep as CR
import ConCat.AltCat (OpCon(..),Sat,type (|-)(..),fmapC)

{--------------------------------------------------------------------
    Free vector spaces
--------------------------------------------------------------------}

type RF = Representable

-- | Zero vector
zeroV :: (RF g, Num a) => g a
zeroV = pureRep 0

infixl 7 *^, <.>, >.<
infixl 6 ^+^, ^-^

-- TODO: Replace Num constraints with Semiring

-- | Scale a vector
scaleV, (*^) :: (Functor f, Num s) => s -> f s -> f s
s *^ v = (s *) <$> v
scaleV = (*^)
{-# INLINE (*^) #-}
{-# INLINE scaleV #-}

-- | Negate a vector
negateV :: (Functor f, Num s) => f s -> f s
negateV = ((-1) *^)
{-# INLINE negateV #-}

-- | Add vectors
addV, (^+^) :: (RF f, Num s) => f s -> f s -> f s
(^+^) = liftR2 (+)
addV = (^+^)
{-# INLINE (^+^) #-}
{-# INLINE addV #-}

-- | Subtract vectors
subV, (^-^) :: (RF f, Num s) => f s -> f s -> f s
(^-^) = liftR2 (-)
subV = (^-^)
{-# INLINE (^-^) #-}
{-# INLINE subV #-}

-- | Inner product. TODO: complex
dotV, (<.>) :: (RF f, Foldable f, Num s) => f s -> f s -> s
x <.> y = sum (liftR2 (*) x y)
dotV = (<.>)
{-# INLINE (<.>) #-}
{-# INLINE dotV #-}

-- | Norm squared
#if 1
normSqr :: (Functor f, Foldable f, Num s) => f s -> s
normSqr = sum . fmap sqr
#else
normSqr :: (RF f, Foldable f, Num s) => f s -> s
normSqr u = u <.> u
#endif
{-# INLINE normSqr #-}

-- | Distance squared
distSqr :: (RF f, Foldable f, Num s) => f s -> f s -> s
distSqr u v = normSqr (u ^-^ v)
{-# INLINE distSqr #-}

-- | Outer product
outerV, (>.<) :: (Num s, Functor f, Functor g) => g s -> f s -> g (f s)
x >.< y = (*^ y) <$> x
outerV = (>.<)
{-# INLINE (>.<) #-}
{-# INLINE outerV #-}

-- | Normalize a vector (scale to unit magnitude)
normalizeV :: (Functor f, Foldable f, Floating a) => f a -> f a
normalizeV xs = (/ sum xs) <$> xs
{-# INLINE normalizeV #-}

-- Would I rather prefer swapping the arguments (equivalently, transposing the
-- result)?

{--------------------------------------------------------------------
    Conversion
--------------------------------------------------------------------}

type RepIsoV s a = (CR.HasRep a, IsoV s (CR.Rep a), V s a ~ V s (CR.Rep a))

class Representable (V s a) => IsoV s a where
  type V s a :: * -> *
  toV :: a -> V s a s
  unV :: V s a s -> a
  -- Default via Rep.
  type V s a = V s (CR.Rep a)
  default toV :: RepIsoV s a => a -> V s a s
  default unV :: RepIsoV s a => V s a s -> a
  toV = toV . CR.repr
  unV = CR.abst . unV
  {-# INLINE toV #-} ; {-# INLINE unV #-}

-- Illegal nested type family application ‘V s (CR.Rep a)’
-- (Use UndecidableInstances to permit this)

inV :: (IsoV s a, IsoV s b) => (a -> b) -> (V s a s -> V s b s)
inV = toV <~ unV

onV :: (IsoV s a, IsoV s b) => (V s a s -> V s b s) -> (a -> b)
onV = unV <~ toV

onV2 :: (IsoV s a, IsoV s b, IsoV s c) => (V s a s -> V s b s -> V s c s) -> (a -> b -> c)
onV2 = onV <~ toV

-- Can I replace my HasRep class with Newtype?

-- type IsScalar s = (IsoV s s, V s s ~ Par1)

instance IsoV s () where
  type V s () = U1
  toV () = U1
  unV U1 = ()

-- -- Replace by special cases as needed
-- instance IsoV s s where
--   type V s s = Par1
--   toV = Par1
--   unV = unPar1

instance IsoV Float Float where
  type V Float Float = Par1
  toV = Par1
  unV = unPar1

instance IsoV Double Double where
  type V Double Double = Par1
  toV = Par1
  unV = unPar1

-- etc

instance (IsoV s a, IsoV s b) => IsoV s (a :* b) where
  type V s (a :* b) = V s a :*: V s b
  toV (a,b) = toV a :*: toV b
  unV (f :*: g) = (unV f,unV g)
  {-# INLINE toV #-} ; {-# INLINE unV #-}

instance OpCon (:*) (Sat (IsoV s)) where
  inOp = Entail (Sub Dict)
  {-# INLINE inOp #-}

instance (IsoV s a, IsoV s b, IsoV s c) => IsoV s (a,b,c)
instance (IsoV s a, IsoV s b, IsoV s c, IsoV s d) => IsoV s (a,b,c,d)

-- Sometimes it's better not to use the default. I think the following gives more reuse:

-- instance IsoV s a => IsoV s (Pair a) where
--   type V s (Pair a) = Pair :.: V s a
--   toV = Comp1 . fmap toV
--   unV = fmap unV . unComp1

-- Similarly for other functors

instance IsoV s (U1 a)
instance IsoV s a => IsoV s (Par1 a)
instance (IsoV s (f a), IsoV s (g a)) => IsoV s ((f :*: g) a)
instance (IsoV s (g (f a))) => IsoV s ((g :.: f) a)

-- instance IsoV s (f a) => IsoV s (SumV f a)

instance IsoV s a => IsoV s (Sum a)
instance IsoV s a => IsoV s (Product a)
-- TODO: More newtypes

-- Sometimes it's better not to use the default. I think the following gives more reuse:

-- instance IsoV s a => IsoV s (Pair a) where
--   type V s (Pair a) = Pair :.: V s a
--   toV = Comp1 . fmap toV
--   unV = fmap unV . unComp1

-- Similarly for other functors

instance IsoV s b => IsoV s (a -> b) where
  type V s (a -> b) = (->) a :.: V s b
  toV = Comp1 . fmap toV
  unV = fmap unV . unComp1
  {-# INLINE toV #-} ; {-# INLINE unV #-}

#ifdef VectorSized
instance (IsoV s b, KnownNat n) => IsoV s (Vector n b) where
  type V s (Vector n b) = Vector n :.: V s b
  toV = Comp1 . fmapC toV
  unV = fmapC unV . unComp1
  {-# INLINE toV #-}
  {-# INLINE unV #-}
#endif

-- TODO: find a better alternative to using fmapC explicitly here. I'd like to
-- use fmap instead, but it gets inlined immediately, as do all class
-- operations.


