{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-imports #-} -- TEMP

{-# LANGUAGE UndecidableInstances #-}  -- see below

-- | Higher-order AD

module AD where

import Prelude hiding (id,(.),curry,uncurry)

import GHC.Types (Type)
import GHC.Generics  (Par1(..), (:*:)(..), (:.:)(..))
import Control.Newtype.Generics
import Data.Constraint hiding ((&&&),(***),(:=>))
import Control.Comonad.Cofree (Cofree (..))
import Data.Functor.Rep (Representable(..),pureRep)
-- import Control.Applicative (liftA2)

import ConCat.Misc ((:*),R,inNew,inNew2)
import ConCat.Category

import Vector

-- | Representable functor for @a@ as container of @s@
type family Obj s a :: Type -> Type

type instance Obj Float  Float  = Par1
type instance Obj Double Double = Par1
-- etc

type instance Obj s (a :* b) = Obj s a :*: Obj s b
type instance Obj s (a -> b) = Obj s a :=> Obj s b

-- TODO: Replace Obj by V from Vector. Oops! I can't define toV for a -> b. I
-- guess I have to move V outside of the IsoV class and then rename IsoV. Then
-- define a derivative operation that works with regular types and requires
-- IsoV. If V is a free-floating type family, can I define a generic-based
-- default?

-- | Infinitely differentiable functions. Maps a domain vector to a trie of
-- partial derivatives.
newtype D s a b = D ((Obj s a :-> Obj s b) s)

type Tower f g = Cofree f :.: g

pattern Uncomps :: Functor h => h ((g :.: f) a) -> h (g (f a))
pattern Uncomps w <- (fmap Comp1 -> w)
 where
   Uncomps w = fmap unComp1 w

pattern (:<|) :: Functor f => g s -> f (Tower f g s) -> Tower f g s
pattern v :<| ts = Comp1 (v :< Uncomps ts)

type (f :-> g) s = f s -> Tower f g s

-- | Functor version of ':->'.
newtype (f :=> g) s = Exp ((f :-> g) s)

-- data Cofree f a = a :< f (Cofree f a)

-- Use Num for now, but change to Semiring later.

-- TODO: Do these derivative tries form a cartesian category? If so, maybe I can
-- get higher-order AD as a special case of generalized first-order AD.

zeroT :: (Representable f, Representable g, Num s) => Tower f g s
zeroT = pureRep 0
-- zeroT = Comp1 (pureRep zeroV)

constT :: (Representable f, Representable g, Num s) => g s -> Tower f g s
constT a = a :<| pureRep zeroT

infixr 2 :-*
type (f :-* g) s = f (g s)

-- Note column-major, which is not usually my preference.

-- TODO: consider instead type f :-* g = f :.: g

kronecker :: (Eq i, Num a) => i -> i -> a
kronecker i j | i == j    = 1
              | otherwise = 0

tabulate2 :: (Representable f, Representable g) => (Rep f -> Rep g -> a) -> f (g a)
tabulate2 h = tabulate (\ i -> tabulate (\ j -> h i j))

-- Equivalently,
-- 
-- tabulate2 h = tabulate (tabulate . h)
-- tabulate2 h = tabulate (fmap tabulate h)
-- tabulate2 = tabulate . fmap tabulate
-- tabulate2 = (fmap tabulate) (fmap tabulate)   -- ;)

idL :: (Representable f, Eq (Rep f), Num s) => (f :-* f) s
idL = tabulate2 kronecker

-- constT id

type OkF f = (Representable f, Eq (Rep f))

class    (OkF (Obj s a), Num s) => OkObj s a
instance (OkF (Obj s a), Num s) => OkObj s a

-- Illegal nested constraint ‘OkF (Obj s a)’
-- (Use UndecidableInstances to permit this)

-- instance Category (D s) where
--   type Ok (D s) = OkObj s
--   id = D (\ a -> a :<| fmap constT idL)
--   D g . D f = D (\ a -> let { b :<| f' = f a ; c :<| g' = g b } in c :<| _)

-- data Cofree f a = a :< f (Cofree f a)

-- Specification

deriv :: (f s -> g s) -> (f s -> (f :-* g) s)
deriv = error "deriv: for specification only"

derivs :: (Functor f) => (f s -> g s) -> (f s -> Tower f g s)
derivs h = ds where ds = h &&&& deriv ds
-- derivs h = h &&&& deriv (derivs h)

(&&&&) :: Functor f => (f s -> g s) -> (f s -> f (Tower f g s)) -> (f :-> g) s
h &&&& h' = \ u -> h u :<| h' u
-- (&&&&) = liftA2 (:<|)
