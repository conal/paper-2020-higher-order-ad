---
title: Higher-order, higher-order automatic differentiation
format: markdown
autolink_bare_uris: true
toc: true
substMap: []
...

[*The simple essence of automatic differentiation*]: http://conal.net/papers/essence-of-ad "paper by Conal Elliott (2018)"

# Outline

## Background & motivation

## Linear maps and vector spaces

Differentiation is about linear maps, which are defined on vector spaces (and on generalizations such as semimodules).

Free vector space as functions and then as representable functors (RFs).

Define a first concrete notion linear maps abstractly as functions that satisfy linearity.

One could formalize this notion in a variety of ways, e.g., with refinement types or dependent types.
Alternatively, introduce a representation that is intrinsically linear so that propositions and proofs are unnecessary.
That change allows us to use a simpler type system such as Haskell's safely, *and* it enables more efficient implementations of the standard operations from linear algebra.

Formalize via a denotation and corresponding homomorphism properties.
Happily, much of the vocabulary of linear maps comes from category theory, and relates to use of this same vocabulary in other settings in a simple and regular way, via homomorphisms.

For the "Naperian" representations (as indexing function (IF) and as RFs), we'd naturally have indices/logarithms or functors as the objects.
For IFs (non-memoized), categorical products and coproducts are both type sums (coproducts).
For RFs (memoized), categorical products and coproducts are both  functor products.
These two choices have a simple and familiar correspondence however, based on type isomorphisms.

Since RFs strictly generalize IFs while providing an efficient representation (memoized and easily mapped to arrays in many cases), we'll go with that choice.
It's often easier, however, to reason in terms of indexing functions, so we'll sometimes do so.
In all such cases, the IF-based reasoning translates to RF reasoning.

Assume per-category products, coproducts, and (especially) exponentials, which is customary in category theory.

We can also use more familiar types in a categorical formulation by moving the representation of vector spaces (whether memoized or not) into the representation of linear maps:
``` haskell
newtype LMap s a b = L ((V s a :-* V s b) s)
```

I think I'll want to use the functor category for most of the paper and switch to this other form later so that I can implement HOAD in CtoC (hence differentiable Haskell), which doesn't currently support functor categories or per-category products, coproducts, and (especially) exponentials.

Hm. I could also change the definition of `Closed` so that exponential objects are synonymous with arrows.
Note that "the differentiable curry" makes this same choice.

## A first-order, first-order category of computably differentiable functions

Review from [*The Simple Essence of Automatic Differentiation*] (TSE), but cast in terms of RFs:
``` haskell
newtype D s u v = D (u s -> v s :* (u :-* v) s)
```

## Higher-order functions (closure)

First doomed attempt.
What should the exponential be?
First try a function.
We'll need to implement `curry` and `eval`:

``` haskell
eval :: D s ((u -> v) :* v)

curry :: D s (u :*: v) w -> D s u (u :->: v)
```

where `(:->:)` is functor-level `(->)`:
``` haskell
newtype (u :->: v) s = Fun (u s -> v s)
```

**Oops.** I don't think `:->` takes RFs to RFs.
It's not the right thing anyway, since `a -> b` is a vector space on `s` whenever `b` is, regardless of `a`:
``` haskell
V s (a -> b) = (a ->) . V s b
```

Maybe I should stick with types as objects in the category after all.
