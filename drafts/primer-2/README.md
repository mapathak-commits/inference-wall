*What a large language model actually computes when it reads your prompt and writes an answer:
how a token becomes a vector, what a transformer block does to it, and how the next token falls
out the end. No benchmarks, no math, no equations. You need to be able to picture what one pass
through the model does.*

*Manas Pathak · September 1, 2026*

# What actually happens inside an LLM

Ask a language model a question and it answers one token at a time, each token produced by
running the whole model once. That single run is a **forward pass**, and this post is about what
happens inside it: what the model is doing between the moment your prompt goes in and the moment
the next token comes out. The picture is smaller than it looks from outside, and once you have it,
a lot of otherwise-mysterious behavior stops being mysterious.

Two examples of that, which this post pays off by the end. First, why generating a token does not
require rereading the whole conversation: the model computes a key and a value for each token
once and reuses them forever after. Second, why reading a long prompt gets disproportionately
expensive: one specific step, attention, has a cost that grows with the *square* of the prompt's
length, while the rest of the model grows only in a straight line with it. Both fall out directly
from how the model is built, and both are visible by the last section.

This is a self-contained tour of the model itself. If you also want the serving side, how a GPU
spends its time and memory turning these forward passes into a live service, the
[first primer](https://mapathak-commits.github.io/inference-wall/articles/primer/) covers that and
treats the forward pass as a black box this post opens.

## The shape of one forward pass

Start from the top, before any detail. To produce the next token, the model does three things
in order:

1. **Turn each input token into a vector.** A token, a short chunk of text, is converted into a
   list of a few thousand numbers. From here on the model works only on these vectors, never on
   words or letters directly.
2. **Push the vectors through a stack of identical blocks.** Each block reads the current vectors
   and rewrites them, adding a little more information about what each token means *in the context
   of the others* and what is likely to come next. This is where nearly all the weights, and
   nearly all the time, go.
3. **Turn the last vector into a guess at the next token.** After the final block, the vector
   sitting at the most recent position is converted into a score for every possible next token,
   and the highest-scoring candidates are the model's prediction.

That is the entire forward pass. The two phases you may have heard of are just two ways of running
it: **prefill** runs all three steps over every prompt token at once, and **decode** runs them
over a single new token at a time while reusing stored work for the rest. Everything below is a
zoom into step 2, because that is the step with the structure worth understanding. We build up one
block, then stack it.

## From tokens to vectors: the embedding

The model keeps a large table with one vector for every token in its vocabulary, learned during
training. That vector is the token's **embedding**: a fixed-length list of a few thousand numbers
that stands in for the token. Turning the prompt into vectors is a table lookup, one embedding
fetched per token, with nothing computed yet.

One property of that vector matters for everything that follows: its length never changes as it
moves through the model. It enters the first block as a few-thousand-number vector, leaves the
same length, enters the next block the same length, and comes out of the last block still the same
length. A block never grows or shrinks the vector; it only **rewrites the numbers in place**,
adjusting them so they carry a bit more meaning. So picture one vector per token, handed from
block to block, revised at each step and passed along. That handoff is the thread the rest of this
post follows.

## What a transformer block does

Right after the lookup, a token's vector depends only on the token itself. The vector for "bank"
is identical whether the sentence is about a river or a loan. But predicting the next token
requires knowing which meaning is in play, and that is fixed entirely by the surrounding tokens.
So a block's job is to update each token's vector using information from the other tokens, and it
does this in two distinct operations run back to back:

1. **Attention** is the only operation in the whole model that moves information *between* token
   positions. It replaces each token's vector with a mixture that draws in information from the
   earlier tokens relevant to it. Everywhere else, positions are processed in isolation; attention
   is where they interact.
2. A **feed-forward network** then processes each token's vector *on its own*, with no reference to
   any other position: the same small two-layer network applied independently at every position.
   This is where most of the model's weights sit, and where the model does the bulk of its
   per-token computation on the context attention just gathered.

The order is deliberate. Attention first collects the relevant context into each token's vector;
the feed-forward network then transforms that now-contextual vector. A block is exactly this pair,
**attention then feed-forward**, and it is the unit that repeats. The next two sections take each
operation in turn.

## Attention: a weighted average, steered by the tokens themselves

Here is the whole operation in one sentence, then the parts. **Attention rewrites each token's
vector as a weighted average of vectors drawn from the earlier tokens, where each token decides
for itself how much weight to put on each of the others.** The only real question is where those
weights come from, and that is what the query, key, and value are for.

From each token's current vector, the block computes three new vectors by multiplying it against
three separate learned weight matrices:

- the **query**: a description of what this token is looking for in the tokens before it,
- the **key**: a description of what this token contains, in the same terms a query is written in,
  so that queries and keys can be compared,
- the **value**: the information this token will contribute to any token that attends to it.

To update token number 50, the block takes token 50's **query** and compares it against the
**key** of every token from 1 to 50. The comparison is a dot product, which is large when two
vectors point in similar directions and small when they do not, so it measures how well token 50's
query lines up with each earlier token's key. That produces one raw score per earlier token: how
relevant is that token to what token 50 is looking for.

Those raw scores are not yet usable as averaging weights. They can be negative, and they do not
sum to anything in particular. **Softmax** is the step that fixes both: it exponentiates each score,
which forces it positive, then divides by the total, which makes the scores sum to one. What comes
out is a set of proportions, so the row now reads as something like "70% of my attention on this
token, 20% on that one, the rest spread thin." A useful side effect of exponentiating is that it
exaggerates gaps: a clearly-best match dominates the average while weak matches contribute almost
nothing. Softmax is there because a weighted average needs weights that are positive and sum to
one, and it is the standard way to turn arbitrary scores into exactly that.

The last step is the average itself. The block takes each earlier token's **value** vector, scales
it by that token's softmax weight, and adds them all up. The result is one blended vector, made
mostly of the values of the tokens token 50 found most relevant, and that blend is written back as
token 50's updated vector. Query to score, softmax to weight, value to average: each token asks
what it is looking for and pulls back a weighted blend of what the earlier tokens contain. This is
the operation the [transformer paper](https://arxiv.org/abs/1706.03762) introduced, and its title
is the claim itself, that attention is enough to let tokens share information.

### Why this is exactly what the KV cache stores

Look at what updating a token needs from the past: the **keys** of the earlier tokens, to score
them, and their **values**, to average them. It never needs their queries. A token's query is used
only to update that same token, never to be looked at from elsewhere. And the key and value a token
produces at a given block do not change once computed: token 50's key at block 3 is the same
whether the sequence is 51 tokens long or 5,000.

So the model computes each token's key and value once, the first time it processes that token, and
keeps them. That store is the **KV cache**. When the model later generates token 5,000 it does not
rerun tokens 1 through 4,999; it forms only the new token's query and looks up the 4,999 keys and
values already saved. Store the keys and values, discard the queries, never recompute the past. The
cache is not a bolt-on optimization; it falls straight out of how attention is defined.

### More than one at a time: attention heads

A block does not run a single query-key-value comparison. It runs several in parallel, each with
its own query, key, and value matrices, called attention **heads**. One head might learn to track
the immediately preceding word, another the last time the subject was mentioned, another matching
brackets or quotation marks. Each head does its own scoring and averaging over the whole sequence,
the results are concatenated, and the vector moves on. The reason it matters for cost: the KV cache
holds a separate key and value for *every head of every block*, which is why cached tokens add up
in memory as quickly as they do.

## The feed-forward network: computing on what attention gathered

After attention, each token's vector carries information about its context. The second operation
in the block is a **feed-forward network**: two large weight matrices with a simple nonlinear
function between them, applied to each token's vector independently. It takes the contextual vector
attention produced and transforms it, position by position, with no further mixing between
positions.

There is less to describe here, and that is the point. If attention is where a token collects what
it needs from its neighbors, the feed-forward network is where it computes on what it collected.
This half holds the large majority of the model's weights, so on a real GPU it is where most of
the work of a forward pass ends up. The one distinction to keep is against attention: this
operation is strictly per-token, where attention was strictly about tokens interacting.

## Stacking blocks

A model is a stack of these blocks, one after another. There is no variety in the wiring: every
block is built identically, attention then feed-forward. Small models stack a dozen or so; a
7B-class model has around thirty; the largest current LLMs stack a hundred or more. What differs
between blocks is the learned weights, and so what each block does to the vector. Early blocks tend
to resolve local, grammatical structure; later blocks assemble longer-range meaning.

Each block reads the vectors the previous block wrote and edits them a little further. One detail
keeps the repetition from washing out: a block *adds* its result into the vector rather than
replacing it, so what an early block established survives to the end unless a later block
deliberately overwrites it. That is what lets dozens of rounds of editing accumulate into a
sharper and sharper representation instead of blurring into noise.

## Turning the last vector into the next token

After the final block, the vector at the most recent position holds a heavily revised
representation of that token in its full context. Turning it into an actual next token is what the
phrase **next-token prediction** names, and it is a smaller step than it sounds.

The model multiplies that final vector by one last large matrix, which produces a single number
for *every* token in the vocabulary: a score for how well each one fits as the continuation. A
softmax turns those scores into a probability for each candidate, the same trick as before, used
here to turn scores into a probability distribution rather than averaging weights. That
distribution is the model's entire output: not a word, but a probability spread across the whole
vocabulary. "The model predicts the next token" means exactly this, a ranked list of candidates
with probabilities attached.

Picking an actual token from that distribution is a separate, cheap step called **sampling**, and
it is where knobs like *temperature* live: take the single most probable token, or roll a weighted
die over the top few. The token that comes out is fed back in as the newest input, and the whole
pass runs again for the token after it. That feedback loop, one full forward pass per output
token, is the decode loop, now with its insides visible.

## The two phases, from the inside

With the block open, the difference between reading a prompt and generating an answer comes down to
one thing, and it is all about attention:

- **Prefill** reads the whole prompt at once, so every one of the N prompt tokens forms a query and
  scores it against the keys of all the others in the same pass. That is N queries against up to N
  keys: an N-by-N grid of scores. Double the prompt and that grid quadruples. This is the one part
  of the model whose cost grows with the *square* of the sequence length.
- **Decode** generates one token at a time, so each step forms exactly *one* query and scores it
  against all the keys stored so far. One row, not a grid. Decode's attention cost grows only in a
  straight line with how deep the sequence is, but it pays that cost again on every single token it
  emits.

Everything else in a block, the three projections, the feed-forward network, the final scoring, is
the same per-token work regardless of how long the sequence is. Attention is the single operation
whose cost depends on sequence length: quadratic while reading a prompt, linear while writing an
answer. That is the fact behind the second promise at the top of this post.

## This is the basic version; real models add to it

Everything above is the plain transformer, and it is the right skeleton to carry in your head. But
no production LLM is exactly this. Real models keep the skeleton, attention then feed-forward,
repeated, and add refinements at nearly every step. A few of the common ones, so the names are not
a surprise when you meet them:

- **Position information.** The attention described above has no notion of token order; scoring a
  query against a key does not care which came first. Real models inject order separately, most
  often with rotary position encodings, so the model knows token 3 from token 300.
- **Cheaper keys and values.** Grouped-query attention lets several heads share one set of keys and
  values, which shrinks the KV cache substantially with little quality loss, and is why modern
  long-context models are practical to serve at all.
- **More feed-forward, used selectively.** Mixture-of-experts replaces the single feed-forward
  network with many, and routes each token to just a few of them, so the model can hold far more
  weights than any one token pays to use.
- **Cheaper mixing in some blocks.** Some architectures replace full attention in a fraction of
  their blocks with a mixing step whose cost stays fixed as the sequence grows, trading a little of
  attention's reach for an escape from its quadratic cost. These hybrid and linear-attention
  designs are a bet that long context is worth restructuring for.
- **Normalization.** Small normalization steps sit around each operation to keep the numbers
  well-behaved so the model trains stably.

None of these change the picture this post drew; they are variations on it. For the full catalog, a
survey like [Zhao et al., 2023](https://arxiv.org/abs/2303.18223) walks through them.

## What you can now see

- A forward pass is: embed each token into a vector, push the vectors through a stack of identical
  blocks that rewrite them, then turn the last vector into a probability over the next token.
- A block runs two operations in order: **attention**, the only place tokens interact, and a
  **feed-forward network**, which processes each token on its own and holds most of the weights.
- Attention rewrites a token's vector as a **weighted average of value vectors**, with weights set
  by scoring the token's **query** against every earlier token's **key** and passing the scores
  through a softmax. The **KV cache** exists because a token's key and value never change once
  computed, so they are stored and reused while the query is discarded.
- The model's output is a **probability over the whole vocabulary**, and picking a token from it is
  a separate, cheap sampling step whose result is fed back to start the next pass.

That is the whole model: embed, a stack of blocks that mix across tokens and then compute per
token, and a final scoring into the next token. Keep this picture for any time you need to reason
about what an LLM is actually computing, rather than treating it as an oracle that turns prompts
into text.

---

**Read next:** for the serving side, how these forward passes become a live service on a GPU,
see [the first primer: how an LLM actually serves a request](https://mapathak-commits.github.io/inference-wall/articles/primer/)

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions, findings,
and conclusions expressed herein are solely my own and do not necessarily represent the views,
policies, or endorsements of my current or past employers.*
