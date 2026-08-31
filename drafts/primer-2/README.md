*A second primer for "The Inference Wall". The
[first primer](https://mapathak-commits.github.io/inference-wall/articles/primer/) built the
machine the series runs on, but it kept one box shut: it said each token gets multiplied
"through every weight matrix, layer by layer," and that the model produces a key and a value
per token, without saying what those are or what they are for. This primer opens that box. Like
the first, it has no benchmarks and no surprises; it explains what a forward pass is really
doing, and stops before any of the findings.*

*Manas Pathak · September 1, 2026*

# A second primer: what actually happens inside the model

The first primer said that producing a token means streaming the model's weights through the
GPU, and it left the forward pass, the thing that does that streaming, as a black box. For
reasoning about *bytes*, which is most of this series, that black box is fine. But a couple of
the series' facts come from inside it, and they are easier to believe once you have seen what
is in there: that decode reuses a stored **key** and **value** per past token instead of
rereading the whole conversation, and that one piece of the work grows with the *square* of the
context while everything else grows in a straight line.

Two things to set expectations before we start. First, this is the stripped-down version. Real
models pile on refinements the basics below leave out, things like rotary position encodings,
grouped-query attention, mixture-of-experts layers, and various normalization tricks; if you
want the full catalog, a survey like
[Zhao et al., 2023](https://arxiv.org/abs/2303.18223) walks through them. The skeleton here is
enough to understand every result in this series, and the refinements are variations on it, not
replacements. Second, none of this is math-heavy. You do not need to know CUDA or read a single
equation. You need to be able to picture what one pass through the model does.

## The shape of one forward pass

Start from the top, before any detail. To produce the next token, the model does three things
in order:

1. **Turn each input token into a vector.** A token, a chunk of text, is first converted into a
   list of a few thousand numbers. From here on the model works only on these vectors, never on
   words or letters directly.
2. **Push the vectors through a stack of identical blocks.** This model has 32 of them, stacked
   one after another. Each block reads the current vectors and rewrites them, adding a little
   more information about what each token means *in the context of the others* and what is
   likely to come next. This is where nearly all the weights, and nearly all the time, go.
3. **Turn the last vector into a guess at the next token.** After the final block, the vector
   sitting at the most recent position is converted into a score for every possible next token,
   and the highest-scoring ones are the model's prediction.

That is the entire forward pass. The two phases from the first primer are just two ways of
running it: **prefill** runs all three steps over every prompt token at once, and **decode**
runs them over a single new token at a time while reusing stored work for the rest. Everything
below is a zoom into step 2, because that is the step with the interesting structure. We will
build up one block, then note that the model is just 32 of them in a row.

## From tokens to vectors: the embedding

The model keeps a big table with one vector for every token in its vocabulary, learned during
training. That vector is the token's **embedding**: a fixed-length list of a few thousand
numbers that stands in for the token. Turning the prompt into vectors is just a lookup, one
embedding fetched per token, nothing computed.

One property of that vector matters for everything that follows: its length never changes as it
moves through the model. It enters the first block as a few-thousand-number vector, leaves the
same length, enters the next block the same length, and comes out of the last block still the
same length. A block never grows or shrinks the vector; it only **rewrites the numbers in
place**, nudging them to carry a bit more meaning. So picture one vector per token, handed from
block to block, revised at each step and passed along. That handoff is the thread the rest of
this primer follows.

## Inside a block: gather, then think

Why does a block need any structure at all? Because of a problem the embedding alone cannot
solve. Right after the lookup, a token's vector depends only on the token itself. The vector
for "bank" is the same whether the sentence is about a river or a loan. But to predict what
comes next, the model has to know *which* meaning is in play, and that depends entirely on the
surrounding tokens. So each token's vector has to be updated to reflect its neighbors before it
is any use.

A block does that in two steps, and the split is the one structural idea worth holding onto:

- **First, gather.** Let each token look at the earlier tokens and pull in information from the
  ones that are relevant to it. This step, and only this step, lets tokens see each other. It is
  called **attention**.
- **Then, think.** Take each token's now context-aware vector and run it through some further
  computation on its own, no looking sideways. This is a plain **feed-forward** step, and it is
  where most of the model's weights live.

Gather across tokens, then compute per token. That pair, attention followed by feed-forward, is
one **transformer block**. The next two sections open up each step.

## Attention: how a token looks at the others

Suppose you wanted to build the "gather" step yourself. For a given token, you need three
things. You need a way for that token to express *what it is looking for* in the earlier
tokens. You need a way for every earlier token to advertise *what it has to offer*. And you need
the actual *content* each earlier token would pass along if chosen. Three roles, one per need.

Attention builds exactly those three, and this is where the model's matrices finally earn their
keep. When a token's vector enters the attention step, it is multiplied by three separate
learned weight matrices, producing three new vectors from it:

- the **query** (Q): what this token is looking for,
- the **key** (K): what this token offers to anyone looking, its advertisement,
- the **value** (V): the content this token hands over if it is chosen.

The gather itself is a directed lookup. To update token number 50, the model takes token 50's
**query** and compares it against the **key** of every token from 1 to 50. Each comparison is a
single number, a score, and it is high when the query and that key point in similar directions,
which is the model's learned way of saying "that earlier token is relevant here." The scores are
then turned into positive weights that add up to one. That last step is the **softmax**; it just
converts raw scores into proportions, so they read as something like "60% of my attention on
this token, 30% on that one, the rest spread thin." Finally the model takes each earlier token's
**value** vector, scales it by that token's weight, and adds them all up. The result is one
blended vector, mostly the values of the tokens this one found relevant, and that blend is what
gets written back into token 50's vector.

Query to find, key to match, value to fetch: a token poses a question and pulls back a weighted
blend of what the earlier tokens offer in answer. This is the idea the
[transformer paper](https://arxiv.org/abs/1706.03762) introduced, and its title says the whole
thing, that attention is all you need to let tokens share information.

### Why this is exactly what the KV cache stores

The first primer asked you to take on faith that the model stores a key and a value per token
and reuses them. Now you can see why, and why it is those two and not the query.

Look at what updating a token needs from the past: the **keys** of the earlier tokens, to score
them, and their **values**, to blend them. It never needs their queries. A token's query is used
only to update that same token, never to look at it from somewhere else. And the key and value a
token produces at a given block do not change once computed: token 50's key at block 3 is the
same whether the conversation is 51 tokens long or 5,000.

So the model computes each token's K and V once, the first time it sees that token, and keeps
them. That store is the **KV cache**. When the model later generates token 5,000, it does not
rerun tokens 1 through 4,999; it forms only the new token's query and looks up the 4,999 keys
and values already saved. Store the keys and values, throw the queries away, never recompute the
past. The cache is not a clever add-on; it falls straight out of how attention is defined.

### More than one at a time: attention heads

One refinement, because the traces later in the series name it. A block does not run a single
query-key-value lookup. It runs several in parallel, each with its own Q, K, and V matrices,
called attention **heads**. One head might learn to track the immediately preceding word,
another the last time the subject was mentioned, another matching brackets. Each head does its
own find-match-fetch over the whole context, the results are stitched together, and the vector
moves on. For counting bytes, the thing to remember is that the KV cache holds a key and a value
for *every head of every block*, which is why the first primer's "around 130 KB per cached
token" piles up as fast as it does.

## The other half: the feed-forward step

After gathering, each token's vector carries information about its context. The second half of
the block is where the model actually does something with it: a **feed-forward** step, two large
weight matrices with a simple nonlinear function between them, applied to each token's vector on
its own with no looking sideways.

There is less to say about it mechanically, and that is the point. If attention is where a token
*collects* what it needs from its neighbors, the feed-forward step is where it *works on* what it
collected. This half holds the large majority of the model's weights, so it is where most of the
byte-streaming from the first primer actually happens. The one fact worth keeping is the
contrast with attention: this step is per-token and independent, where attention was all about
tokens looking at each other.

## Stacking blocks

The model is 32 of these blocks in a row, and there is no variety in the wiring: block 12 is
built exactly like block 3. What differs is the learned weights, and so what each block does to
the vector. Early blocks tend to settle local, grammatical structure; later ones assemble
longer-range meaning. Each block reads the vectors the previous block wrote and edits them a
little further, and because a block adds its result into the vector rather than replacing it,
what an early block established survives to the end unless a later block deliberately overwrites
it. That is what keeps 32 rounds of editing from blurring into noise.

One honest wrinkle, since the series' rig runs straight into it. The clean "every block is
attention then feed-forward" picture is the textbook transformer, and it is the right thing to
carry in your head. But this series' 4B model is a *hybrid*: only some of its blocks run the
full query-against-all-keys attention above, while the rest use a cheaper mixing step whose cost
stays fixed as the context grows instead of growing with it. That changes none of the intuition
here, so the primer sticks with the textbook block, but it is the seed of a real result later in
the series, where the mix of expensive and cheap blocks turns out to move the wall.

## Turning the last vector into the next token

After the 32nd block, the vector at the most recent position holds a heavily revised
representation of that token in context. Turning it into an actual next token is what the phrase
**next-token prediction** names, and it is a smaller step than it sounds.

The model multiplies that final vector by one last large matrix, which produces a single number
for *every* token in the vocabulary, a score for how well each one fits as the continuation. A
softmax turns those scores into a probability for each candidate. And that is the model's whole
output: not a word, but a probability spread across the entire vocabulary. "The model predicts
the next token" means exactly this, a ranked list of candidates with probabilities attached.

Picking an actual token from that list is a separate, cheap step called **sampling**, and it is
where knobs like *temperature* live: take the single most probable token, or roll a weighted die
over the top few. The token that comes out is then fed back in as the next input, and the whole
32-block pass runs again for the token after it. That feedback loop, one full pass per output
token, is the decode loop from the first primer, now with its insides visible.

## The two phases, from the inside

With the block open, the first primer's two phases sharpen into a single clean difference, and it
is all about attention:

- **Prefill** reads the whole prompt at once, so every one of the N prompt tokens forms a query
  and looks back at all the others in the same pass. That is N queries against up to N keys, an
  N-by-N grid of scores. Double the prompt and that grid quadruples. This is the one part of the
  model that grows with the *square* of the context, and it is why the first primer said a prompt
  does "roughly N-by-N work."
- **Decode** writes one token at a time, so each step forms exactly *one* query and scores it
  against all the keys saved so far. One row, not a grid. Decode's attention grows only in a
  straight line with how deep the conversation is, but it pays that cost again on every single
  token it emits.

Everything else in a block, the three projections, the feed-forward step, the final scoring, is
the same per-token weight-streaming the first primer already described. Attention is the one
piece whose cost depends on how much context there is: quadratic while reading a prompt, linear
while writing an answer. That single fact is what one of the later posts turns a dial to expose;
this primer just wanted you to see where in the model it comes from.

## What you can now see that you could not before

Nothing here changes the first primer's frame, that inference is mostly about moving bytes. It
fills in the shapes those bytes have:

- A forward pass is: embed each token into a vector, push the vectors through a stack of
  identical blocks that rewrite them, then turn the last vector into a probability over the next
  token.
- A block has two halves: **attention**, the only place tokens look at each other, and a
  **feed-forward** step, where each token is processed on its own and where most of the weights
  live.
- Attention makes three vectors from each token, a **query** to look, a **key** to be matched,
  and a **value** to be fetched, and the **KV cache** exists precisely because a token's key and
  value never change once computed, so they are saved and reused while the query is thrown away.
- The model's actual output is a **probability over the whole vocabulary**, and picking a token
  from it is a separate, cheap sampling step whose result is fed back to start the next pass.

Keep the first primer's picture for reasoning about throughput and memory. Keep this one for the
moment a post starts talking about attention scaling, split-KV, or why a hybrid model dodges a
wall. Both describe the same model, at two different zoom levels.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
