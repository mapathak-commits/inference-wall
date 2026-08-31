*A second primer for "The Inference Wall". The
[first primer](https://mapathak-commits.github.io/inference-wall/articles/primer/) built the
machine the series runs on, but it kept one box shut: it said each token gets multiplied
"through every weight matrix, layer by layer," and that the model produces a key and a value
per token, without ever saying what those are or what they are for. This primer opens that box.
Like the first, it has no benchmarks and no surprises; it explains the mechanism the later
posts measure, and stops before any of the findings.*

*Manas Pathak · September 1, 2026*

# A second primer: what actually happens inside the model

The first primer treated the model as a pile of weight matrices you stream through the GPU to
produce a token, and treated the forward pass as a black box that does the streaming. That is
the right picture for reasoning about *bytes*, which is what most of this series is about. But
two of the series' load-bearing facts come from inside the box, and they are easier to trust
once you have seen the machinery: that decode reuses a **key** and a **value** per past token
instead of reprocessing the conversation, and that one part of the work grows with the
*square* of the context while everything else grows linearly. This primer is the tour of the
inside. It is still not math-heavy, and you still do not need to know CUDA; you need to be able
to picture what a single one of those "layers" is actually doing.

## A token is a list of numbers, and it stays that shape the whole way through

Before anything happens, each token is turned into a vector, a fixed-length list of a few
thousand numbers, by a lookup table called the **embedding**. Every token in the vocabulary
has its own learned vector, and "embedding the prompt" is just fetching one vector per token.
That is the model's native material: not words, but vectors.

Hold onto one fact about that vector, because the whole architecture rests on it. Its length
never changes. The token goes into the first layer as a vector of, say, a few thousand numbers,
comes out the same length, goes into the next layer the same length, and emerges from the last
layer still the same length. A layer does not grow the vector or shrink it; it **rewrites it in
place**, nudging those numbers so they carry a little more meaning about what this token is and
what should come after it. Picture a single lane that the token's vector rides from the first
layer to the last, getting revised at each stop. Researchers call that lane the **residual
stream**, and it is the backbone everything else attaches to.

## One layer, half one: attention, where a token looks at the others

A layer has two halves. The first half is **attention**, and it is the only place in the entire
model where tokens are allowed to look at each other. Everywhere else, each token is processed
on its own; here, and only here, a token gathers information from the tokens that came before
it. This is the half the first primer kept shut, so we open it slowly.

When a token's vector enters the attention half, the layer multiplies it by three different
learned weight matrices to produce three new vectors from it. These are the **query**, the
**key**, and the **value**, or Q, K, and V. Three views of the same token, each for a different
job:

- the **query** (Q) is what this token is *looking for* in the tokens before it,
- the **key** (K) is what a token *offers* to anyone looking, its advertisement,
- the **value** (V) is what a token actually *hands over* once it has been chosen.

The mechanism is a directed lookup. To revise token number 50, the model takes token 50's
**query** and compares it against the **key** of every token from 1 to 50, one comparison per
earlier token. Each comparison is a single number, a score, high when this token's query and
that token's key point in similar directions, which is the model's learned way of saying "that
earlier token is relevant to this one." Those scores are then squashed into positive weights
that sum to one (the **softmax** step: it turns raw scores into a set of proportions, so the
scores become "60% of my attention here, 30% there, 10% spread over the rest"). Finally the
model takes each earlier token's **value** vector, multiplies it by that token's weight, and
adds them all up. The result is one blended vector, mostly the values of the tokens this token
found relevant, and *that* is what gets written back into token 50's lane in the residual
stream. Query to find, key to match, value to fetch: a token asks a question and pulls in a
weighted blend of the answers the earlier tokens offer.

This is the machinery the [transformer paper](https://arxiv.org/abs/1706.03762) introduced, and
its title is the whole idea: attention is all you need to let tokens share information.

### Now the KV cache makes sense

The first primer asserted that the model stores a key and a value per token and reuses them, and
asked you to take it on faith. Here is why the trick works and why it is exactly K and V that get
stored.

Look again at what revising a token needs from the past. It needs the **keys** of the earlier
tokens (to score them) and their **values** (to blend them). It does *not* need their queries;
a token's query is only ever used to revise that token itself, never to look at it. And a
crucial point: the key and value a token produces in a given layer *never change* once computed.
Token 50's key in layer 3 is the same whether the conversation is 51 tokens long or 5,000. So
the model computes each token's K and V once, the first time it sees that token, and files them
away. That filing cabinet is the **KV cache**. When the model later generates token 5,000, it
does not reprocess tokens 1 through 4,999; it forms only the new token's query and looks up the
4,999 keys and values already sitting in the cache. Store K and V, skip the queries, never
recompute the past: that is the whole cache, and now you can see it falls straight out of how
attention is defined.

### Many heads, one layer

One detail, because the later posts and the traces mention it. A layer does not run a single
query-key-value lookup; it runs several in parallel, each with its own Q, K, and V matrices,
called attention **heads**. One head might learn to look at the immediately preceding word,
another at the last mention of the subject, another at matching brackets. Each head does the
find-match-fetch dance over the whole context on its own, the results are concatenated, and the
lane moves on. For counting bytes, what matters is that the KV cache holds a key and a value for
every head of every layer, which is why the first primer's "around 130 KB per cached token"
adds up as fast as it does.

## The other half: the part that does the thinking, alone

The second half of a layer is a plain **feed-forward network**, an MLP: two big weight matrices
with a nonlinearity between them, applied to each token's lane on its own with no looking
sideways. If attention is where a token *gathers* information from its neighbors, the MLP is
where it sits with what it gathered and *computes* on it. This half holds the large majority of
the model's weights, and it is where most of the streaming from the first primer actually goes.
It is deliberately dull mechanically; the interesting structural fact is only that it is
per-token and independent, the opposite of attention.

So a full layer is: gather across tokens (attention), then think per token (MLP), each writing
its result back into the same fixed-length lane. That pair, attention then MLP, is one
**transformer block**, and it is the unit that stacks.

## Stacking blocks: the same operation, over and over, on a slowly sharpening vector

The model this series uses has 32 such blocks in a row. There is no variety in the *wiring* from
block to block; block 12 has the exact same structure as block 3. What differs is the learned
weights and, as a result, what each block does to the lane. Early blocks resolve local and
syntactic structure, later ones assemble longer-range meaning, and by construction each block
reads the residual stream the previous block wrote and edits it a little more. Because each block
only ever adds its result into the lane rather than replacing it, information laid down by an
early block survives all the way to the end unless a later block chooses to overwrite it, which
is what keeps 32 rounds of editing from washing out into noise.

One honest complication, since the series' rig runs into it directly. The clean "every block is
attention then MLP" picture is the canonical transformer, and it is the right thing to hold in
your head, but modern models economize. The 4B model here is a *hybrid*: only some of its blocks
run the full query-over-all-keys attention above; the rest use a cheaper mixing step whose cost
stays fixed as the context grows instead of growing with it. That distinction does nothing to
the intuition you need here, so the primer keeps the canonical block, but it is the seed of a
real result later in the series, where the mix of expensive and cheap blocks turns out to change
where the wall is.

## From the last block to the next token: what "predict" actually means

After the 32nd block, the lane for the most recent token holds a heavily-revised vector. Turning
that into an actual next token is the step the acronym NTP, **next-token prediction**, names, and
it is smaller than it sounds.

The model multiplies that final vector by one last big matrix, the **unembedding**, which is
essentially the embedding table run in reverse: it produces one number for *every* token in the
vocabulary, a score for how well that token fits as the continuation. A softmax turns those
scores into a probability for each possible next token. And that is the model's entire output: not
a word, but a probability distribution over the whole vocabulary. The words "the model predicts
the next token" mean exactly this, a ranked list of candidates with probabilities, nothing more.

Choosing an actual token from that distribution is a separate, cheap step called **sampling**,
and it is where the knobs like *temperature* live: take the single highest-probability token, or
roll a weighted die over the top few. The model proposes a distribution; sampling disposes. The
token that comes out is then fed back in as the next input, and the whole 32-block sweep runs
again for the token after it. That feedback loop, one full pass per output token, is the decode
loop from the first primer, now with its insides visible.

## The two phases, seen from inside

With the machinery open, the first primer's two phases sharpen into one clean distinction about
attention:

- **Prefill** processes the whole prompt at once, so every one of the N prompt tokens forms a
  query and looks back at all the others in the same pass. That is N queries against up to N
  keys, an N-by-N block of scores. Double the prompt and you quadruple that block. This is the
  one part of the model that grows with the *square* of the context, and it is why the first
  primer noted a prompt does "roughly N-by-N work."
- **Decode** processes one token at a time, so each step forms exactly *one* query and scores it
  against all the cached keys so far. One row, not a grid. Decode's attention grows only linearly
  with how deep the conversation is, but it pays that cost again on every single token it emits.

Everything else in a block, the three projections, the MLP, the unembedding, is the same
per-token weight-streaming the first primer already described. Attention is the one piece whose
cost depends on how much context there is, quadratically while reading a prompt and linearly
while writing an answer. That single fact is what one of the later posts turns a dial to expose;
this primer just wanted you to see where in the machine it comes from.

## What you can now see that you could not before

Nothing here changes the first primer's frame, that inference is mostly about moving bytes; it
fills in the shapes those bytes have:

- A token is a fixed-length vector that rides one lane, the residual stream, from the first block
  to the last, getting revised in place at each block and never changing length.
- A block has two halves: **attention**, the only place tokens look at each other, and an
  **MLP**, where each token computes on its own and where most of the weights live.
- Attention works by three views of each token, **query** to look, **key** to be matched,
  **value** to be fetched, and the **KV cache** exists precisely because a token's key and value
  never change once computed, so they are stored and reused while its query is thrown away.
- The model's actual output is a **probability over the whole vocabulary**, produced by one final
  unembedding matrix; picking a token from it is a separate, cheap sampling step, and the picked
  token is fed back to start the next pass.

Keep the first primer's picture for reasoning about throughput and memory; keep this one for the
moment a post starts talking about attention scaling, split-KV, or why a hybrid model dodges a
wall. Both are the same machine, described at two zoom levels.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
