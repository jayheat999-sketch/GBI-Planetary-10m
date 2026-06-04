"""
gbi_v10_planets.py — 8 probes, each with its OWN fully learnable rotation
generator. Center pinned. "Cut the wire and let them loose like planets."

The fork (v9 -> v10)
--------------------
v9 freed the probe POSITIONS but confined all 8 movers to ONE shared 2-plane (a
single fixed J1,J2 per head) -- they could only trace a single planar ellipse;
beads on one wire. v10 cuts the wire: each mover probe gets its OWN full skew-
symmetric generator G_k per head, with EVERY entry learnable. Each probe chooses
both its rotation plane and its angle, independently of the others. The pinned
center (identity) is the only fixed reference.

Mechanics
---------
A skew-symmetric matrix is fixed by its strictly-upper-triangular entries; those
are the free Parameter (gen_tri, shape [8, heads, hd*(hd-1)/2]). Inside forward,
G = U - U^T is rebuilt (exactly skew, so matrix_exp(G) is exactly a rotation) and
applied to Q for that probe. Random small init -> 8 distinct small rotations at
step 0 (not a stencil, not copies of each other).

NO GUARDS -- on purpose. If a probe's generator decays toward 0 it collapses to
identity (a duplicate of the pinned center); if generators converge to each other
the probes collapse together; if they blow up the rotation degenerates. None of
these are prevented. Each is a RESULT: the telemetry records ||G||_F per probe
(gen_norm), the max rotation angle, pairwise generator distance, and how many
probes have collapsed to center. This run is pure exploration -- let them go and
read the files for whether their choices mattered (watch w_m: if the probes do
something elaborate but w_m stays low, the model isn't using them).

The experiment output
---------------------
gbi_v10_probes_step*.json every EXPORT_EVERY: per probe, gen_norm (rotation
amount; ->0 = decayed to center), max/mean rotation angle, dist_to_center,
nearest other probe + distance (collapse detection), and global mean pairwise
generator distance + count collapsed to center.

Inherited from v7m/v9
---------------------
  - v5 skeleton: pre-norm LayerNorm, learned positional embeddings, tied
    embed/unembed (NB: tied + LayerNorm head, no temperature -> a structural CE
    floor independent of this fork).
  - Detuning gate (cQED analog): per-token Hopfield matter/cavity coupling,
    UNCHANGED -- center via w_c, the 8 movers via w_m.
  - Data: SimpleStories via the RibbonStreamer (continuous filaments) + MorphTokenizer.

Data setup
----------
Put SimpleStories in ./SimpleStories/ (the HuggingFace parquet shards, which carry
a `story` column; a .txt corpus with stories separated by <|endoftext|> also works).
Both stream through the RibbonStreamer.

Checkpointing
-------------
Atomic saves; separate dir (./gbi_exports_v10). Resume with --resume-from <path>:
a v10 checkpoint is exact; a v9/v7m checkpoint warm-starts only the shared weights
(embed, q/k/v/out, gates) and leaves the generators fresh.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import json
import argparse
import sys
import traceback
import numpy as np
import re
import collections
import pyarrow.parquet as pq


# -- Configuration --
# v11 fork: SAME architecture as v10 planets, pointed at SimpleStories instead of
# TinyStories. The architecture is frozen on purpose -- this run exists to test
# whether the 8 self-organized probe generators are INVARIANT to the corpus
# (the "Geometry Is Real" thesis on a new axis) or reorganize under real long-
# range syntax. Resume a v10 TinyStories checkpoint with --resume-from to start
# the probes from their trained arrangement; load_checkpoint warm-starts the
# shared weights and keeps the generators exact.
DATA_DIR     = "./SimpleStories"
VOCAB_PATH   = "./simplestories_word_vocab.json"   # SEPARATE cache -- do NOT clobber the TinyStories vocab (needed for the invariance comparison)
# SimpleStories is clean, ASCII-only, model-generated short stories -- much closer
# to TinyStories than to web text. Drop both gates back toward TinyStories levels
# (no hundreds of thousands of typos / URLs / hapax junk to defend against).
WORD_MIN_COUNT = 2          # keep morpheme tokens seen >= this many times
MORPH_KNOWN_MINCOUNT = 5    # a leftover stem must appear >= this often as a whole word to be a valid split target
MORPH_MIN_STEM = 2          # never strip an affix if the leftover stem would be shorter than this
# HARD CAP on vocab size. SimpleStories has a much smaller true vocabulary than
# web text, but keep the cap as a safety bound on the tied embedding matrix AND
# the per-batch logits tensor. 32k will almost certainly not bind here.
VOCAB_CAP = 32000
VOCAB_SIZE   = None         # set at runtime from the built/loaded word vocab
EMBED_DIM    = 384
NUM_HEADS    = 6           # head_dim = 64 (matches v5)
THETA_DEG    = 5.0
CARDINAL_STRETCH = 5.0
USE_DETUNING = True        # per-token detuning gate (cQED bandgap analog). Init = no-op.

# -- Parallax-inspired diagnostics (Zuo et al. 2026, arXiv:2605.29157) -----------
# Each is PURE TELEMETRY: it reads tensors already computed in forward, adds a
# per-token signal to the export, and NEVER touches the loss or the forward
# output. All default OFF -> flipping none of them changes the run at all.
# Flip ONE at a time to read it cleanly. They cost a little extra compute only
# when return_signals=True (i.e. only on EXPORT_EVERY steps), so training-step
# throughput is unchanged regardless.
#
#  PROBE_COR        correction-to-output ratio: ||mix|| / ||center_out||, the
#                   single "is the geometry doing work or has it collapsed to
#                   plain attention" number. Their COR (eq.16). >1 = movers
#                   dominate; ->0 = inert, you're running a softmax baseline.
#  PROBE_ALIGN      magnitude-vs-direction split (their CPA, eq.17). How much of
#                   the mover output lies ALONG the center output's direction vs
#                   orthogonal to it. High norm + low alignment = the movers are
#                   spending themselves in directions that don't move the answer.
#  PROBE_GATE_STATS distribution of the detuning gates, not just the mean. Their
#                   gating analysis found a learned correction gate collapsing
#                   to ~0.26 under one optimizer (= switched off). Logs the
#                   fraction of tokens with w_m / w_c below GATE_OFF_THRESH so a
#                   silent gate-collapse shows up as a number.
#  PROBE_SIGN       sign distribution of the per-token mover contribution along
#                   the center direction. Their correction lets effective weights
#                   go NEGATIVE (actively subtract a token). Tests whether your
#                   signed differential geometry already routes subtractively
#                   "for free" -- fraction of tokens where mix opposes center.
#  PROBE_SINK       attention-sink ratio: weight mass on the first token. Their
#                   correction branch absorbs the routing the softmax dumps on
#                   token 0. Logs token-0 attention_norm share so you can see if
#                   the movers pull mass off the sink.
PROBE_COR        = True
PROBE_ALIGN      = True
PROBE_GATE_STATS = True
PROBE_SIGN       = True
PROBE_SINK       = True
GATE_OFF_THRESH  = 0.26    # below this a gate counts as "switched off" (their value)
BATCH_SIZE   = 8           # halved from 16 for 16GB XPU headroom (9 SDPA probe passes + AdamW state + matrix_exp). The experiment measures probe geometry, not throughput.
SEQ_LEN      = 512
STEPS        = 100000
LR           = 1e-3
EXPORT_EVERY = 500
CHECKPOINT_EVERY = 1000
SAMPLE_EVERY = 1000        # print a sample story this often
SAMPLE_PROMPT = "Once upon a time"  # narrative opener (SimpleStories are short stories)
# SimpleStories parquet has NO per-document `score` column (unlike FineWeb-Edu),
# so the score filter is disabled by default. Left wired in so a FineWeb/scored
# corpus still filters if you point --data-dir at one and pass --score-min.
SCORE_MIN = 0.0


# -- Morpheme tokenizer (corpus-validated affix stripping) --
# Words/numbers + space-aware punctuation (as before), but each WORD is further
# split at REAL linguistic seams: a prefix/suffix is peeled off ONLY IF the
# leftover stem is itself a word seen in the corpus (>= MORPH_KNOWN_MINCOUNT).
# That single rule kills false splits ("this" -/-> "thi"+s, "string" stays whole)
# without any external dictionary. Splits are pure string slices -> lossless;
# decode is concatenation using the '#' markers (prefix "un#", suffix "#ing").
# Honest limitation: words needing a respelling to segment ("running"->run+ing)
# stay whole, because respelling would break lossless concatenation.
class MorphTokenizer:
    WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+| ?[^\sA-Za-z\d] ?")
    WORDONLY_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+")
    SPECIALS = ["<pad>", "<unk>", "<eot>"]
    PREFIXES = ["counter", "inter", "trans", "under", "super", "anti", "semi",
                "over", "fore", "mis", "dis", "non", "pre", "sub", "out", "mid",
                "re", "un", "de", "en", "im", "in", "ex"]
    SUFFIXES = ["ization", "fulness", "iousness", "ational", "ation", "ition",
                "ments", "ities", "nesses", "ingly", "edly", "ment", "ness",
                "less", "ful", "able", "ible", "tion", "sion", "ship", "ward",
                "wise", "hood", "ing", "ity", "ous", "ive", "ize", "ise", "ant",
                "ent", "est", "ery", "ish", "ed", "er", "ly", "es", "ic", "al", "s"]

    def __init__(self):
        self.itos = []
        self.stoi = {}
        self.known = set()        # lowercase whole words that can serve as stems
        self.pad, self.unk, self.eot = 0, 1, 2
        self._pre = sorted(self.PREFIXES, key=len, reverse=True)
        self._suf = sorted(self.SUFFIXES, key=len, reverse=True)

    @property
    def eot_token(self):
        return self.eot

    def _pieces(self, text):
        return self.WORD_RE.findall(text)

    @staticmethod
    def _is_word(tok):
        return bool(MorphTokenizer.WORDONLY_RE.fullmatch(tok))

    def segment_word(self, w):
        """Split one word into [prefix#?, stem, #suffix...] using corpus-validated
        affix stripping. Pure slicing -> prefix + stem + suffixes == w (lossless)."""
        if not w.isalpha() or len(w) < 2 * self.MIN_STEM:
            return [w]
        prefix_tok = None
        low = w.lower()
        for p in self._pre:
            if low.startswith(p):
                resid = w[len(p):]
                if len(resid) >= self.MIN_STEM and resid.lower() in self.known:
                    prefix_tok = w[:len(p)] + "#"
                    w = resid; low = w.lower()
                    break
        suffixes = []
        while True:
            stripped = False
            for s in self._suf:
                if low.endswith(s) and len(w) - len(s) >= self.MIN_STEM:
                    resid = w[:len(w) - len(s)]
                    if resid.lower() in self.known:
                        suffixes.append("#" + w[len(w) - len(s):])
                        w = resid; low = w.lower()
                        stripped = True
                        break
            if not stripped:
                break
        out = ([prefix_tok] if prefix_tok else []) + [w] + list(reversed(suffixes))
        return out

    # MIN_STEM read from module config at build/encode time
    @property
    def MIN_STEM(self):
        return MORPH_MIN_STEM

    def _morphemes(self, text):
        out = []
        for piece in self._pieces(text):
            if self._is_word(piece):
                out.extend(self.segment_word(piece))
            else:
                out.append(piece)      # punctuation (space-aware), unchanged
        return out

    def build(self, piece_counts, min_count):
        """piece_counts: Counter of word/punct pieces (pre-morpheme). Builds the
        known-word set, then the morpheme vocabulary."""
        self.known = {w.lower() for w, c in piece_counts.items()
                      if self._is_word(w) and c >= MORPH_KNOWN_MINCOUNT}
        morph_counts = collections.Counter()
        for piece, c in piece_counts.items():
            if self._is_word(piece):
                for m in self.segment_word(piece):
                    morph_counts[m] += c
            else:
                morph_counts[piece] += c
        self.itos = list(self.SPECIALS)
        # sorted by descending frequency, so applying the cap == keeping the
        # most frequent tokens and routing the long tail to <unk>.
        n_specials = len(self.SPECIALS)
        for tok, c in sorted(morph_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= min_count:
                self.itos.append(tok)
            if VOCAB_CAP is not None and len(self.itos) >= VOCAB_CAP:
                break
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return morph_counts

    def encode_ordinary(self, text):
        s, unk = self.stoi, self.unk
        return [s.get(t, unk) for t in self._morphemes(text)]

    def encode_ordinary_batch(self, texts):
        return [self.encode_ordinary(t) for t in texts]

    def encode(self, text):
        return self.encode_ordinary(text)

    def decode(self, ids):
        out = ""
        prev_was_prefix = False
        glue_next = False          # set by an opening punct that wants the next word attached
        for i in ids:
            i = int(i)
            if i == self.eot:
                out += "\n"; prev_was_prefix = False; glue_next = False; continue
            if i == self.pad:
                continue
            tok = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            has_alnum = any(ch.isalnum() for ch in tok)
            if not has_alnum:
                # punctuation token (carries its own spaces)
                if out.endswith(" ") and tok.startswith(" "):
                    out += tok[1:]
                else:
                    out += tok
                # an opener like ' "' or ' (' ends in a non-space punct char ->
                # the following word should glue (no inserted space)
                glue_next = (len(tok) > 0 and not tok.endswith(" "))
                prev_was_prefix = False
                continue
            if tok.endswith("#"):              # prefix -> starts a new word
                if out and out[-1] not in " \n" and not glue_next:
                    out += " "
                out += tok[:-1]
                prev_was_prefix = True
            elif tok.startswith("#"):          # suffix -> glue to current word
                out += tok[1:]
                prev_was_prefix = False
            else:                              # bare stem / whole word
                if prev_was_prefix or glue_next:
                    out += tok                 # glue onto prefix or opener
                else:
                    if out and out[-1] not in " \n":
                        out += " "
                    out += tok
                prev_was_prefix = False
            glue_next = False
        return out.strip()

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "known": sorted(self.known),
                       "morph": True}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.itos = d["itos"]
        self.known = set(d.get("known", []))
        self.stoi = {t: i for i, t in enumerate(self.itos)}


def build_or_load_vocab(data_dir, tokenizer, min_count, cache_path, rebuild=False):
    """Scan corpus once, count word/punct pieces, build the corpus-validated
    morpheme vocabulary. Reports fertility (avg morphemes per word). Sets VOCAB_SIZE."""
    global VOCAB_SIZE
    if os.path.exists(cache_path) and not rebuild:
        tokenizer.load(cache_path)
        VOCAB_SIZE = len(tokenizer.itos)
        print(f"Loaded morpheme vocab: {VOCAB_SIZE} tokens, "
              f"{len(tokenizer.known)} known stems  ({cache_path})")
        return

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")) +
                   glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No .parquet or .txt files in {data_dir} to build a vocab from.")

    print(f"Building morpheme vocab from {len(files)} file(s) "
          f"(min_count={min_count}, known_mincount={MORPH_KNOWN_MINCOUNT})...")
    piece_counts = collections.Counter()
    # Reuse the SAME score-aware iterator the trainer uses, so the vocab is built
    # from exactly the documents that will be trained on (no drift between the two
    # paths, and the FineWeb-Edu score filter is applied here too).
    for text in _iter_corpus_texts(files):
        piece_counts.update(tokenizer._pieces(text))

    morph_counts = tokenizer.build(piece_counts, min_count)

    # fertility / segmentation stats (token-weighted over words)
    word_tokens = sum(c for w, c in piece_counts.items() if tokenizer._is_word(w))
    morph_tokens = sum(c for w, c in piece_counts.items()
                       if tokenizer._is_word(w)
                       for _ in tokenizer.segment_word(w))
    distinct_words = [w for w in piece_counts if tokenizer._is_word(w)]
    split_distinct = sum(1 for w in distinct_words if len(tokenizer.segment_word(w)) > 1)
    fertility = morph_tokens / max(word_tokens, 1)
    VOCAB_SIZE = len(tokenizer.itos)

    print(f"  distinct word/punct pieces: {len(piece_counts)}   "
          f"known stems (>= {MORPH_KNOWN_MINCOUNT}): {len(tokenizer.known)}")
    # how much corpus mass falls outside the (capped) vocab -> <unk>
    kept = set(tokenizer.itos)
    in_vocab_mass, total_mass = 0, 0
    for piece, c in piece_counts.items():
        for mtok in (tokenizer.segment_word(piece) if tokenizer._is_word(piece) else [piece]):
            total_mass += c
            if mtok in kept:
                in_vocab_mass += c
    unk_rate = 100.0 * (1.0 - in_vocab_mass / max(total_mass, 1))
    capped = (VOCAB_CAP is not None and VOCAB_SIZE >= VOCAB_CAP)
    print(f"  morpheme vocab size (incl 3 specials): {VOCAB_SIZE}"
          f"{'  [CAPPED at VOCAB_CAP]' if capped else ''}")
    print(f"  token mass routed to <unk> by the cap/min_count: {unk_rate:.2f}%  "
          f"(high == cap too aggressive; raise VOCAB_CAP)")
    print(f"  fertility (avg morphemes / word): {fertility:.3f}  "
          f"(1.0 = no splitting; BPE on English ~1.2)")
    print(f"  distinct words that segment: {split_distinct}/{len(distinct_words)} "
          f"({100*split_distinct/max(len(distinct_words),1):.1f}%)")
    print(f"  lossless: yes (pure slicing).  saved -> {cache_path}")
    tokenizer.save(cache_path)


def _iter_corpus_texts(files):
    """Yield raw document strings from parquet/txt files, in order. The
    RibbonStreamer drives this generator; it tokenizes lazily, so this only
    needs to hand back raw text the same way the corpus is laid out (parquet
    'text' column rows, or .txt stories separated by the <|endoftext|> marker).

    FineWeb-Edu: if SCORE_MIN > 0 and the parquet has a `score` column, documents
    scoring below SCORE_MIN are skipped. Falls back to reading every row if the
    column is absent (so TinyStories parquet still works unchanged)."""
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".parquet":
            pf = pq.ParquetFile(fp)
            names = pf.schema_arrow.names
            # SimpleStories stores the document in a `story` column; TinyStories /
            # FineWeb use `text`. Pick whichever this parquet actually has.
            text_col = 'story' if 'story' in names else 'text'
            has_score = SCORE_MIN > 0 and ("score" in names)
            cols = [text_col, 'score'] if has_score else [text_col]
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                texts = tbl.column(text_col).to_pylist()
                if has_score:
                    scores = tbl.column('score').to_pylist()
                    for text, sc in zip(texts, scores):
                        if sc is not None and sc >= SCORE_MIN:
                            yield text
                else:
                    for text in texts:
                        yield text
                del tbl
        elif ext == ".txt":
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for chunk in iter(lambda: f.read(8_000_000), ""):
                    for story in chunk.split("<|endoftext|>"):
                        if story.strip():
                            yield story


# -- Streaming loader: BATCH_SIZE persistent continuous filaments (RibbonStreamer) --
# Ported from gbi_v7m_jamo.py. Replaces the old 100M-token ring buffer, which was
# the real startup bottleneck: it tokenized an ENTIRE file (hundreds of thousands
# of stories through the pure-Python morpheme tokenizer) before step 0, and zeroed
# a 400MB tensor. The RibbonStreamer does NO up-front fill: it tokenizes only
# seq_len+1 tokens per batch row to prime (~8k tokens total, instant), then tops up
# lazily one document at a time as each window slides forward. Constant memory;
# loops the corpus forever. Each batch row i is a continuous continuation of
# filament i's document stream (sequential windows, not random samples), so token
# i+1 really follows token i -- better for the narrative/coherence goal than the
# old random windows that also straddled story boundaries.
#
# DATA SOURCE UNCHANGED: this still reads the TinyStories parquet/txt corpus and
# tokenizes with the MorphTokenizer (encode_ordinary). Only the SLIDING/BATCHING
# mechanism was swapped from the ring buffer to filaments.
class TinyStoriesStreamer:
    def __init__(self, data_dir, seq_len, batch_size, device, tokenizer,
                 autoload=True):
        # `autoload` accepted for call-site compatibility; priming is cheap and
        # always done in the constructor regardless.
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.enc = tokenizer
        self.eot = tokenizer.eot_token

        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.parquet")) +
            glob.glob(os.path.join(data_dir, "*.txt"))
        )
        if not self.files:
            raise FileNotFoundError(
                f"No .parquet or .txt files in {data_dir}\n"
                f"  Download SimpleStories from HuggingFace 'SimpleStories/SimpleStories'\n"
                f"  (the parquet shards under data/, with a `story` column) and place\n"
                f"  them in {data_dir}/  -- e.g.:\n"
                f"    huggingface-cli download SimpleStories/SimpleStories \\\n"
                f"      --repo-type dataset --local-dir {data_dir}\n"
                f"  A .txt corpus with stories separated by <|endoftext|> also works."
            )

        # file_idx is exposed for checkpoint compatibility. The infinite text
        # generator advances it as it walks files.
        self.file_idx = 0
        self.text_generator = self._stream_texts()

        # each batch slot has its own persistent token buffer (filament)
        self.filaments = [[] for _ in range(batch_size)]
        self._prime_filaments()

    def _stream_texts(self):
        """Infinite generator over document text; loops the corpus forever and
        keeps self.file_idx roughly tracking progress for checkpointing."""
        while True:
            for text in _iter_corpus_texts(self.files):
                yield text
            # finished a full pass; reset and loop
            self.file_idx = 0

    def _refill(self, i):
        # Tokenize one document at a time (MorphTokenizer.encode_ordinary) and
        # seal each with eot, until this filament has a full window + 1.
        while len(self.filaments[i]) < self.seq_len + 1:
            story_text = next(self.text_generator)
            tokens = self.enc.encode_ordinary(story_text) + [self.eot]
            self.filaments[i].extend(tokens)

    def _prime_filaments(self):
        print("Priming filaments...")
        for i in range(self.batch_size):
            self._refill(i)

    # kept as a no-op for call-site compatibility (old code called this to fill
    # the ring; the ribbon primes lazily, so there is nothing to do here).
    def _load_next_file(self):
        return

    def get_batch(self):
        """Pull a sequential window from each filament, slide each forward by one
        SEQ_LEN. Each row is a continuous continuation of that filament."""
        x_batch, y_batch = [], []
        for i in range(self.batch_size):
            self._refill(i)
            seq = self.filaments[i][:self.seq_len + 1]
            x_batch.append(torch.tensor(seq[:-1], dtype=torch.long))
            y_batch.append(torch.tensor(seq[1:], dtype=torch.long))
            # slide the persistent window forward by one full sequence
            self.filaments[i] = self.filaments[i][self.seq_len:]
        x = torch.stack(x_batch).to(self.device)
        y = torch.stack(y_batch).to(self.device)
        return x, y


# -- Orthogonal generators (from v6P) --
def make_orthogonal_generators(head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    def random_skew():
        M = torch.randn(head_dim, head_dim, generator=g)
        return 0.5 * (M - M.T)
    J1, J2 = random_skew(), random_skew()
    J2 = J2 - ((J1 * J2).sum() / (J1 * J1).sum()) * J1
    return (J1 / J1.norm() * math.sqrt(head_dim),
            J2 / J2.norm() * math.sqrt(head_dim))

def rotation_from_combined(J1, J2, alpha, beta):
    return torch.linalg.matrix_exp(alpha * J1 + beta * J2)


# -- v9 FREE-PROBE interferometer: 9 LEARNABLE rotated probes, no collocation --
# This is the "option 3" fork. The thesis-bearing fixed stencil is GONE:
#
#   v4/v6P/v7m: 9 sample points at FIXED, symmetric offsets (center + 4 corners +
#   4 stretched cardinals). Those locations are hard-coded precisely so that
#   (R-L)/theta IS a derivative and the 4-corner stencil IS a mixed partial -- the
#   differential-geometry readout is an ASSUMPTION baked into the geometry.
#
#   v9 free-probe: the 8 NON-CENTER probes can MOVE. Each has its own learnable
#   offset (alpha_k, beta_k) in the per-head rotation plane spanned by the (fixed)
#   generators J1, J2. The rotations R_k = exp(alpha_k J1 + beta_k J2) are rebuilt
#   INSIDE forward, so gradient flows to (alpha_k, beta_k) and the data chooses
#   where the probes sit. The center probe is PINNED at (0,0) (the one fixed
#   reference; everything else is free, exactly as requested).
#
# CONSEQUENCE FOR THE READOUT: once the points can move, "grad_x", "lapl_y",
# "torsion", "kappa", "tau" are NO LONGER DEFINED -- those labels only meant
# "derivative" because the points sat at known symmetric locations. So the
# Lagrange collocation solve is REMOVED. The 9 samples are instead mixed by a
# small learned head (one dim x dim projection per probe + a shared per-mix
# nonlinear correction), gated by the SAME detuning weights as before (center via
# w_c, the 8 movers via w_m) so the gate instrumentation still works.
#
# THE EXPERIMENT IS THE OFFSET POSITIONS. After training, read where the 8 probes
# migrated (export_offsets_json / the offsets in telemetry):
#   - back toward a symmetric, near-orthogonal arrangement -> the geometry is an
#     ATTRACTOR the model rediscovers from scratch (strongest form of the thesis);
#   - collapsed onto each other / one axis -> the 2D stereoscope was over-
#     parameterized;
#   - structured-but-non-derivative -> the model wants geometry, but not the
#     specific stencil that was imposed.
# Loss at this scale is weak evidence; the positions are the measurement.
class InterferometerBlockV7(nn.Module):
    def __init__(self, dim, heads, theta_deg, cardinal_stretch=5.0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.theta_rad = theta_deg * math.pi / 180.0
        self.cardinal_stretch = cardinal_stretch
        self.n_probe = 9          # 1 pinned center + 8 movers
        self.n_mover = self.n_probe - 1

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # -- One full dim x dim projection PER PROBE -------------------------------
        # No derivative labels anymore: each of the 9 probes is just a rotated
        # attention readout, and the mixer is free to combine them however lowers
        # loss. proj[0] is the CENTER probe (init = identity). The 8 mover
        # projections are SMALL-RANDOM (std 0.02), NOT zero.
        #
        # Why not zero-init the movers (the v7m no-op convention)? Because the
        # offsets reach the loss only through their probe's projection
        #   offsets -> R_k -> movers[k] -> proj[k+1](movers[k]) -> mix,
        # so a zero proj[k+1] gives the offsets ZERO gradient at step 0 -- it
        # starves the exact parameter this experiment exists to measure. There is
        # no checkpoint to resume byte-identically here (v9 is a fresh fork), so
        # the no-op-resume property buys nothing. Small-random projections give the
        # offsets a live gradient from the first step; the trade is that step-0
        # output is not exactly plain attention (it's attention + a small mover
        # perturbation), which is fine for a from-scratch run.
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False)
                                   for _ in range(self.n_probe)])
        with torch.no_grad():
            self.proj[0].weight.copy_(torch.eye(dim))      # center = identity
            for k in range(1, self.n_probe):
                nn.init.normal_(self.proj[k].weight, std=0.02)   # movers: live gradient

        # -- FULLY LEARNABLE per-probe rotation generators (v10 "planets") ---------
        # v9 confined all 8 movers to ONE shared 2-plane (fixed J1,J2 per head) and
        # let each move only its (alpha,beta) position in that plane -> the probes
        # could only ever trace a single planar ellipse; they were beads on one wire.
        # v10 cuts the wire. Each mover gets its OWN full skew-symmetric generator
        # G_k per head, and EVERY entry of G_k is learnable. The probe is free to
        # pick BOTH its rotation plane (any 2-plane, or a more general rotation) AND
        # its angle -- planets, not beads. Center is still pinned (identity, no
        # generator).
        #
        # Parameterization: a skew-symmetric matrix is fixed by its strictly-upper-
        # triangular entries; we store those as a free Parameter and rebuild
        # G = U - U^T inside forward, so G is EXACTLY skew (matrix_exp(G) is exactly
        # a rotation in SO(head_dim)) while all head_dim*(head_dim-1)/2 dofs move.
        #
        # NO GUARDS. If a generator collapses toward 0 (probe -> identity ->
        # duplicate of the center) or grows large (near-degenerate rotation), that
        # is a RESULT, not something to prevent -- it tells you the probe didn't
        # want to be its own thing. Random small init so the 8 probes start as 8
        # DISTINCT small rotations (not the stencil, not each other).
        hd = self.head_dim
        self.tri_idx = torch.triu_indices(hd, hd, offset=1)   # (2, n_upper)
        n_upper = self.tri_idx.shape[1]
        # one upper-triangular vector per (mover, head); small random init
        self.gen_tri = nn.Parameter(0.05 * torch.randn(self.n_mover, heads, n_upper))
        # h/H kept only for telemetry labels / the offsets exporter's "init" column
        h = self.theta_rad / 2.0
        H = self.cardinal_stretch * h
        self.h_corner, self.h_card = h, H

        # -- Shared per-mix nonlinear correction -----------------------------------
        # The per-probe projections are all linear and summed, so the mix collapses
        # to a single linear map. One shared nonlinear correction (GELU between two
        # dim x dim maps) lifts that ceiling. nl_out ZERO-INIT -> exactly 0 at step
        # 0 -> step-0 output is still pure center-identity attention.
        self.nl_in  = nn.Linear(dim, dim, bias=True)
        self.nl_out = nn.Linear(dim, dim, bias=False)
        self.nl_act = nn.GELU()
        with torch.no_grad():
            self.nl_out.weight.zero_()

        # -- Detuning gate (cQED bandgap analog), unchanged from v7m ---------------
        # center ("cavity") gated by w_c; the 8 movers ("continuum") gated by w_m.
        # Both init ~1 (no-op). See v7m for the full rationale.
        self.use_detuning = USE_DETUNING
        if self.use_detuning:
            self.gate_proj = nn.Linear(dim, 1, bias=True)
            self.detune_temp = nn.Parameter(torch.ones(1))
            with torch.no_grad():
                self.gate_proj.weight.normal_(std=1e-3)
                self.gate_proj.bias.zero_()
            self.gate_proj_center = nn.Linear(dim, 1, bias=True)
            self.detune_temp_center = nn.Parameter(torch.ones(1))
            with torch.no_grad():
                self.gate_proj_center.weight.normal_(std=1e-3)
                self.gate_proj_center.bias.zero_()

        # Readout direction (fixed random), kept for signed telemetry continuity.
        proj = torch.randn(dim, generator=torch.Generator().manual_seed(9999))
        self.register_buffer("readout_dir", proj / proj.norm())

    def _rotations(self):
        """Build the 8 mover rotation stacks from the CURRENT learnable generators.
        Each (mover, head) has its own full skew-symmetric G = U - U^T; matrix_exp
        gives an exact rotation and gradient flows to every entry. fp32 for exp.
        Returns R of shape (n_mover, heads, head_dim, head_dim)."""
        hd = self.head_dim
        i, j = self.tri_idx[0], self.tri_idx[1]
        U = torch.zeros(self.n_mover, self.heads, hd, hd,
                        device=self.gen_tri.device, dtype=torch.float32)
        U[:, :, i, j] = self.gen_tri.float()          # strictly-upper entries
        G = U - U.transpose(-1, -2)                   # exactly skew-symmetric
        R = torch.linalg.matrix_exp(G)
        return R

    def _attn_with(self, R_k, Q, K, V):
        """SDPA with queries rotated by R_k. R_k: (heads, head_dim, head_dim)."""
        Q_rot = torch.einsum("hde,bshe->bshd", R_k.to(Q.dtype), Q).transpose(1, 2)
        A = F.scaled_dot_product_attention(Q_rot, K, V, is_causal=True)
        return A.transpose(1, 2).contiguous()

    def forward(self, x, return_signals=False):
        B, S, C = x.shape
        Q = self.q_proj(x).view(B, S, self.heads, self.head_dim)
        K = self.k_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)

        # CENTER probe: identity rotation (pinned at (0,0)) -> plain causal attention
        center = self._attn_with(
            torch.eye(self.head_dim, device=x.device, dtype=Q.dtype)
                 .unsqueeze(0).expand(self.heads, -1, -1),
            Q, K, V).view(B, S, C)

        # MOVER probes: rotations built from the current learnable offsets
        R = self._rotations()                              # (8, heads, hd, hd)
        movers = [self._attn_with(R[k], Q, K, V).view(B, S, C)
                  for k in range(self.n_mover)]

        # Per-mix nonlinear correction (shared), zero at init via nl_out
        def nl(v):
            return self.nl_out(self.nl_act(self.nl_in(v)))

        # Per-token coupling weights (Hopfield pair), init ~1 (no-op)
        if self.use_detuning:
            delta_m = self.gate_proj(x).squeeze(-1)
            temp_m = self.detune_temp.clamp_min(1e-3)
            w_m = torch.exp(-(delta_m * delta_m) / (2.0 * temp_m * temp_m)).unsqueeze(-1).to(center.dtype)
            delta_c = self.gate_proj_center(x).squeeze(-1)
            temp_c = self.detune_temp_center.clamp_min(1e-3)
            w_c = torch.exp(-(delta_c * delta_c) / (2.0 * temp_c * temp_c)).unsqueeze(-1).to(center.dtype)
        else:
            w_m = w_c = None

        # center -> proj[0]; movers -> proj[1..8]. Shared nonlinear correction is
        # added once to each probe's projection (cheap, lifts the linear ceiling).
        center_out = self.proj[0](center) + nl(center)
        mix = movers[0].new_zeros(B, S, C)
        for k in range(self.n_mover):
            mix = mix + self.proj[k + 1](movers[k]) + nl(movers[k])

        if w_m is not None:
            out = w_c * center_out + w_m * mix
        else:
            out = center_out + mix
        out = self.out_proj(out)

        if not return_signals:
            return out

        # -- Telemetry: per-probe GENERATOR geometry is the experiment now --------
        # No (alpha,beta) anymore. The meaningful quantities for a free generator:
        #   gen_norm[k]  = ||G_k||_F summed/averaged over heads -> total rotation
        #                  "amount" the probe applies. -> 0 means the probe decayed
        #                  to identity (a duplicate of the pinned center); large
        #                  means a hard rotation.
        #   max_angle[k] = largest rotation angle (rad) of R_k, per head, from the
        #                  imaginary parts of its eigenvalues -> how far the probe
        #                  actually turns Q, independent of plane.
        # Computed on the detached generators (cheap; head_dim is small).
        Gdet = self.gen_tri.detach().float()                       # (8,heads,n_upper)
        gen_norm = (Gdet.pow(2).sum(-1) * 2.0).sqrt().mean(dim=1)  # (8,)  ||G||_F mean over heads
        with torch.no_grad():
            R_det = self._rotations()                              # (8,heads,hd,hd)
            # rotation angle per 2x2 block ~ from eigenvalues on unit circle
            ev = torch.linalg.eigvals(R_det)                       # (8,heads,hd) complex
            ang = torch.atan2(ev.imag.abs(), ev.real)              # true angle in [0, pi] -- no false pi/2 ceiling
            max_angle = ang.amax(dim=(-1, -2))                     # (8,)
        # pairwise distinctness: mean ||G_i - G_j|| over probe pairs (are the 8
        # probes DIFFERENT, or did they collapse onto each other?)
        Gflat = Gdet.reshape(self.n_mover, -1)
        pdist = torch.cdist(Gflat.unsqueeze(0), Gflat.unsqueeze(0)).squeeze(0)  # (8,8)
        mean_pair_dist = pdist.sum() / (self.n_mover * (self.n_mover - 1))

        c0d = center.detach().float()
        mover_norms = torch.stack([m.detach().float().norm(dim=-1) for m in movers], dim=0)

        signals = {
            "attention_norm": c0d.norm(dim=-1),                       # (B,S)
            "center_signed":  (c0d * self.readout_dir).sum(dim=-1),   # (B,S)
            "gen_norm":        gen_norm,                              # (8,) rotation amount per probe
            "gen_norm_mean":   gen_norm.mean().reshape(1),            # (1,)
            "max_angle":       max_angle,                            # (8,) max turn angle per probe
            "probe_pair_dist": mean_pair_dist.reshape(1),            # (1,) are probes distinct?
            "mover_norm_mean": mover_norms.mean(dim=(1, 2)),         # (8,) activation per probe
        }
        if self.use_detuning:
            signals["detuning"] = delta_m.detach().float()
            signals["coupling_w"] = w_m.squeeze(-1).detach().float()
            signals["detuning_center"] = delta_c.detach().float()
            signals["coupling_w_center"] = w_c.squeeze(-1).detach().float()

        # -- Parallax-inspired diagnostics (all per-token (B,S), all OFF by default) --
        # Computed only here (return_signals=True path), so training steps are
        # unaffected. Read from tensors already built above: center, mix,
        # center_out, w_m, w_c. fp32, detached.
        cen_f  = center.detach().float()           # (B,S,C) plain-attention (cavity) output
        mix_f  = mix.detach().float()              # (B,S,C) summed mover (continuum) contribution
        cout_f = center_out.detach().float()       # (B,S,C) center after its own proj+nl
        eps = 1e-6

        if PROBE_COR:
            # ||mix|| / ||center_out|| per token. Their COR. The headline
            # "are the movers doing work" scalar. (post-gate variant below)
            cor = mix_f.norm(dim=-1) / (cout_f.norm(dim=-1) + eps)        # (B,S)
            signals["cor"] = cor
            if w_m is not None:
                # gated COR: what actually reaches the output after w_m/w_c.
                gm = w_m.detach().float(); gc = w_c.detach().float()
                cor_g = (gm * mix_f).norm(dim=-1) / ((gc * cout_f).norm(dim=-1) + eps)
                signals["cor_gated"] = cor_g.squeeze(-1) if cor_g.dim() == 3 else cor_g

        if PROBE_ALIGN:
            # decompose mix into the part ALONG center's direction vs orthogonal.
            # alignment = |<mix, center>| / (||mix|| ||center||) in [0,1].
            # low alignment + high COR = movers active but pointing "sideways".
            dot = (mix_f * cen_f).sum(dim=-1)                             # (B,S)
            denom = mix_f.norm(dim=-1) * cen_f.norm(dim=-1) + eps
            align = (dot.abs() / denom)                                  # (B,S) in [0,1]
            # signed projection length of mix onto unit center direction
            proj_len = dot / (cen_f.norm(dim=-1) + eps)                  # (B,S)
            signals["mix_center_align"] = align
            signals["mix_proj_len"] = proj_len

        if PROBE_GATE_STATS and w_m is not None:
            # per-token "is this gate switched off" indicator (1.0 = off).
            gm = w_m.detach().float().squeeze(-1)                         # (B,S)
            gc = w_c.detach().float().squeeze(-1)                         # (B,S)
            signals["gate_m_off"] = (gm < GATE_OFF_THRESH).float()
            signals["gate_c_off"] = (gc < GATE_OFF_THRESH).float()

        if PROBE_SIGN:
            # does the mover contribution ADD to or SUBTRACT from the center
            # readout direction? sign of <mix, readout_dir>. 1.0 = subtractive.
            mix_along_readout = (mix_f * self.readout_dir).sum(dim=-1)    # (B,S)
            signals["mix_subtracts"] = (mix_along_readout < 0).float()
            signals["mix_along_readout"] = mix_along_readout

        if PROBE_SINK:
            # share of total attention_norm sitting on token 0 (the sink).
            an = cen_f.norm(dim=-1)                                       # (B,S)
            tot = an.sum(dim=-1, keepdim=True) + eps                      # (B,1)
            sink_share = an / tot                                        # (B,S); [:,0] is the sink
            signals["sink_share"] = sink_share

        return out, signals


# -- Feed Forward --
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# -- Top-level model: v5 skeleton (LayerNorm + pos-embed) + v7 interferometer --
class HolographicModelV7m(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Embedding(SEQ_LEN, EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.interferometer = InterferometerBlockV7(
            EMBED_DIM, NUM_HEADS, THETA_DEG, CARDINAL_STRETCH)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = MLP(EMBED_DIM)
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.unembed = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.embed.weight = self.unembed.weight

    def forward(self, idx, return_signals=False):
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_embed(pos)

        res = x
        x = self.ln1(x)
        if return_signals:
            attn_out, signals = self.interferometer(x, return_signals=True)
        else:
            attn_out, signals = self.interferometer(x, return_signals=False), None
        x = res + attn_out

        x = x + self.mlp(self.ln2(x))
        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits, signals


# -- Inline sample generation (watch coherence emerge) --
@torch.no_grad()
def generate_sample(model, enc, device, prompt=SAMPLE_PROMPT, max_n=150,
                    temperature=0.8, top_k=40):
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    eot_token = enc.eot_token  # Fetch the <|endoftext|> ID
    
    for _ in range(max_n):
        idx_cond = idx[:, -SEQ_LEN:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(idx_cond, return_signals=False)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Stop generating if the model decides the story is over
        if idx_next.item() == eot_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
        
    text = enc.decode(idx[0].tolist())
    model.train()
    return text
    

# -- Export telemetry --
# Per-token signals in v9 are just attention_norm and center_signed (the
# derivative/curvature channels are gone). The HEADLINE export is the probe
# offsets -- see export_offsets_json below, which is the experiment.

# signals that are per-token (B,S) and safe to average over the batch:
_PER_TOKEN_KEYS = ("attention_norm", "center_signed", "detuning",
                   "coupling_w", "detuning_center", "coupling_w_center",
                   # Parallax-inspired diagnostics (only present when their
                   # toggle is on; the exporter skips any key not in `signals`):
                   "cor", "cor_gated",
                   "mix_center_align", "mix_proj_len",
                   "gate_m_off", "gate_c_off",
                   "mix_subtracts", "mix_along_readout",
                   "sink_share")


def export_geometry_json(model, loader, device, path, n_batches=4):
    model.eval()
    accum = None
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = loader.get_batch()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, signals = model(x, return_signals=True)
            # only the genuinely per-token (B,S) signals get batch-averaged;
            # offset_* signals are (8,)/(1,) and are handled by export_offsets_json
            bm = {k: signals[k].float().mean(dim=0).cpu()
                  for k in _PER_TOKEN_KEYS if k in signals}
            if accum is None:
                accum = bm
            else:
                for k in accum: accum[k] += bm[k]
    for k in accum: accum[k] = (accum[k] / n_batches).tolist()
    n_tok = len(next(iter(accum.values())))
    tokens = {f"tok_{i:04d}": {k: accum[k][i] for k in accum} for i in range(n_tok)}
    out = {"tokens": tokens,
           "meta": {"source": "GBI_v10_planets",
                    "theta_deg": THETA_DEG,
                    "cardinal_stretch": CARDINAL_STRETCH,
                    "seq_len": SEQ_LEN}}
    with open(path, "w") as f: json.dump(out, f, indent=2)
    model.train()


def export_trajectory_json(model, loader, device, path, enc, seq_index=0):
    """Single-sequence per-token trajectory (no batch averaging). v9 has no
    grad_x/grad_y, so the old phase-from-gradient derivation is gone; we emit the
    per-token center readout signals directly. The figure-8 / phase analysis the
    fixed-stencil versions did is not defined here, by design."""
    model.eval()
    with torch.no_grad():
        x, _ = loader.get_batch()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, signals = model(x, return_signals=True)
    ids = x[seq_index].tolist()
    s = {k: signals[k][seq_index].float().cpu().numpy()
         for k in _PER_TOKEN_KEYS if k in signals}

    tokens = {}
    for i in range(len(ids)):
        rec = {"token_str": enc.decode([int(ids[i])])}
        for k in s:
            rec[k] = float(s[k][i])
        tokens[f"tok_{i:04d}"] = rec

    out = {"tokens": tokens,
           "meta": {"source": "GBI_v10_trajectory",
                    "sequence": True,
                    "averaged": False,
                    "n_tokens": len(ids),
                    "theta_deg": THETA_DEG,
                    "cardinal_stretch": CARDINAL_STRETCH,
                    "seq_len": SEQ_LEN}}
    with open(path, "w") as f: json.dump(out, f, indent=2)
    model.train()


def export_offsets_json(model, device, path):
    """THE EXPERIMENT. Each mover probe now has its OWN full rotation generator
    (v10 planets), so there is no (alpha,beta) position -- instead we dump, per
    probe: ||G||_F (rotation amount; ->0 means it decayed to identity == a copy of
    the pinned center), the max rotation angle, the dominant rotation plane, and
    how DISTINCT the 8 probes are from each other (did they spread, or collapse
    onto one another / onto the center?). Read this, not the loss."""
    blk = model.interferometer
    with torch.no_grad():
        Gtri = blk.gen_tri.detach().float().cpu()          # (8,heads,n_upper)
        R = blk._rotations().detach().cpu()                # (8,heads,hd,hd)
    hd = blk.head_dim
    gen_norm = (Gtri.pow(2).sum(-1) * 2.0).sqrt()          # (8,heads) ||G||_F per head
    gen_norm_mean = gen_norm.mean(dim=1)                   # (8,)
    # rotation angles per probe from eigenvalues of R (per head, then summarized)
    ev = torch.linalg.eigvals(R)                           # (8,heads,hd) complex
    ang = torch.atan2(ev.imag.abs(), ev.real)              # (8,heads,hd) true angle in [0, pi] -- no false pi/2 ceiling
    max_angle = ang.amax(dim=(-1, -2))                     # (8,)
    mean_angle = ang.mean(dim=(-1, -2))                    # (8,)
    # pairwise distinctness in generator space
    Gflat = Gtri.reshape(blk.n_mover, -1)
    pdist = torch.cdist(Gflat.unsqueeze(0), Gflat.unsqueeze(0)).squeeze(0)
    # distance of each probe's generator to ZERO (= to the pinned center / identity)
    dist_to_center = Gflat.norm(dim=-1)                    # same as ||G||_F flat

    probes = {}
    for k in range(blk.n_mover):
        # nearest OTHER probe, to see collapse onto a neighbour
        row = pdist[k].clone(); row[k] = float("inf")
        nn_dist, nn_idx = float(row.min()), int(row.argmin())
        probes[f"probe_{k}"] = {
            "gen_norm_mean": float(gen_norm_mean[k]),      # rotation amount (0 => identity/center)
            "gen_norm_min_head": float(gen_norm[k].min()),
            "gen_norm_max_head": float(gen_norm[k].max()),
            "max_angle_rad": float(max_angle[k]),
            "mean_angle_rad": float(mean_angle[k]),
            "dist_to_center": float(dist_to_center[k]),    # ||G_k|| (collapse-to-center if ->0)
            "nearest_probe": nn_idx,
            "nearest_probe_dist": nn_dist,                 # collapse-onto-neighbour if ->0
        }
    out = {"probes": probes,
           "meta": {"source": "GBI_v10_planets",
                    "n_mover": blk.n_mover,
                    "center": "pinned (identity, no generator)",
                    "head_dim": hd,
                    "mean_pairwise_gen_dist": float(pdist.sum() / (blk.n_mover*(blk.n_mover-1))),
                    "n_collapsed_to_center": int((dist_to_center < 1e-2).sum()),
                    "theta_deg": THETA_DEG,
                    "cardinal_stretch": CARDINAL_STRETCH}}
    with open(path, "w") as f: json.dump(out, f, indent=2)


# -- Checkpoint helpers --
CKPT_DIR = "./gbi_exports_v11_simplestories"   # v11 SimpleStories — separate dir, never collides with the v10 TinyStories or v11 FineWeb run
CKPT_PREFIX = "v11_simplestories_step"

def save_checkpoint(model, optimizer, step, loader, tag="regular"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    if tag == "regular":
        ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{step}.pt")
    else:
        ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{step}_{tag}.pt")
    tmp_path = ckpt_path + ".tmp"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loader_file_idx': loader.file_idx,
        'config': {
            'source': 'GBI_v10_planets',
            'VOCAB_SIZE': VOCAB_SIZE,
            'vocab_path': VOCAB_PATH,
            'EMBED_DIM': EMBED_DIM,
            'NUM_HEADS': NUM_HEADS,
            'THETA_DEG': THETA_DEG,
            'CARDINAL_STRETCH': CARDINAL_STRETCH,
            'SEQ_LEN': SEQ_LEN,
            'BATCH_SIZE': BATCH_SIZE,
            'USE_DETUNING': USE_DETUNING,
        },
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX
    return ckpt_path

def _step_of(p):
    stem = os.path.basename(p).replace(CKPT_PREFIX, "").replace(".pt", "")
    return int(stem.split("_")[0])

def prune_checkpoints():
    ckpts = sorted(
        glob.glob(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}*.pt")),
        key=_step_of,
    )
    tagged = ("_emergency", "_interrupt", "_final")
    regular = [p for p in ckpts if not any(t in p for t in tagged)]
    keep = set(regular[-3:])
    for p in regular:
        if _step_of(p) % 25000 == 0:
            keep.add(p)
    for p in regular:
        if p not in keep:
            try:
                os.remove(p)
                print(f"  [pruned] {os.path.basename(p)}")
            except OSError:
                pass

def load_checkpoint(model, optimizer, path, device):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # non-strict load. v9's parameter set differs from the fixed-stencil versions:
    # it has `offsets` and a per-probe `proj.0..8` ModuleList, and NONE of the old
    # proj_grad_x/proj_lapl_*/proj_kappa/proj_tau channels. Resuming a v9 checkpoint
    # is exact. Resuming a v7m/frenet checkpoint warm-starts only the shared pieces
    # (embed, q/k/v/out, gates, ln) and leaves the probes/offsets fresh at the
    # stencil init -- a legitimate way to start the probes from a trained attention.
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  [resume] fresh (not in ckpt): {missing}")
    if unexpected:
        print(f"  [resume] ignored (not in model): {unexpected}")
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except (ValueError, KeyError) as e:
        print(f"  [resume] optimizer state skipped (param set changed): {e}")
        print(f"  [resume] -> new params get fresh Adam moments; existing params "
              f"keep their weights but restart momentum. Use a short LR warmup if needed.")
    step = ckpt.get('step', 0)
    file_idx = ckpt.get('loader_file_idx', 0)
    cfg = ckpt.get('config', {})
    print(f"  step={step}, loader.file_idx={file_idx}")
    if cfg:
        for k in ('EMBED_DIM','NUM_HEADS','THETA_DEG','CARDINAL_STRETCH','SEQ_LEN'):
            if k in cfg and cfg[k] != globals()[k]:
                print(f"  [warn] config mismatch: {k} ckpt={cfg[k]} current={globals()[k]}")
    return step, file_idx


# -- Training loop --
def train(resume_from=None, min_count=WORD_MIN_COUNT, rebuild_vocab=False, steps=STEPS):
    device = torch.device(
        "xpu" if torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Workbench powered up on: {device}")

    # Build/load the word vocab FIRST — it sets VOCAB_SIZE, which the model needs.
    tokenizer = MorphTokenizer()
    build_or_load_vocab(DATA_DIR, tokenizer, min_count, VOCAB_PATH, rebuild=rebuild_vocab)

    model = HolographicModelV7m().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M  (vocab={VOCAB_SIZE})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    start_step = 0
    start_file_idx = 0
    if resume_from is not None and os.path.exists(resume_from):
        start_step, start_file_idx = load_checkpoint(
            model, optimizer, resume_from, device)

    print(f"\nInitializing DataLoader at file {start_file_idx}...")
    loader = TinyStoriesStreamer(
        DATA_DIR, SEQ_LEN, BATCH_SIZE, device, tokenizer, autoload=False)
    loader.file_idx = start_file_idx
    loader._load_next_file()

    print("\nStarting v10 PLANETS (8 probes, each its OWN learnable rotation "
          "generator; center pinned) on TinyStories:")
    print("  g_mean = mean rotation amount ||G||_F (->0 = probe decayed to the")
    print("  center); dist = mean pairwise generator distance (->0 = probes")
    print("  collapsed together). The probe geometry is the experiment, not loss.")
    print(f"{'Step':>6} | {'Loss':>6} | {'g_mean':>7} | {'dist':>7} | "
          f"{'g_min':>7} | {'g_max':>7}")
    print("-" * 72)

    enc = loader.enc
    last_step = start_step
    if start_step >= steps:
        print(f"  [warn] start_step ({start_step}) >= target steps ({steps}); "
              f"nothing to do. Pass --steps {start_step + 100000} to keep training.")
    try:
        for step in range(start_step, steps):
            last_step = step
            x, y = loader.get_batch()
            optimizer.zero_grad(set_to_none=True)

            return_signals = (step % 50 == 0)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, signals = model(x, return_signals=return_signals)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if return_signals:
                gn = signals["gen_norm"]                  # (8,) per-probe rotation amount
                g_mean = signals["gen_norm_mean"].item()
                dist = signals["probe_pair_dist"].item()
                print(f"{step:>6} | {loss.item():>6.3f} | {g_mean:>7.4f} | "
                      f"{dist:>7.4f} | {gn.min().item():>7.4f} | {gn.max().item():>7.4f}")

            if step > 0 and step % EXPORT_EVERY == 0:
                os.makedirs(CKPT_DIR, exist_ok=True)
                path = os.path.join(CKPT_DIR, f"gbi_v11_geometry_step{step}.json")
                export_geometry_json(model, loader, device, path)
                traj = os.path.join(CKPT_DIR, f"gbi_v11_trajectory_step{step}.json")
                export_trajectory_json(model, loader, device, traj, enc)
                # THE EXPERIMENT: per-probe generator geometry this step
                offp = os.path.join(CKPT_DIR, f"gbi_v11_probes_step{step}.json")
                export_offsets_json(model, device, offp)

            if step > 0 and step % SAMPLE_EVERY == 0:
                story = generate_sample(model, enc, device)
                print(f"\n  --- sample @ step {step} ---")
                print(f"  {story}\n  ---------------------------")
                if USE_DETUNING:
                    blk = model.interferometer
                    with torch.no_grad():
                        xg, _ = loader.get_batch()
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            _, sg = model(xg, return_signals=True)
                    wm = sg["coupling_w"].float()
                    wc = sg["coupling_w_center"].float()
                    print(f"  [gate matter] temp={blk.detune_temp.item():.3f}  "
                          f"w_m mean={wm.mean():.3f} min={wm.min():.3f} max={wm.max():.3f}")
                    print(f"  [gate cavity] temp={blk.detune_temp_center.item():.3f}  "
                          f"w_c mean={wc.mean():.3f} min={wc.min():.3f} max={wc.max():.3f}  "
                          f"(w_m=continuum, w_c=center; 1=full on)\n")
                else:
                    print()

                # -- Parallax-inspired diagnostic summary (only the toggled-on ones) --
                # Reuses `sg` from the gate block above when available; otherwise
                # does one extra signals pass. Each line prints only if its probe
                # is on, so a clean run shows nothing here.
                if PROBE_COR or PROBE_ALIGN or PROBE_GATE_STATS or PROBE_SIGN or PROBE_SINK:
                    if not USE_DETUNING:
                        with torch.no_grad():
                            xg, _ = loader.get_batch()
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                _, sg = model(xg, return_signals=True)
                    if PROBE_COR and "cor" in sg:
                        cor = sg["cor"].float()
                        line = f"  [COR] ||mix||/||center|| mean={cor.mean():.3f} min={cor.min():.3f} max={cor.max():.3f}"
                        if "cor_gated" in sg:
                            cg = sg["cor_gated"].float()
                            line += f"  | gated mean={cg.mean():.3f}"
                        print(line + "   (>1 movers dominate; ->0 inert)")
                    if PROBE_ALIGN and "mix_center_align" in sg:
                        al = sg["mix_center_align"].float()
                        pl = sg["mix_proj_len"].float()
                        print(f"  [ALIGN] |cos(mix,center)| mean={al.mean():.3f}  "
                              f"signed proj_len mean={pl.mean():+.3f}   (low align + high COR = sideways)")
                    if PROBE_GATE_STATS and "gate_m_off" in sg:
                        mo = sg["gate_m_off"].float().mean()
                        co = sg["gate_c_off"].float().mean()
                        print(f"  [GATE OFF<{GATE_OFF_THRESH:.2f}] matter={mo*100:.1f}% of tokens  "
                              f"cavity={co*100:.1f}%   (high % = branch being switched off)")
                    if PROBE_SIGN and "mix_subtracts" in sg:
                        sub = sg["mix_subtracts"].float().mean()
                        mar = sg["mix_along_readout"].float()
                        print(f"  [SIGN] mix subtracts on {sub*100:.1f}% of tokens  "
                              f"(mean along-readout={mar.mean():+.3f})   (subtractive routing 'for free')")
                    if PROBE_SINK and "sink_share" in sg:
                        ss = sg["sink_share"].float()
                        tok0 = ss[:, 0].mean()
                        print(f"  [SINK] token-0 attn share={tok0*100:.2f}%  "
                              f"(mean per-token={ss.mean()*100:.2f}%)   (lower = movers pull mass off sink)")
                    print()

            if step > 0 and step % CHECKPOINT_EVERY == 0:
                ckpt_path = save_checkpoint(model, optimizer, step, loader)
                print(f"  [checkpoint] {os.path.basename(ckpt_path)}  "
                      f"(file {loader.file_idx}/{len(loader.files)})")
                prune_checkpoints()

    except KeyboardInterrupt:
        print(f"\n[interrupt] Ctrl-C at step {last_step}. Saving emergency checkpoint...")
        try:
            ckpt_path = save_checkpoint(model, optimizer, last_step, loader, tag="interrupt")
            print(f"  saved: {ckpt_path}")
        except Exception as e:
            print(f"  failed to save interrupt checkpoint: {e}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[exception] {type(e).__name__} at step {last_step}: {e}")
        traceback.print_exc()
        try:
            ckpt_path = save_checkpoint(model, optimizer, last_step, loader, tag="emergency")
            print(f"  emergency checkpoint saved: {ckpt_path}")
        except Exception as e2:
            print(f"  failed to save emergency checkpoint: {e2}")
        raise

    final_path = save_checkpoint(model, optimizer, last_step, loader, tag="final")
    print(f"\nDone. Final checkpoint: {final_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume-from", default=None,
                   help="Path to a v9 checkpoint (exact) or a v7m/frenet "
                        "checkpoint (warm-starts shared weights; probes start fresh)")
    p.add_argument("--min-count", type=int, default=WORD_MIN_COUNT,
                   help="keep word tokens seen >= this many times "
                        "(1 for TinyStories, ~5 for FineWeb)")
    p.add_argument("--rebuild-vocab", action="store_true",
                   help="force rebuilding the word vocab even if cached")
    p.add_argument("--score-min", type=float, default=None,
                   help=f"FineWeb-Edu: skip documents with edu-score below this "
                        f"(default {SCORE_MIN}; 0 disables). Affects BOTH vocab "
                        f"building and training, so change it with --rebuild-vocab.")
    p.add_argument("--vocab-cap", type=int, default=None,
                   help=f"hard cap on vocab size; keeps the top-N most frequent "
                        f"morpheme tokens, rest -> <unk> (default {VOCAB_CAP}). "
                        f"Bounds the embedding + logits memory. Change with "
                        f"--rebuild-vocab.")
    p.add_argument("--data-dir", default=None,
                   help="override the corpus directory (default ./SimpleStories)")
    p.add_argument("--steps", type=int, default=STEPS,
                   help=f"total step count to train UP TO (absolute, not "
                        f"additional). Default {STEPS}. When resuming at step N, "
                        f"pass a value > N, e.g. --steps 200000.")
    args = p.parse_args()
    if args.score_min is not None:
        SCORE_MIN = args.score_min
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    if args.vocab_cap is not None:
        VOCAB_CAP = args.vocab_cap
    print(f"[v11 config] corpus={DATA_DIR}")
    print(f"[v11 config] score_min={SCORE_MIN}  min_count={args.min_count}  "
          f"vocab_cap={VOCAB_CAP}  batch={BATCH_SIZE}  seq={SEQ_LEN}")
    print(f"[v11 config] vocab_cache={VOCAB_PATH}")
    train(resume_from=args.resume_from, min_count=args.min_count,
          rebuild_vocab=args.rebuild_vocab, steps=args.steps)
