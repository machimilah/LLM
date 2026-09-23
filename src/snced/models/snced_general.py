"""SN-CED, general architecture (plan sections 4, 6, 8, 10).

    input -> one causal encoder pass
               |-> detailed memory      (one slot per token, projected)
               |-> note compiler -> note encoder -> Semantic Note Memory
                                                      (one slot per note)
             query -> sufficiency router -> decoder reads SNM,
                                            plus detailed memory when asked for

Nothing here knows about entities, values or hops. The compiler points into the
source rather than choosing from a closed set, and the router is learned rather
than walking a known chain, so the same model applies to any text.

The composite loss follows plan section 10:

    L = L_task(notes path)                      reason from notes by default
      + lambda_route  * L_task(routed path)     the path actually served
      + lambda_detail_task * L_task(detailed)   keep the reference path competent
      + lambda_suff  * KL(detail || notes)      notes should behave like context
      + lambda_notes * E[number of notes]       keep the notebook small (ramped in;
                                                see `compression_weight`)
      + lambda_sep   * pointer overlap          anchor and micro-context must not
                                                select the same token
      + lambda_detail * E[detailed slots read]  make detailed access earn itself

Implementation choices that are NOT part of the architecture: the short causal
convolution in the encoder (`conv_kernel`) and chain decoding (`generative`).
Both were needed to make the synthetic benchmark learnable and are kept as
flags, not as components.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ced import CEDBaseline
from .memory_tiers import CompressedDetailMemory, Indexer, LocalMemory
from .note_compiler import NoteFields, OpenVocabNoteCompiler, segment_by_token, segment_fixed
from .note_memory import NoteEncoder
from .sufficiency_router import SufficiencyRouter, access_cost


@dataclass
class Notebook:
    """A compiled, frozen notebook: memory slots plus the fields behind them."""

    slots: torch.Tensor
    mask: torch.Tensor
    fields: NoteFields

    @property
    def n_slots(self) -> float:
        return float((~self.mask).float().sum(-1).mean())


@dataclass
class RoutedMemory:
    memory: torch.Tensor
    mask: torch.Tensor
    p_detail: torch.Tensor
    fetched_slots: torch.Tensor  # source tokens actually reread, per query


class SNCEDGeneral(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_enc: int = 2,
                 n_dec: int = 3, d_ff: int = 128, conv_kernel: int = 6, generative: bool = True,
                 compiler_hidden: int = 128, context_heads: int = 2, note_hidden: int = 128,
                 router_hidden: int = 64, segmentation: str = "token", boundary_token: int | None = None,
                 window: int = 24, fallback_span_mode: bool = True, fallback_spans: int = 3,
                 max_span_tokens: int = 24, local_window: int = 64, detail_stride: int = 4,
                 index_notes: int = 16, index_detail: int = 16, use_indexers: bool = True) -> None:
        super().__init__()
        self.backbone = CEDBaseline(vocab_size, d_model=d_model, n_heads=n_heads, n_enc=n_enc,
                                    n_dec=n_dec, d_ff=d_ff, conv_kernel=conv_kernel,
                                    generative=generative)
        self.compiler = OpenVocabNoteCompiler(d_model, compiler_hidden, context_heads)
        self.note_encoder = NoteEncoder(d_model, context_heads, note_hidden)
        self.router = SufficiencyRouter(d_model, router_hidden)
        self.segmentation, self.boundary_token, self.window = segmentation, boundary_token, window
        # Targeted source fallback: reread `fallback_spans` note sources instead
        # of the whole document (plan section 9: "recover exact source span").
        self.fallback_span_mode = fallback_span_mode
        self.fallback_spans, self.max_span_tokens = fallback_spans, max_span_tokens
        # Memory tiers: recent tokens, notes, compressed detail.
        self.local_memory = LocalMemory(local_window)
        self.detail_memory = CompressedDetailMemory(d_model, detail_stride)
        # Indexers keep decoder cost tied to k rather than to how much is remembered.
        self.use_indexers = use_indexers
        self.semantic_indexer = Indexer(d_model, index_notes)
        self.detail_indexer = Indexer(d_model, index_detail)

    # ----- memory construction (query-independent) -------------------------

    def segments(self, ctx: torch.Tensor) -> list[list[tuple[int, int]]]:
        if self.segmentation == "token" and self.boundary_token is not None:
            return segment_by_token(ctx, self.boundary_token)
        return segment_fixed(ctx, self.window)

    def encode(self, ctx: torch.Tensor):
        """One causal encoder pass, shared by both memory branches."""
        states = self.backbone.encode_states(ctx)
        return states, self.backbone.mem_proj(states)

    def compile(self, ctx: torch.Tensor, states: torch.Tensor | None = None) -> Notebook:
        """Compile the notebook before any question is known."""
        if states is None:
            states = self.backbone.encode_states(ctx)
        fields = self.compiler(states, self.segments(ctx))
        slots, mask = self.note_encoder(fields, context_length=ctx.size(1))
        return Notebook(slots, mask, fields)

    # ----- adaptive access -------------------------------------------------

    def route(self, query: torch.Tensor, notebook: Notebook, detailed: torch.Tensor,
              detail_mask: torch.Tensor, hard: bool = False, threshold: float = 0.5) -> RoutedMemory:
        """Assemble the cheapest sufficient memory for this query.

        Always: local memory (recent tokens) + the notes the semantic indexer
        retrieves. Only when the router asks: detail, either the source spans
        behind the relevant notes or indexed compressed detail. Every extra
        slot is charged to the access cost, so widening the memory has to pay
        for itself in task loss.
        """
        q_states = self.backbone.tok(query)
        q_mask = query != self.backbone.tok.padding_idx
        q_vec = (q_states * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
        p_detail = self.router(q_states, q_mask, notebook.slots, notebook.mask,
                               notebook.fields.confidence)
        gate = (p_detail > threshold).float() if hard else p_detail

        local, local_mask = self.local_memory(detailed, detail_mask)
        if self.use_indexers:
            notes, notes_mask, _ = self.semantic_indexer(q_vec, notebook.slots, notebook.mask)
        else:
            notes, notes_mask = notebook.slots, notebook.mask

        parts, masks = [local, notes], [local_mask, notes_mask]
        if self.fallback_span_mode:
            extra, extra_mask, fetched = self._source_spans(q_states, q_mask, notebook, detailed)
        else:
            compressed, compressed_mask = self.detail_memory(detailed, detail_mask)
            extra, extra_mask, _ = self.detail_indexer(q_vec, compressed, compressed_mask)
            fetched = (~extra_mask).float().sum(-1)
        parts.append(extra * gate.view(-1, 1, 1))
        masks.append(extra_mask | (gate < 1e-6).view(-1, 1))

        memory = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        return RoutedMemory(memory, mask, p_detail, fetched * gate)

    def _source_spans(self, q_states: torch.Tensor, q_mask: torch.Tensor, notebook: Notebook,
                      detailed: torch.Tensor):
        """Reread only the source spans behind the notes this query needs."""
        relevance = self.router.note_relevance(q_states, q_mask, notebook.slots, notebook.mask)
        k = min(self.fallback_spans, relevance.size(1))
        chosen = relevance.topk(k, dim=1).indices
        starts = notebook.fields.source.gather(1, chosen).long()
        ends = notebook.fields.source_end.gather(1, chosen).long()
        B, width = q_states.size(0), self.max_span_tokens * k
        spans = detailed.new_zeros(B, width, detailed.size(-1))
        span_mask = torch.ones(B, width, dtype=torch.bool, device=detailed.device)
        fetched = torch.zeros(B, device=detailed.device)
        for b in range(B):
            pos = 0
            for j in range(k):
                a = int(starts[b, j])
                e = min(int(ends[b, j]), a + self.max_span_tokens)
                n = max(0, min(e - a, width - pos))
                if n <= 0:
                    continue
                spans[b, pos : pos + n] = detailed[b, a : a + n]
                span_mask[b, pos : pos + n] = False
                pos += n
            fetched[b] = pos
        return spans, span_mask, fetched

    def decode(self, memory: torch.Tensor, mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        return self.backbone.decode_all(memory, mask, seq)

    def decode_all(self, memory: torch.Tensor, mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        return self.backbone.decode_all(memory, mask, seq)

    def decode_states(self, memory: torch.Tensor, mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        return self.backbone.decode_states(memory, mask, seq)

    # ----- training --------------------------------------------------------

    @staticmethod
    def compression_weight(weights: dict[str, float], step: int | None) -> float:
        """Ramp the note-count penalty in, rather than applying it from step 0.

        Applied immediately it prunes the notebook (20.6 -> 6.7 notes) before the
        compiler has learned to write anything worth keeping, so the model ends
        up compressing noise.
        """
        target = weights["notes"]
        start, ramp = weights.get("notes_start_step", 0), max(1, weights.get("notes_ramp_steps", 1))
        if step is None or step >= start + ramp:
            return target
        if step < start:
            return 0.0
        return target * (step - start) / ramp

    def loss(self, ctx: torch.Tensor, seq: torch.Tensor, target: torch.Tensor, rows: torch.Tensor,
             weights: dict[str, float], step: int | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """Composite loss. `rows` maps each query to its document in the batch."""
        pad = self.backbone.tok.padding_idx
        states, detailed = self.encode(ctx)
        detail_mask = ctx == pad
        notebook = self.compile(ctx, states)

        notes_logits = self.decode(notebook.slots[rows], notebook.mask[rows], seq)
        l_notes = F.cross_entropy(notes_logits.flatten(0, 1), target.flatten(), ignore_index=-100)

        routed = self.route(seq, Notebook(notebook.slots[rows], notebook.mask[rows],
                                          _index_fields(notebook.fields, rows)),
                            detailed[rows], detail_mask[rows])
        routed_logits = self.decode(routed.memory, routed.mask, seq)
        l_routed = F.cross_entropy(routed_logits.flatten(0, 1), target.flatten(), ignore_index=-100)

        # The detailed path is the reference the sufficiency term imitates, so it
        # must be trained too. Leaving it out of the loss left it at chance and
        # made the KL target noise (plan Rule 2).
        detail_logits = self.decode(detailed[rows], detail_mask[rows], seq)
        l_detail_task = F.cross_entropy(detail_logits.flatten(0, 1), target.flatten(), ignore_index=-100)

        scored = target != -100
        l_suff = F.kl_div(notes_logits[scored].log_softmax(-1),
                          detail_logits[scored].detach().log_softmax(-1),
                          log_target=True, reduction="batchmean")

        l_notes_count = notebook.fields.expected_notes.mean() / max(ctx.size(1), 1)
        l_access = access_cost(routed.p_detail, routed.fetched_slots, notebook.n_slots)
        l_separation = notebook.fields.pointer_overlap()
        w_notes = self.compression_weight(weights, step)

        total = (l_notes
                 + weights["route"] * l_routed
                 + weights.get("detail_task", 1.0) * l_detail_task
                 + weights["sufficiency"] * l_suff
                 + w_notes * l_notes_count
                 + weights.get("separation", 0.0) * l_separation
                 + weights["detail"] * l_access)
        parts = {"task_notes": float(l_notes), "task_routed": float(l_routed),
                 "task_detail": float(l_detail_task), "sufficiency": float(l_suff),
                 "note_count": float(notebook.fields.expected_notes.mean()), "access": float(l_access),
                 "p_detail": float(routed.p_detail.mean()), "note_slots": notebook.n_slots,
                 "separation": float(l_separation), "w_notes": w_notes,
                 "reread_tokens": float(routed.fetched_slots.mean())}
        return total, parts


def _index_fields(fields: NoteFields, rows: torch.Tensor) -> NoteFields:
    return NoteFields(**{k: (v[rows] if torch.is_tensor(v) and v.dim() >= 1 else v)
                         for k, v in vars(fields).items()})
