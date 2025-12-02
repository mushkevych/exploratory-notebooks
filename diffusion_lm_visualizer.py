import math
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from transformers import AutoTokenizer


@dataclass
class DiffusionStepInfo:
    step_index: int
    unresolved_count: int
    newly_committed_positions: List[int]


class MaskedDiffusionLMVisualizer:
    """
    Simplified visualizer for diffusion-style *resolution schedules*.

    This class does not call a language model. Instead, it:
      - takes a fixed prompt and a fixed completion paragraph,
      - tokenizes them,
      - starts from a fully-masked completion,
      - and gradually reveals the *true* completion tokens across several steps.

    The goal is didactic: show that, unlike AR decoding, multiple
    token positions can be resolved in parallel and in a non-left-to-right
    order.
    """

    def __init__(self, model_name: str = "roberta-base") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer does not define a [MASK]/<mask> token.")

        self.mask_token_id: int = self.tokenizer.mask_token_id
        # Not all tokenizers have CLS/SEP; they may be None.
        self.cls_token_id: Optional[int] = getattr(self.tokenizer, "cls_token_id", None)
        self.sep_token_id: Optional[int] = getattr(self.tokenizer, "sep_token_id", None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_ids(
        self,
        prompt: str,
        completion_text: str,
    ) -> Tuple[List[int], int, int, List[int]]:
        """
        Build an input sequence of the form:

            [CLS] prompt_tokens [SEP] [MASK] * L [SEP]

        where L is the number of tokens in `completion_text`.

        Returns:
            input_ids: full sequence of token ids (with [MASK]s in completion)
            completion_start: index of the first completion token
            completion_end: index one past the last completion token
            completion_ids: token ids of the *true* completion
        """
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        completion_ids: List[int] = self.tokenizer.encode(
            completion_text,
            add_special_tokens=False,
        )

        ids: List[int] = []
        if self.cls_token_id is not None:
            ids.append(self.cls_token_id)

        ids.extend(prompt_ids)

        if self.sep_token_id is not None:
            ids.append(self.sep_token_id)

        completion_start: int = len(ids)

        # Fill completion region with [MASK]
        ids.extend([self.mask_token_id] * len(completion_ids))
        completion_end: int = len(ids)

        if self.sep_token_id is not None:
            ids.append(self.sep_token_id)

        return ids, completion_start, completion_end, completion_ids

    def _decode_tokens_for_display(self, ids: List[int]) -> List[str]:
        """
        Convert token ids to human-readable tokens, roughly aligning
        1 token id -> 1 visible piece of text.

        We work at the subword level and then apply
        `convert_tokens_to_string` on each token to strip artifacts
        like Ġ / ## etc.
        """
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids)
        pretty_tokens: List[str] = []
        for tok in raw_tokens:
            text = self.tokenizer.convert_tokens_to_string([tok])
            # Make newlines visible but compact
            text = text.replace("\n", "¶")
            if text.strip() == "":
                text = tok
            pretty_tokens.append(text)
        return pretty_tokens

    def _render_html_state(
        self,
        tokens: List[str],
        completion_start: int,
        completion_end: int,
        unresolved_positions: List[int],
        newly_committed: List[int],
        step_info: DiffusionStepInfo,
    ) -> str:
        """
        Build an HTML snippet visualizing the current state:

          - prompt tokens in gray,
          - unresolved completion positions as [MASK] in light gray,
          - resolved completion tokens in blue,
          - *newly* committed tokens in green and underlined.
        """
        unresolved_set = set(unresolved_positions)
        newly_committed_set = set(newly_committed)

        html_tokens: List[str] = []
        for idx, tok in enumerate(tokens):
            safe_tok = (
                tok.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            if idx < completion_start:
                # Prompt region
                html_tok = f'<span style="color:#555;">{safe_tok}</span>'
            elif idx >= completion_end:
                # Trailing [SEP] or anything else
                html_tok = f'<span style="color:#999;">{safe_tok}</span>'
            else:
                # Completion region
                local_pos = idx - completion_start
                if local_pos in unresolved_set:
                    html_tok = '<span style="color:#bbb; font-style:italic;">[MASK]</span>'
                else:
                    if local_pos in newly_committed_set:
                        html_tok = (
                            '<span style="color:#008000; font-weight:bold; '
                            'text-decoration:underline;">'
                            f"{safe_tok}</span>"
                        )
                    else:
                        html_tok = f'<span style="color:#0066cc;">{safe_tok}</span>'

            html_tokens.append(html_tok)

        header = (
            f"<b>Diffusion step {step_info.step_index}</b> "
            f"(unresolved: {step_info.unresolved_count})"
        )
        tokens_html = " ".join(html_tokens)

        return (
            "<div style='font-family:monospace; line-height:1.6;'>"
            f"{header}<br/><br/>{tokens_html}</div>"
        )

    # ------------------------------------------------------------------ #
    # Public API: schedule-based visualization
    # ------------------------------------------------------------------ #

    def visualize_schedule_on_text(
        self,
        prompt: str,
        completion_text: str,
        num_steps: int = 8,
        schedule: str = "entropy_sink",
        seed: int = 0,
    ) -> Iterator[str]:
        """
        Generator that yields HTML frames showing a diffusion-style
        resolution schedule over a *fixed* completion text.

        Args:
            prompt: conditioning text (fixed; never changed).
            completion_text: final paragraph to be gradually revealed.
            num_steps: number of refinement steps.
            schedule: one of {"left_to_right", "random", "entropy_sink"}.
            seed: RNG seed for the "random" schedule.
        """
        (
            input_ids,
            completion_start,
            completion_end,
            completion_ids,
        ) = self._build_ids(prompt=prompt, completion_text=completion_text)

        L: int = len(completion_ids)
        positions: List[int] = list(range(L))

        # Decide resolution order over completion positions.
        if schedule == "left_to_right":
            order = positions
        elif schedule == "random":
            rng = random.Random(seed)
            order = positions[:]
            rng.shuffle(order)
        elif schedule == "entropy_sink":
            # Simple proxy for "entropy sink":
            # positions near the prompt (left side) are resolved earlier.
            order = sorted(positions, key=lambda k: k)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        # Split order into ~equal-sized chunks for each step.
        chunk_size = max(1, math.ceil(L / max(1, num_steps)))
        chunks: List[List[int]] = [
            order[i : i + chunk_size] for i in range(0, L, chunk_size)
        ]

        unresolved_positions: List[int] = positions[:]
        committed_step: Dict[int, int] = {}

        # Iterate over chunks, committing tokens.
        for step_idx, chunk in enumerate(chunks, start=1):
            newly_committed: List[int] = []

            for local_pos in chunk:
                if local_pos not in unresolved_positions:
                    continue
                idx = completion_start + local_pos
                input_ids[idx] = completion_ids[local_pos]
                unresolved_positions.remove(local_pos)
                committed_step[local_pos] = step_idx
                newly_committed.append(local_pos)

            step = DiffusionStepInfo(
                step_index=step_idx,
                unresolved_count=len(unresolved_positions),
                newly_committed_positions=newly_committed,
            )

            tokens = self._decode_tokens_for_display(input_ids)
            frame_html = self._render_html_state(
                tokens=tokens,
                completion_start=completion_start,
                completion_end=completion_end,
                unresolved_positions=unresolved_positions,
                newly_committed=newly_committed,
                step_info=step,
            )
            yield frame_html

        # If num_steps is larger than the number of chunks, emit no-op frames.
        for step_idx in range(len(chunks) + 1, num_steps + 1):
            step = DiffusionStepInfo(
                step_index=step_idx,
                unresolved_count=len(unresolved_positions),
                newly_committed_positions=[],
            )
            tokens = self._decode_tokens_for_display(input_ids)
            frame_html = self._render_html_state(
                tokens=tokens,
                completion_start=completion_start,
                completion_end=completion_end,
                unresolved_positions=unresolved_positions,
                newly_committed=[],
                step_info=step,
            )
            yield frame_html
