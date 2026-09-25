"""Transformer model definitions (public architecture hyperparameters)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    n_experts: int = 1        # 1 = dense
    top_k: int = 1

    @property
    def attn_params_per_layer(self) -> int:
        q_o = 2 * self.d_model * self.n_heads * self.head_dim
        k_v = 2 * self.d_model * self.n_kv_heads * self.head_dim
        return q_o + k_v

    @property
    def expert_params_per_layer(self) -> int:
        return 3 * self.d_model * self.d_ff  # gated MLP: gate, up, down

    @property
    def always_active_params(self) -> int:
        """Matmul weights every token touches: attention + LM head."""
        return self.layers * self.attn_params_per_layer + self.vocab * self.d_model

    @property
    def total_params(self) -> int:
        experts = self.layers * self.n_experts * self.expert_params_per_layer
        return self.layers * self.attn_params_per_layer + experts + 2 * self.vocab * self.d_model

    @property
    def active_params_per_token(self) -> int:
        return self.always_active_params + self.layers * self.top_k * self.expert_params_per_layer

    def experts_touched(self, tokens: float) -> float:
        """Expected distinct experts hit per layer by `tokens` tokens (uniform routing)."""
        if self.n_experts == 1:
            return 1.0
        p_miss = (1 - self.top_k / self.n_experts) ** tokens
        return self.n_experts * (1 - p_miss)

    def weight_params_touched(self, tokens: float) -> float:
        return self.always_active_params + self.layers * self.experts_touched(tokens) * self.expert_params_per_layer

    def kv_bytes_per_token(self, kv_bytes: float) -> float:
        return 2 * self.layers * self.n_kv_heads * self.head_dim * kv_bytes

    def attn_macs_per_query(self, ctx: float) -> float:
        """QK^T + AV MACs for one query position attending to `ctx` keys."""
        return 2 * self.layers * self.n_heads * self.head_dim * ctx


LLAMA_8B = Model("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256)
LLAMA_70B = Model("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28672, 128256)
MIXTRAL_8X7B = Model("Mixtral-8x7B", 32, 4096, 32, 8, 128, 14336, 32000, n_experts=8, top_k=2)

MODELS = [LLAMA_8B, LLAMA_70B, MIXTRAL_8X7B]


@dataclass(frozen=True)
class ChatProfile:
    name: str = "chat"
    prompt_tokens: int = 1024
    output_tokens: int = 512
    spec_accept: float = 0.35     # prompt-lookup acceptance on free-form text


@dataclass(frozen=True)
class AgentProfile:
    """A tool-using agent episode: a loop of (tool result -> think -> tool call)."""
    name: str = "coding-agent"
    shared_prefix: int = 8000      # system prompt + tool schemas, identical across agents
    task_tokens: int = 500
    turns: int = 25
    tool_output_tokens: int = 1200  # file contents, search results, test logs...
    gen_tokens: int = 300           # reasoning + structured tool call
    tool_latency_s: float = 3.0
    spec_accept: float = 0.70       # tool calls copy paths/identifiers from context
    max_turn_latency_s: float = 10.0  # agent-turn SLO (prefill + generation)

    def ctx_at_turn(self, i: int) -> int:
        """Context length when turn i's new input has been appended (before generation)."""
        return self.shared_prefix + self.task_tokens + i * (self.gen_tokens + self.tool_output_tokens)

    def new_tokens_at_turn(self, i: int) -> int:
        return self.task_tokens if i == 0 else self.tool_output_tokens

    @property
    def final_ctx(self) -> int:
        return self.ctx_at_turn(self.turns - 1) + self.gen_tokens
