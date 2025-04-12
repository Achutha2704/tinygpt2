import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

class LayerNorm(nn.Module):
    """Layer normalization module with optional bias"""
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for flash attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.n_positions
        self.flash = config.flash
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(config.n_positions, config.n_positions))
                .view(1, 1, config.n_positions, config.n_positions)
            )
    
    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass for attention layer
        
        Args:
            x: Input tensor of shape [B, T, C]
            layer_past: Cached key and value tensors for generation
            use_cache: Whether to return present state for incremental decoding
            
        Returns:
            y: Output tensor of shape [B, T, C]
            present: Tuple of key and value tensors (if use_cache is True)
        """
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present = (k, v) if use_cache else None
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :T, :k.size(2)] == 0, 
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present

class FeedForward(nn.Module):
    """MLP block with GELU activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Apply feedforward network to input tensor"""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block with attention and feedforward networks"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass for decoder block
        
        Args:
            x: Input tensor
            layer_past: Past key-value pairs for attention
            use_cache: Whether to return key-value pairs for incremental decoding
            
        Returns:
            x: Output tensor (and present state if use_cache is True)
        """
        residual = x
        x_norm = self.ln_1(x)
        attn_out, present = self.attn(x_norm, layer_past=layer_past, use_cache=use_cache)
        x = residual + attn_out
        
        residual = x
        x = self.ln_2(x)
        ff_out = self.mlp(x)
        x = residual + ff_out
        
        if use_cache:
            return x, present
        return x

class TinyGPT2(nn.Module):
    """
    Base TinyGPT2 implementation with transformer architecture
    A smaller implementation of GPT2 for lightweight applications
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Do not initialize head separately if tying weights
        if not config.tie_word_embeddings:
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        """Initialize weights for the model following GPT-2 initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, position_ids=None, past_key_values=None, use_cache=False):
        """
        Forward pass for the base model
        
        Args:
            input_ids: Token IDs
            position_ids: Position IDs (optional)
            past_key_values: Cache for key-value pairs from previous steps
            use_cache: Whether to return key-value pairs for next steps
            
        Returns:
            hidden_states: Output embeddings
            presents: Key-value pairs for next steps (if use_cache is True)
        """
        batch_size, seq_length = input_ids.size()
        
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            seq_length = 1
        else:
            past_length = 0
            past_key_values = tuple([None] * len(self.blocks))
        
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + seq_length, 
                                        dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        presents = () if use_cache else None
        
        for i, (block, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
            if use_cache:
                hidden_states, present = block(hidden_states, past_key_value, use_cache=True)
                presents = presents + (present,)
            else:
                hidden_states = block(hidden_states)
        
        hidden_states = self.ln_f(hidden_states)
        
        if use_cache:
            return hidden_states, presents
        return hidden_states

class TinyGPT2Config(PretrainedConfig):
    """Configuration class for TinyGPT2 model"""
    model_type = "tiny-gpt2"
    
    def __init__(
        self,
        vocab_size=(50257+64),
        n_layer=8,
        n_head=8,
        n_embd=512,
        n_positions=512,
        dropout=0.0,
        bias=True,
        use_cache=True,
        flash=True,
        tie_word_embeddings=True,
        **kwargs
    ):
        """
        Initialize TinyGPT2 configuration
        
        Args:
            vocab_size: Size of vocabulary
            n_layer: Number of decoder layers
            n_head: Number of attention heads
            n_embd: Embedding dimension
            n_positions: Maximum sequence length
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            use_cache: Whether to return key-value pairs for incremental decoding
            flash: Whether to use flash attention if available
            tie_word_embeddings: Whether to tie input and output embeddings
        """
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.dropout = dropout
        self.bias = bias
        self.flash = flash
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings 
        super().__init__(**kwargs)


class TinyGPT2ForCausalLM(PreTrainedModel, GenerationMixin):
    """TinyGPT2 with language modeling head"""
    config_class = TinyGPT2Config
    base_model_prefix = "transformer"
    
    @property
    def supports_gradient_checkpointing(self):
        return True

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TinyGPT2(config)
        
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.config = config
        if config.tie_word_embeddings:
            self._tie_weights()
        
        # Add attribute to track gradient checkpointing state
        self.gradient_checkpointing = False

    def _tie_weights(self):
        """Tie input and output embeddings if configured"""
        pass
    
    def get_input_embeddings(self):
        """Get input embedding layer"""
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embedding layer"""
        self.transformer.wte = new_embeddings
    
    def get_output_embeddings(self):
        """Get output embedding layer (or None if weights are tied)"""
        return None if self.config.tie_word_embeddings else self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embedding layer if weights aren't tied"""
        if not self.config.tie_word_embeddings:
            self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the model to save memory
        """
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        self.gradient_checkpointing = True
        # Pass kwargs to transformer blocks if needed (optional)
        for block in self.transformer.blocks:
            block.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model
        """
        self.gradient_checkpointing = False
        for block in self.transformer.blocks:
            block.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        """
        Forward pass with support for language modeling, caching, and gradient checkpointing
        
        Args:
            input_ids: Token IDs for input
            attention_mask: Mask to avoid attention on padding tokens
            inputs_embeds: Embedded inputs (alternative to input_ids)
            labels: Labels for computing language modeling loss
            past_key_values: Cache for key-value pairs from previous steps
            use_cache: Whether to return key-value pairs for incremental decoding
            position_ids: Position IDs for positional embeddings
            token_type_ids: Token type IDs (not used but included for compatibility)
            output_attentions: Whether to return attention probabilities
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return as a dictionary
            
        Returns:
            CausalLMOutput containing loss and logits
        """
        from torch.utils.checkpoint import checkpoint

        use_cache = use_cache if use_cache is not None else self.config.use_cache
    
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.size()
            hidden_states = self.transformer.wte(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.size()
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
    
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            seq_length = 1 if input_ids is not None else seq_length
        else:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.blocks))
    
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0)
    
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        hidden_states = self.transformer.drop(hidden_states)
    
        presents = () if use_cache else None
        
        # Modify the forward pass to use checkpointing when enabled
        for i, (block, past_key_value) in enumerate(zip(self.transformer.blocks, past_key_values)):
            if use_cache:
                if self.gradient_checkpointing and self.training:
                    # Use checkpointing for memory efficiency during training
                    hidden_states, present = checkpoint(
                        block,
                        hidden_states,
                        past_key_value,
                        True,  # use_cache=True
                        use_reentrant=False  # Set to False for PyTorch 2.0+ compatibility
                    )
                else:
                    hidden_states, present = block(hidden_states, past_key_value, use_cache=True)
                presents = presents + (present,)
            else:
                if self.gradient_checkpointing and self.training:
                    hidden_states = checkpoint(
                        block,
                        hidden_states,
                        None,  # past_key_value
                        False,  # use_cache=False
                        use_reentrant=False
                    )
                else:
                    hidden_states = block(hidden_states)
    
        hidden_states = self.transformer.ln_f(hidden_states)
    
        if labels is not None:
            if self.config.tie_word_embeddings:
                logits = F.linear(hidden_states, self.transformer.wte.weight)
            else:
                logits = self.lm_head(hidden_states)
        else:
            hidden_states = hidden_states[:, -1:, :]
            if self.config.tie_word_embeddings:
                logits = F.linear(hidden_states, self.transformer.wte.weight)
            else:
                logits = self.lm_head(hidden_states)
    
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
        if not return_dict:
            return (loss, logits, past_key_values)
    
        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """
        Prepare inputs for generation
        
        Args:
            input_ids: Input token IDs
            past_key_values: Cached key-value pairs from previous steps
            attention_mask: Mask to avoid attention on padding tokens
            
        Returns:
            Dictionary of inputs for the forward pass
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cache for beam search
        
        Args:
            past_key_values: Cached key-value pairs
            beam_idx: Indices to reorder beams
            
        Returns:
            Reordered cache
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

# Register the custom config and model
CONFIG_MAPPING.register("tiny-gpt2", TinyGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(TinyGPT2Config, TinyGPT2ForCausalLM)

# Utility function to get default config
def get_config():
    """Return default configuration for TinyGPT2"""
    return {
        "vocab_size": 50257,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "n_positions": 512,
        "dropout": 0.0,
        "bias": True,
        "use_cache": True,
        "flash": True,
        "tie_word_embeddings": True,
    }
    
def get_tokenizer():
    """Return the tokenizer instance"""
    return tokenizer


if __name__ == "__main__":
    # Create config and instantiate model
    config = TinyGPT2Config(**get_config())
    model = TinyGPT2ForCausalLM(config)
    tokenizer = get_tokenizer() 

    # Save the model
    model_path = 'tiny_gpt2_init'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Print model size
    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nTinyGPT-2 size: {model_size/1000**2:.1f}M parameters")