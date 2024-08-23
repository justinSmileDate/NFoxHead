import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .NFoxHead_choices import mc_sim_7b_63
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class NFoxHeadModel(PreTrainedModel):
    """The NFoxHead Language Model Head.

    This module creates a series of prediction heads (based on the 'NFoxHead' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the NFoxHeadModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs
        NFoxHead_num_heads = config.NFoxHead_num_heads
        NFoxHead_num_layers = config.NFoxHead_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.NFoxHead = NFoxHead_num_heads
        self.NFoxHead_num_layers = NFoxHead_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of NFoxHead heads
        self.NFoxHead_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * NFoxHead_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(NFoxHead_num_heads)
            ]
        )

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the NFoxHead_num_heads parameter is loaded
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            config=config,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        NFoxHead_forward=False,
        **kwargs,
    ):
        """Forward pass of the NFoxHeadModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all NFoxHead heads.
            (Optional) Original predictions from the base model's LM head.
        """
        """Forward pass of the NFoxHeadModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all NFoxHead heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not NFoxHead_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        NFoxHead_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.NFoxHead):
            NFoxHead_logits.append(self.NFoxHead_head[i](hidden_states))
        if output_orig:
            return torch.stack(NFoxHead_logits, dim=0), outputs, orig
        return torch.stack(NFoxHead_logits, dim=0)



class NFoxHeadLlamaModel(KVLlamaForCausalLM):
    """The NFoxHead Language Model Head.

    This module creates a series of prediction heads (based on the 'NFoxHead' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the NFoxHeadModel.
        """   
        # Load the base model
        super().__init__(config)
        # For compatibility with the old APIs

        NFoxHead_num_heads = config.NFoxHead_num_heads
        NFoxHead_num_layers = config.NFoxHead_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.NFoxHead = NFoxHead_num_heads
        self.NFoxHead_num_layers = NFoxHead_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of NFoxHead heads
        self.NFoxHead_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * NFoxHead_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(NFoxHead_num_heads)
            ]
        )

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
        
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the NFoxHead_num_heads parameter is loaded
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            config=config,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        NFoxHead_forward=False,
        **kwargs,
    ):
        """Forward pass of the NFoxHeadModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all NFoxHead heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not NFoxHead_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        NFoxHead_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.NFoxHead):
            NFoxHead_logits.append(self.NFoxHead_head[i](hidden_states))
        if output_orig:
            return torch.stack(NFoxHead_logits, dim=0), outputs, orig
        return torch.stack(NFoxHead_logits, dim=0)

    def NFoxHead_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the NFoxHead
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        NFoxHead_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of NFoxHead output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            NFoxHead_choices (list, optional): A list of integers indicating the number of choices for each NFoxHead head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache NFoxHead buffers (the fixed patterns for tree attention)
        if hasattr(self, "NFoxHead_choices") and self.NFoxHead_choices == NFoxHead_choices:
            # Load the cached NFoxHead buffer
            NFoxHead_buffers = self.NFoxHead_buffers
        else:
            # Initialize the NFoxHead buffer
            NFoxHead_buffers = generate_NFoxHead_buffers(
                NFoxHead_choices, device=self.base_model.device
            )
        self.NFoxHead_buffers = NFoxHead_buffers
        self.NFoxHead_choices = NFoxHead_choices


        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_NFoxHead_mode(self)
        # Initialize tree attention mask and process prefill tokens
        NFoxHead_logits, logits = initialize_NFoxHead(
            input_ids, self, NFoxHead_buffers["NFoxHead_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from NFoxHead heads
            candidates, tree_candidates = generate_candidates(
                NFoxHead_logits,
                logits,
                NFoxHead_buffers["tree_indices"],
                NFoxHead_buffers["retrieve_indices"],
            )

            # Use tree attention to verify the candidates and get predictions
            NFoxHead_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                NFoxHead_buffers["NFoxHead_position_ids"],
                input_ids,
                NFoxHead_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            # Update the input_ids and logits
            input_ids, logits, NFoxHead_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                NFoxHead_buffers["retrieve_indices"],
                outputs,
                logits,
                NFoxHead_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

# Currently only support LlamaModel
NFoxHeadModel = NFoxHeadLlamaModel