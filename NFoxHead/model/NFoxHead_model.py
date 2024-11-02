import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .NFoxHead_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

class NFoxHeadConfig(PretrainedConfig):
    """
    Configuration class for NFoxHead model.

    Args:
        NFoxHead_num_heads (int, optional): Number of heads for the NFoxHead layer. Default is 2.
        NFoxHead_num_layers (int, optional): Number of NFoxHead layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        NFoxHead_num_heads=5,
        NFoxHead_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.NFoxHead_num_heads = NFoxHead_num_heads
        self.NFoxHead_num_layers = NFoxHead_num_layers
        self.base_model_name_or_path = base_model_name_or_path

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


class NFoxHeadModelABC(nn.Module):
    """The NFoxHead Language Model Head.

    This module creates a series of prediction heads (based on the 'NFoxHead' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

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
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the NFoxHead_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = NFoxHeadConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.NFoxHead_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.NFoxHead_num_layers = config.NFoxHead_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            NFoxHead_head_path = os.path.join(pretrained_model_name_or_path, "NFoxHead_lm_head.pt")
            if os.path.exists(NFoxHead_head_path):
                filename = NFoxHead_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "NFoxHead_lm_head.pt")
            NFoxHead_head_state_dict = torch.load(filename, map_location=model.device)
            model.NFoxHead_head.load_state_dict(NFoxHead_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


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

    def generate_NFoxHead_choices(logits, token_probabilities, top_k, n_heads):
        """
        Generate NFoxHead_choices token path combinations, sorted by acceptance probability.
    
        Parameters:
        - logits (np.ndarray): Representing the logits for each head.
        - token_probabilities (np.ndarray): Representing the acceptance probabilities for each token.
        - top_k (int): The number of top-k tokens to select from each head.
        - n_heads (int): The number of heads.
    
        Returns:
        - top_choices (list): The top 5 token path combinations sorted by acceptance probability.
        """
        # Validate inputs
        if logits.shape[1] != token_probabilities.shape[0]:
            raise ValueError("The number of tokens in logits must match the size of token_probabilities.")
        if top_k <= 0 or top_k > logits.shape[1]:
            raise ValueError("top_k must be a positive integer and less than or equal to the vocabulary size.")
        if n_heads <= 0 or n_heads > logits.shape[0]:
            raise ValueError("n_heads must be a positive integer and less than or equal to the number of heads in logits.")
    
        NFoxHead_choices = []
    
        # Select top-k tokens from each head based on logits
        top_tokens_per_head = []
        for head_logits in logits:
            # Get indices of the top-k logits
            top_indices = np.argsort(head_logits)[-top_k:][::-1]
            top_tokens_per_head.append(top_indices)
    
        # Generate path combinations using a loop
        current_paths = [[]]  # Start with an empty path
    
        for depth in range(n_heads):
            new_paths = []
            for current_path in current_paths:
                for token in top_tokens_per_head[depth]:
                    new_path = current_path + [token]
                    new_paths.append(new_path)
            
            current_paths = new_paths  # Move to the next depth
    
        # Calculate acceptance probabilities for each completed path
        for path in current_paths:
            path_probability = np.prod([token_probabilities[token] for token in path])
            NFoxHead_choices.append((path, path_probability))
    
        # Sort by acceptance probability and retain the top 5 paths
        NFoxHead_choices.sort(key=lambda x: x[1], reverse=True)
        top_choices = [choice for choice, _ in NFoxHead_choices[:5]]

    return top_choices

    def NFoxHead_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the NFoxHead
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        NFoxHead_choices=None,
        posterior_threshold=0.09,  # threshold validation of NFoxHead output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            NFoxHead_choices (list, optional): A list of integers indicating the number of choices for each NFoxHead head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache NFoxHead buffers (the fixed patterns for tree attention)
        if NFoxHead_choices is None:
            NFoxHead_choices = self.generate_NFoxHead_choices(self.NFoxHead_logits, self.pro, self.top_k, self.NFoxHead_heads)

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
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
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
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
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


class NFoxHeadModelLlama(NFoxHeadModelABC, KVLlamaForCausalLM):
    pass

class NFoxHeadModelMistral(NFoxHeadModelABC, KVMistralForCausalLM):
    pass


class NFoxHeadModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the NFoxHead_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # NFoxHead-v0.1 load
            config = NFoxHeadConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return NFoxHeadModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return NFoxHeadModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
