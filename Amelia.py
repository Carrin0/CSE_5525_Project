import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig, HqqConfig

MODEL_NAME = "chandar-lab/NeoBERT"
PAD_IDX = 0
MAX_LENGTH = 4096
DROUPOUT_IN = 0.2 

class Amelia(nn.Module):
    def __init__(
        self,
        num_ffnn_layers: int,
        ffnn_hidden_dim: int = -1,
        dropout: float = 0.5
    ):
        """
        Initializes the Amelia model.
        This is a model that has a BERT encoder and then a binary classification head.
        Args:
            model_name (str): Hugging Face model identifier.
            num_ffnn_layers (int): Number of hidden layers in the FFNN head.
            ffnn_hidden_dim (int): The internal dimension for all FFNN layers.
            dropout (float): Dropout probability to use after activation.
        """
        super(Amelia, self).__init__()
        # Load the pretrained BERT-like model.
        self.bert = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # quantization_config=quant_config,
            device_map="auto",
            # torch_dtype=torch.float16
            # attention_implementation='flash_attention_2'
        )
        #Freeze all but two layers
        for param in self.bert.parameters():
            param.requires_grad = False
            
        for layer in self.bert.transformer_encoder[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Determine the dimensionality of BERT's output.
        bert_hidden_size = self.bert.config.hidden_size
        if ffnn_hidden_dim == -1:
            ffnn_hidden_dim = bert_hidden_size 
        # Build the feed-forward classifier head.
        layers = []
        # First layer: Map BERT output to the FFNN internal dimension.
        layers.append(nn.Linear(bert_hidden_size, ffnn_hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(DROUPOUT_IN)) #This should stay constant
        
        # Additional hidden layers, if requested.
        for i in range(num_ffnn_layers - 1):
            layers.append(nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Final output layer: Reduce to a single value.
        layers.append(nn.Linear(ffnn_hidden_dim, 1))

        # Init params
        for module in layers:
            if isinstance(module, nn.Linear):
                if module.out_features == 1:
                    # Make the last layer Xavier Uniform
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Wrap layers into a Sequential module.
        self.ffnn = nn.Sequential(*layers)
    
    def forward(self, inputs, attention_mask, **kwargs):
        outputs = self.bert(inputs, attention_mask=attention_mask, max_seqlen=kwargs.get('max_seqlen'), cu_seqlens=kwargs.get('cu_seqlens'), position_ids=kwargs.get('position_ids'))
    
        # If packing metadata is provided, manually unpack.
        if "cu_seqlens" in kwargs:
            # outputs[0] is expected to be of shape [1, total_length, hidden_size]
            # Remove the singleton batch dimension.
            raw_output = outputs[0].squeeze(0)  # shape: [total_length, hidden_size]
            # Use cu_seqlens to get the start index (CLS token position) for each original sequence.
            cu_seqlens = kwargs["cu_seqlens"]  # should be a tensor of shape [batch_size+1]
            cls_indices = cu_seqlens[:-1].long().to(raw_output.device)  # shape: [batch_size]
            # Gather the token embeddings at these start positions.
            pooled_output = raw_output[cls_indices]  # shape: [batch_size, hidden_size]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs[0][:, 0, :]

        logits = self.ffnn(pooled_output)
        return logits.squeeze(1)  # Now logits has shape [batch_size]
    
    #As of now this not being used, but I made this method in case I want to do gradual unfreezing in the future.
    # def set_trainable_layers(self, num_trainable_layers: int):
    #     """
    #     Gradually unfreezes the BERT encoder layers for fine-tuning.
        
    #     This method freezes all parameters first and then unfreezes 
    #     the last `num_trainable_layers` from the encoder.
        
    #     Args:
    #         num_trainable_layers (int): The number of transformer layers (from the end)
    #                                     to set as trainable.
    #     """
    #     # Freeze all parameters in BERT.
    #     for param in self.bert.parameters():
    #         param.requires_grad = False
        
    #     # Assume that the BERT-like model has an encoder with a list of layers.
    #     # This is common for Hugging Face BERT models.
    #     try:
    #         encoder_layers = self.bert.encoder.layer
    #     except AttributeError:
    #         raise AttributeError("The model does not have `encoder.layer` attribute. "
    #                              "Ensure you are using a BERT-like model with accessible encoder layers.")
        
    #     total_layers = len(encoder_layers)
    #     # Unfreeze the last 'num_trainable_layers'.
    #     for layer in encoder_layers[total_layers - num_trainable_layers:]:
    #         for param in layer.parameters():
    #             param.requires_grad = True
        
    #     # Optionally, unfreeze the pooler if it exists.
    #     if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
    #         for param in self.bert.pooler.parameters():
    #             param.requires_grad = True