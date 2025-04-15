import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "chandar-lab/NeoBERT"
PAD_IDX = 0
MAX_LENTH = 4096

class Amelia(nn.Module):
    def __init__(
        self,
        num_ffnn_layers: int,
        ffnn_hidden_dim: int = -1,
        dropout: float = 0.1
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
        self.bert = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Determine the dimensionality of BERT's output.
        bert_hidden_size = self.bert.config.hidden_size
        if ffnn_hidden_dim == -1:
            ffnn_hidden_dim = bert_hidden_size 
        # Build the feed-forward classifier head.
        layers = []
        # First layer: Map BERT output to the FFNN internal dimension.
        layers.append(nn.Linear(bert_hidden_size, ffnn_hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Additional hidden layers, if requested.
        for _ in range(num_ffnn_layers - 1):
            layers.append(nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Final output layer: Reduce to a single value.
        layers.append(nn.Linear(ffnn_hidden_dim, 1))
        
        # Wrap layers into a Sequential module.
        self.ffnn = nn.Sequential(*layers)
    
    def forward(self, inputs, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of token IDs.
            attention_mask: (Optional) Attention mask tensor.
            token_type_ids: (Optional) Token type IDs tensor.
        
        Returns:
            Sigmoid activated output for binary classification.
        """
        # Run inputs through the BERT model.
        outputs = self.bert(inputs, attention_mask=attention_mask)
        
        # Retrieve the pooled output.
        # Many BERT models return a tuple or an object with a pooler_output attribute.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback: use the CLS token (assumed to be the first token).
            pooled_output = outputs[0][:, 0, :]
        
        # Pass the pooled output through the classifier head.
        logits = self.ffnn(pooled_output)
        
        # Apply sigmoid to get a score between 0 and 1.
        #output = torch.sigmoid(logits)
        return logits.squeeze(1)
    
    #As of now this not being used, but I made this method in case I want to do gradual unfreezing in the future.
    def set_trainable_layers(self, num_trainable_layers: int):
        """
        Gradually unfreezes the BERT encoder layers for fine-tuning.
        
        This method freezes all parameters first and then unfreezes 
        the last `num_trainable_layers` from the encoder.
        
        Args:
            num_trainable_layers (int): The number of transformer layers (from the end)
                                        to set as trainable.
        """
        # Freeze all parameters in BERT.
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Assume that the BERT-like model has an encoder with a list of layers.
        # This is common for Hugging Face BERT models.
        try:
            encoder_layers = self.bert.encoder.layer
        except AttributeError:
            raise AttributeError("The model does not have `encoder.layer` attribute. "
                                 "Ensure you are using a BERT-like model with accessible encoder layers.")
        
        total_layers = len(encoder_layers)
        # Unfreeze the last 'num_trainable_layers'.
        for layer in encoder_layers[total_layers - num_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Optionally, unfreeze the pooler if it exists.
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
