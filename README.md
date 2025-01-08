# Transformer Implementation in PyTorch

This repository contains a PyTorch implementation of the Transformer model, including both the encoder and decoder components.

## Key Components

1. **TokenAndPositionEmbedding**:
   - Combines token embeddings and positional embeddings.
   - Ensures sequence information is retained.

2. **TransformerEncoderBlock**:
   - Contains a multi-head self-attention mechanism and a feed-forward network.
   - Includes layer normalization and dropout for regularization.

3. **TransformerEncoder**:
   - Stacks multiple `TransformerEncoderBlock` layers to encode input sequences.

4. **TransformerDecoderBlock**:
   - Combines self-attention, cross-attention (with encoder outputs), and a feed-forward network.

5. **TransformerDecoder**:
   - Stacks multiple `TransformerDecoderBlock` layers to decode sequences with encoder information.

6. **Transformer**:
   - Combines the encoder and decoder, providing a complete Transformer architecture.
  
## Visualization

Below is a visualization of the Transformer architecture for reference:

![Transformer Architecture](image/transformer.png)
