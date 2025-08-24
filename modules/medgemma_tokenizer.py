# In your new file, e.g., medgemma_tokenizer.py

from transformers import AutoProcessor
from PIL import Image
import torch
from typing import List, Dict

class MedGemmaTokenizer:
    """
    A wrapper to use the Hugging Face MedGemma processor while maintaining
    an interface similar to the original custom Tokenizer.
    """
    def __init__(self, args=None):
        """
        Initializes the MedGemma processor.
        The `args` object is no longer needed for vocab creation but is kept for compatibility.
        """
        # The model ID for the pre-trained MedGemma model
        self.model_id = "google/medgemma-4b-pt"
        
        print(f"INFO: Loading MedGemma processor for '{self.model_id}'...")
        
        # The AutoProcessor includes the text tokenizer and the image processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Expose attributes for compatibility with code that might expect them
        self.token2idx = self.processor.tokenizer.get_vocab()
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.bos_token_id = self.processor.tokenizer.bos_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        print(f"INFO: MedGemma processor loaded. Vocabulary size: {self.get_vocab_size()}")

    def __call__(self, report: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Processes a single image and the target report text into a format
        ready for the MedGemma model.

        Args:
            report (str): The ground truth report text.
            image (PIL.Image.Image): The input WSI patch or image.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'attention_mask',
                                     and 'pixel_values'.
        """
        # Create the full prompt required for training/inference.
        # The model learns to generate the `report` part after the prompt.
        prompt = f"<img> findings: {report}"
        
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt",
            padding="max_length", # Or another strategy depending on your dataloader
            truncation=True,
            max_length=512 # Adjust as needed
        )
        
        # Squeeze the tensors to remove the batch dimension (for a single example)
        return {key: val.squeeze(0) for key, val in inputs.items()}

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a text string."""
        # Use the highly optimized decoder from the Hugging Face processor
        # `skip_special_tokens=True` removes tokens like <bos>, <eos>, <pad>
        decoded_text = self.processor.decode(ids, skip_special_tokens=True)
        return decoded_text

    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """Decodes a batch of token ID lists."""
        return self.processor.batch_decode(ids_batch, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        """Returns the size of the MedGemma vocabulary."""
        return len(self.processor.tokenizer)

    # The following methods from your old tokenizer are no longer needed
    # as the AutoProcessor handles them automatically:
    # - create_vocabulary()
    # - clean_report_brca()
    # - get_token_by_id()
    # - get_id_by_token()