import json
import re
from collections import Counter
import os
import torch
from transformers import AutoTokenizer
from typing import List

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        #self.dataset_name = args.dataset_name
        self.dataset_name = 'BRCA'
        if self.dataset_name == 'BRCA':
            self.clean_report = self.clean_report_brca
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []
        root = self.ann_path
        for dir in os.listdir(root):
            file_name = os.path.join(root, dir, 'annotation')

            anno = json.loads(open(file_name, 'r').read())
            tokens = self.clean_report(anno['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    # def clean_report_brca(self, report):
    #     report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ') \
    #         .replace('  ', ' ').replace('  ', ' ')\
    #         .replace(' 10. ', ' ').replace(' 11. ', ' ').replace(' 12. ', ' ').replace(' 13. ', ' ').replace(' 14.', ' ')    \
    #         .replace(' 1. ', ' ').replace(' 2. ', ' ') \
    #         .replace(' 3. ', ' ').replace(' 4. ', ' ').replace(' 5. ', ' ').replace(' 6. ', ' ').replace(' 7. ', ' ').replace(' 8. ', ' ') .replace(' 9. ', ' ')   \
    #         .strip().lower() + ' ').split('. ')
    #     sent_cleaner = lambda t: re.sub('[#,?;*!^&_+():-\[\]{}]', '', t.replace('"', '').
    #                                 replace('\\', '').replace("'", '').strip().lower())
    #     tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    #     report = ' . '.join(tokens) 
    #     return report
    
    def clean_report_brca(self, report: str) -> str:
        """
        A conservative cleaning function optimized for modern biomedical LLMs 
        like Llama3-OpenBioLLM-8B.
        """
        # 1. Normalize newlines and strip leading/trailing whitespace. Case is preserved.
        report = report.replace('\n', ' ').strip()
        # 2. Add spaces around key punctuation to ensure they are treated as separate tokens.
        report = re.sub(r'([():+,;])', r' \1 ', report)
        # 3. Carefully add spaces around periods to handle sentence ends and numbered lists
        # without breaking decimal numbers.
        report = re.sub(r'(?<=\d)\.(?!\d)', r' . ', report)
        report = re.sub(r'(?<!\d)\.(?!\d)', r' . ', report)
        report = re.sub(r'(?<!\d)\.(?=\d)', r' . ', report)
        # 4. Remove only a very restricted set of characters that are highly likely to be noise.
        report = re.sub(r'[#*?^&_\[\]{}]', '', report)
        # 5. Collapse multiple spaces into one and perform a final strip.
        report = re.sub(r'\s{2,}', ' ', report).strip()
        return report


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

class LlamaTokenizer(object):
    """
    A modern tokenizer that wraps the pre-trained Llama3-OpenBioLLM-8B tokenizer.

    This class replaces the custom, rule-based Tokenizer. It does NOT need to
    manually build a vocabulary or clean the text, as the Llama tokenizer
    handles raw text perfectly.
    """
    def __init__(self, args):
        """
        Initializes the tokenizer by loading it from the Hugging Face Hub.
        Assumes 'args' has 'max_seq_length'.
        """
        model_name = "aaditya/Llama3-OpenBioLLM-8B"
        print(f"INFO: Loading Hugging Face tokenizer: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = args.max_seq_length

        # Llama 3 models often don't have a default PAD token.
        # Setting it to the EOS token is a standard and safe practice.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"INFO: Tokenizer 'pad_token' not set. Using 'eos_token' as 'pad_token'.")

        # --- FIX for Model Compatibility ---
        # Create .idx2token and .token2idx attributes so your main model code doesn't crash.
        print("INFO: Creating .idx2token and .token2idx attributes for backward compatibility.")
        self.token2idx = self.tokenizer.get_vocab()
        self.idx2token = {id: token for token, id in self.token2idx.items()}

    def __call__(self, report: str) -> List[int]:
        """
        Encodes a single report string into a list of token IDs, handling
        all tokenization, padding, and truncation automatically.
        """
        encoded = self.tokenizer.encode(
            report,
            add_special_tokens=True,  # Adds BOS/EOS tokens automatically
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True
        )
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a clean string."""
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return decoded_text.strip()

    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """Decodes a batch of token ID lists efficiently."""
        return self.tokenizer.batch_decode(ids_batch, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        """Returns the total size of the pre-trained vocabulary."""
        return len(self.tokenizer)

    def get_token_by_id(self, token_id: int) -> str:
        """Converts a single token ID to its string representation."""
        return self.idx2token.get(token_id, self.tokenizer.unk_token)

    def get_id_by_token(self, token: str) -> int:
        """Converts a single token string to its ID."""
        return self.token2idx.get(token, self.token2idx.get(self.tokenizer.unk_token))
