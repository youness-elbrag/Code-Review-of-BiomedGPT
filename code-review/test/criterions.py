import unittest
import torch
from BioMedGPT.fairseq import utils
from BioMedGPT.fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from BioMedGPT.fairseq.modules import AdaptiveInput, AdaptiveSoftmax
from BioMedGPT.fairseq.criterions import FairseqCriterion
from blocks_Model.criterions import AdjustLabelSmoothedCrossEntropyCriterion  # Import your AdjustLabelSmoothedCrossEntropyCriterion class from the module

class DummyEncoder(FairseqEncoder):
    def forward(self, src_tokens, src_lengths):
        return src_tokens

class DummyDecoder(FairseqIncrementalDecoder):
    def forward(self, prev_output_tokens, encoder_out):
        return prev_output_tokens

class TestAdjustLabelSmoothedCrossEntropyCriterion(unittest.TestCase):

    def test_forward_pass(self):
        batch_size = 8
        src_len = 10
        tgt_len = 12
        vocab_size = 10000
        padding_idx = 0

        src_tokens = torch.randint(1, vocab_size, (batch_size, src_len))
        src_lengths = torch.randint(1, src_len + 1, (batch_size,))
        tgt_tokens = torch.randint(1, vocab_size, (batch_size, tgt_len))
        tgt_lengths = torch.randint(1, tgt_len + 1, (batch_size,))

        encoder = DummyEncoder()
        decoder = DummyDecoder()

        sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths},
            "target": tgt_tokens,
            "ntokens": tgt_lengths.sum().item()
        }

        criterion = AdjustLabelSmoothedCrossEntropyCriterion(
            task=None,
            sentence_avg=False,
            label_smoothing=0.1,
            padding_idx=padding_idx
        )

        loss, sample_size, logging_output = criterion(encoder, decoder, sample)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(sample_size, int)
        self.assertIsInstance(logging_output, dict)

if __name__ == '__main__':
    unittest.main()
