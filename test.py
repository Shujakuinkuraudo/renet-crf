import torch
from pytorch_partial_crf import CRF

# Create
num_tags = 6
model = CRF(num_tags)

batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)
print(emissions)
tags = torch.LongTensor([
    [1, 2, 3, 3, 5],
    [1, 3, 4, 2, 1],
    [1, 0, 2, 4, 4],
])

# Computing negative log likelihood

model(emissions, tags)

model.viterbi_decode(emissions)
possible_tags = torch.randn(batch_size, sequence_length, num_tags)
possible_tags[possible_tags <= 0] = 0  # `0` express that can not pass.
possible_tags[possible_tags > 0] = 1  # `1` express that can pass.
possible_tags = possible_tags.byte()
print(model.restricted_viterbi_decode(emissions, possible_tags))
