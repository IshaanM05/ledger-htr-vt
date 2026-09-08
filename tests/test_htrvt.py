import torch

from ledger_htr.decode.ctc_codec import CTCCodec
from ledger_htr.models.cnn_stem import HTRConvStem
from ledger_htr.models.htr_vt import HTRViT


def test_ctc_codec_encode_roundtrip_lengths():
    codec = CTCCodec(["a", "b", "c"])
    flat, lengths = codec.encode(["ab", "cab"])
    assert lengths.tolist() == [2, 3]
    assert flat.tolist() == [codec.stoi["a"], codec.stoi["b"], codec.stoi["c"], codec.stoi["a"], codec.stoi["b"]]


def test_ctc_codec_decode_collapses_repeats_and_drops_blanks():
    codec = CTCCodec(["a", "b"])
    a, b, blank = codec.stoi["a"], codec.stoi["b"], codec.blank_index
    # CTC greedy decode: collapse consecutive repeats, then drop blanks --
    # blank between two same-character symbols prevents them collapsing.
    index_batch = torch.tensor(
        [
            [a, a, blank, b, b, b],  # -> "ab"
            [a, blank, a, b, blank, blank],  # -> "aab" (blank separates the two a's)
        ]
    )
    assert codec.decode_greedy(index_batch) == ["ab", "aab"]


def test_htr_conv_stem_collapses_height_to_one_row():
    stem = HTRConvStem(out_channels=64)
    x = torch.randn(2, 1, 64, 1024)
    out = stem(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 1  # height fully collapsed for target_height=64
    assert out.shape[3] == 1024 // 4  # width downsampled 4x


def test_htr_vt_forward_output_shape_matches_num_classes():
    model = HTRViT(nb_classes=10, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0)
    x = torch.randn(2, 1, 64, 256)
    out = model(x, mask_ratio=0.0, max_span_length=1, use_masking=False)
    assert out.shape[0] == 2
    assert out.shape[2] == 10


def test_htr_vt_forward_with_span_masking_does_not_crash():
    model = HTRViT(nb_classes=10, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0)
    x = torch.randn(2, 1, 64, 256)
    out = model(x, mask_ratio=0.4, max_span_length=4, use_masking=True)
    assert out.shape[0] == 2
    assert out.shape[2] == 10
