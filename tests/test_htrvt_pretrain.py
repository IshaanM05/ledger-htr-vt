import torch

from ledger_htr.data.pretrain_dataset import mask_image_strips
from ledger_htr.models.htr_vt import HTRMaskedAutoencoder, create_model, load_pretrained_encoder


def test_mask_image_strips_covers_expected_fraction_and_fills_correctly():
    images = torch.zeros(2, 1, 8, 100)
    masked, mask = mask_image_strips(images, mask_ratio=0.4, num_strips=2, fill_value=1.0)
    assert masked.shape == images.shape
    assert mask.shape == images.shape
    # every masked pixel was set to fill_value
    assert torch.all(masked[mask.bool()] == 1.0)
    # unmasked pixels are untouched
    assert torch.all(masked[~mask.bool()] == 0.0)
    # roughly mask_ratio of the width is covered (2 strips x 20px each = 40/100)
    frac_masked = mask[:, :, 0, :].float().mean().item()
    assert 0.3 < frac_masked < 0.5


def test_htr_masked_autoencoder_reconstructs_input_shape():
    model = HTRMaskedAutoencoder(embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0, target_height=8)
    x = torch.randn(2, 1, 8, 64)
    recon = model(x)
    assert recon.shape == x.shape


def test_load_pretrained_encoder_transfers_stem_blocks_norm_not_head():
    pretrain_model = HTRMaskedAutoencoder(embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0, target_height=8)
    ckpt_path = "/tmp/_test_htrvt_pretrain_ckpt.pth"
    torch.save({"model": pretrain_model.state_dict()}, ckpt_path)

    finetune_model = create_model(nb_classes=10)
    # override its architecture to match the tiny pretraining model for this test
    from ledger_htr.models.htr_vt import HTRViT

    finetune_model = HTRViT(nb_classes=10, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0)
    head_before = finetune_model.head.weight.clone()

    n_transferred = load_pretrained_encoder(finetune_model, ckpt_path)
    assert n_transferred > 0
    assert torch.equal(finetune_model.stem.stem.conv.weight, pretrain_model.stem.stem.conv.weight)
    assert torch.equal(finetune_model.head.weight, head_before)  # untouched, no pretraining counterpart
