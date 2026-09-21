# MLAW-CBM architecture

## Research question

The original fixed-bin concept pooling is strong because ViT patch tokens can
carry global semantic context, but this can also produce an unintuitive
explanation: a concept may score highly in a spatial bin where the visible
clinical sign is absent. MLAW-CBM keeps the strong baseline concept interface
while injecting Attribute-selected high-frequency evidence into the patches
and the global preference branch.

The intended story is:

- disease classification needs a global summary;
- concept scoring still benefits from multiple fixed spatial observations;
- wavelet details make local boundaries and textures harder to average away;
- Attribute queries decide which frequency evidence is useful at each layer;
- multi-layer aggregation lets low-, middle-, and high-level evidence jointly
  support the final concepts.

## Main model: MLAW-CBM (historical V5-4)

At layer `l`, let `X_l` be the 196 ViT patch tokens. A fixed one-level Haar
transform constructs the LH, HL, and HH detail components. For each Attribute,
the model measures the change caused by removing one detail band. Sparsemax
routing assigns positive counterfactual evidence to the three bands or to a
no-wavelet route. Attribute spatial maps then fuse the routed residuals into a
shared selected high-frequency patch tensor `H*_l`.

The selected residual has two uses.

### Global preference / CLS path

```text
X_l + softplus(s_l) H*_l
          -> LayerNorm
          -> semantic + frequency Top-K selection
          -> wavelet-guided CLS_l
          -> Attribute preference
```

This path asks which Attributes are globally relevant to the image at the
current layer.

### Concept path

```text
X_l + softplus(s_l) H*_l          [B, 196, 768]
          -> fixed-bin AP_A
          -> shared MLP projector [B, A, 512]
          -> concept text similarity per Attribute
          -> global Attribute preference gate
```

The concept branch deliberately uses the pre-LayerNorm residual sum. LayerNorm
would remove much of the magnitude information introduced by the selected
wavelet residual. The fixed-bin adaptive pooling and concept projector are
inherited from the baseline, preserving its useful spatial coverage and
checkpoint interface.

Finally, MCSAF aggregates the per-concept scores from all ViT layers. The
resulting concept vector feeds a linear disease classifier.

## Energy ablation: MLAW-CBM-Energy (historical V5-7)

The Energy ablation tests whether wavelet magnitude should remain explicitly
separate from semantic patch content:

```text
content = AP_A(X_l)
energy  = sqrt(AP_A((softplus(s_l) H*_l)^2) + eps) - sqrt(eps)

Z_l = MLP_content(content)
    + softplus(g_l,a) * MLP_energy(energy)
```

Both branches use the same fixed spatial bins. `AP_A(X_l)` remains a clean
semantic path; RMS energy cannot be cancelled by positive/negative wavelet
coefficients. The separate energy projector and layer/Attribute gates make
this a larger and more explicit ablation, not the main paper model.

## Public and historical names

| Public role | Stable module | Historical implementation |
|---|---|---|
| Main model | `model.mlaw_cbm` | V5-4 |
| Energy ablation | `model.mlaw_cbm_energy` | V5-7 |
| Earlier Attribute-wavelet stage | V3 module | V3 |
| Counterfactual selector stage | newV5 module | newV5 |

The stable modules subclass the audited historical implementations so existing
V5-4/V5-7 state dictionaries remain compatible.
