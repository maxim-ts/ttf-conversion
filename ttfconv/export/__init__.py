"""ttfconv.export: IR emitters for the conversion's module classes (QTad, Emb4) and converted families.
Importing it registers them with ttf.export.emit.
  voxcpm2   VoxCPM2 (ternary student): its graphs for the lm_diffusion driver
"""
import torch

from ttf.export.emit import emits, _beta_cuda


# ------------------------------------------------------------------------------------------- conversion modules

@emits("QTad")
def _e_qtad(ctx, m, p, x):
    lin, dn, up = m.lin, m.ad[0], m.ad[1]
    for sub, l in ((".lin", lin), (".ad.0", dn), (".ad.1", up)):
        ctx.derived[p + sub + ".beta"] = _beta_cuda(l)
        ctx.use_part(p + sub)
    assert lin.act_bits == dn.act_bits == up.act_bits
    attr = {"K": lin.K, "N": lin.N, "r": dn.N, "act_bits": lin.act_bits, "use_beta": bool(lin.use_beta),
            "adapter": True, "Ks": {".lin": lin.K, ".ad.0": dn.K, ".ad.1": up.K}}
    return ctx.node("ttf.linear", [x], p, attr)


@emits("Emb4")
def _e_emb4(ctx, m, p, ids):
    c = m.codes.to(torch.uint8)
    assert m.bits <= 4 and m.K % 2 == 0
    ctx.replace[p + ".codes"] = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()     # nib2: low nibble = even column
    ctx.use(p + ".codes", p + ".lo", p + ".s")
    ctx.layout_intk = "nib2"
    return ctx.node("ttf.embed", [ids], p, {"kind": "intk", "bits": m.bits, "g": m.g, "K": m.K, "V": m.N})
