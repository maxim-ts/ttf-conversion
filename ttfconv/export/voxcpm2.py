"""ttfconv.export.voxcpm2: VoxCPM2 (ternary student) -> .ttf with its IR graphs.

Emitters for the module classes VoxCPM2 is made of (registered by class name: voxcpm is not a ttf dependency), the
Emb4 (registered in ttfconv.export), and the family builder that emits the named graphs the lm_diffusion driver calls:
  text_embed(ids [T]) -> e [T, 2048]                       int-k text embedding (Emb4, 4-bit, g 64)
  locenc(feat [T, P, 64]) -> embd [T, 2048]                 local encoder per patch + enc_to_lm_proj
  base_step(embd [T, 2048], pos [T]) -> hidden [T, 2048]    text-semantic LM (28 layers, kv.base.i), final norm
  fsq(h [T, 2048]) -> h                                     scalar quantisation bottleneck
  res_step(lm [T, 2048], feat [T, 2048], pos [T]) -> hidden residual LM (8 layers, no rope, kv.res.i) on
                                                            fusion_concat_proj(cat(lm, feat))
  dit_cond(lm [B, 2048], res [B, 2048]) -> mu [B, 2048]     cat(lm_to_dit_proj(lm), res_to_dit_proj(res))
  dit(x [B, P, 64], mu [B, 2048], t [B], cond [B, P, 64], dt [B]) -> v [B, P, 64]
                                                            the local DiT (bidirectional); time-major patches
  stop(h [T, 2048]) -> logits [T, 2]
  vae_decode(z [64, T]) -> wav [L]                          AudioVAE decoder at the out rate (48 kHz), streaming:
                                                            every causal conv keeps its input tail as state conv.*
  vae_encode(wav [1, L]) -> z [64, L / 640]                 AudioVAE encoder (mu), 16 kHz input
  python -m ttfconv.export.voxcpm2 MODEL_DIR CKPT OUT.ttf [--check [--int8]]   (cwd: the vox scripts, for vox_ttf)
"""
import json
import math
import os
import sys

import torch

from ttf.export.emit import Ctx, emits, _beta_cuda, float_path
from ttf.export.io import write_ttf, read_ttf
from ttf.export import ir as ttf_ir
from . import _e_qtad  # noqa: F401  (registers QTad)


# --------------------------------------------------------------------------------------------------- MiniCPM4

@emits("MiniCPMRMSNorm")
def _e_mnorm(ctx, m, p, x):
    ctx.use(p + ".weight")
    return ctx.node("rms_norm", [x], p, {"eps": m.variance_epsilon})


def _rope_attr(model):
    r = model.rope_emb
    if r is None:
        return None
    assert r.max_position_embeddings <= r.original_max_position_embeddings, "long factors: not exported"
    return {"theta": float(r.base), "style": "half", "dims": r.dim,
            "scaling": {"type": "longrope", "factors": [float(f) for f in r.short_factor], "mscale": float(r.scaling_factor)}}


@emits("MiniCPMAttention")
def _e_mattn(ctx, m, p, x, pos, rope, causal, sid, lead):
    """x [*lead, d]; causal: lead = [T] (kv state sid, written at pos); bidirectional: lead = [B, S]"""
    H, Hkv, hd = m.num_heads, m.num_key_value_heads, m.head_dim
    q = ctx.emit(m.q_proj, p + ".q_proj", x)
    k = ctx.emit(m.k_proj, p + ".k_proj", x)
    v = ctx.emit(m.v_proj, p + ".v_proj", x)
    q = ctx.node("reshape", [q], attr={"shape": lead + [H, hd]})
    k = ctx.node("reshape", [k], attr={"shape": lead + [Hkv, hd]})
    v = ctx.node("reshape", [v], attr={"shape": lead + [Hkv, hd]})
    if rope is not None:
        q = ctx.node("rope", [q, pos], attr=rope)
        k = ctx.node("rope", [k, pos], attr=rope)
    if causal:
        ctx.add_state(sid, "kv", heads_kv=Hkv, head_dim=hd)
        o = ctx.node("attn", [q, k, v, pos], attr={"causal": True, "state": sid, "scale": 1.0 / math.sqrt(hd)})
    else:
        o = ctx.node("attn", [q, k, v], attr={"causal": False, "scale": 1.0 / math.sqrt(hd)})
    o = ctx.node("reshape", [o], attr={"shape": lead + [H * hd]})
    return ctx.emit(m.o_proj, p + ".o_proj", o)


@emits("MiniCPMMLP")
def _e_mmlp(ctx, m, p, x):
    g = ctx.emit(m.gate_proj, p + ".gate_proj", x)
    u = ctx.emit(m.up_proj, p + ".up_proj", x)
    return ctx.emit(m.down_proj, p + ".down_proj", ctx.node("glu", [g, u], attr={"act": "silu", "order": "gate|up"}))


@emits("MiniCPMDecoderLayer")
def _e_mlayer(ctx, m, p, x, pos, rope, causal, sid, lead):
    def res(x, y):
        if m.use_mup:
            y = ctx.node("scale", [y], attr={"s": m.scale_depth / math.sqrt(m.num_hidden_layers)})
        return ctx.node("add", [x, y], hint="x")
    h = ctx.emit(m.input_layernorm, p + ".input_layernorm", x)
    x = res(x, ctx.emit(m.self_attn, p + ".self_attn", h, pos, rope, causal, sid, lead))
    h = ctx.emit(m.post_attention_layernorm, p + ".post_attention_layernorm", x)
    return res(x, ctx.emit(m.mlp, p + ".mlp", h))


@emits("MiniCPMModel")
def _e_mmodel(ctx, m, p, x, pos, causal, state_prefix, lead):
    rope = _rope_attr(m)
    for i, l in enumerate(m.layers):
        x = ctx.emit(l, f"{p}.layers.{i}", x, pos, rope, causal, f"{state_prefix}.{i}" if causal else None, lead)
    return ctx.emit(m.norm, p + ".norm", x)


# ------------------------------------------------------------------------------------------- VoxCPM2 local stacks

@emits("VoxCPMLocEnc")
def _e_locenc(ctx, m, p, feat):
    """feat [T, P, 64] -> [T, h]: CLS token + P patch frames through the bidirectional encoder, CLS out"""
    P = None
    x = ctx.emit(m.in_proj, p + ".in_proj", feat)                                   # [T, P, h]
    ctx.use(p + ".special_token")
    h = m.special_token.shape[-1]
    tok = ctx.node("const", [], attr={"w": p + ".special_token", "shape": [1, 1, h]})
    like = ctx.node("slice", [x], attr={"dim": 1, "start": 0, "len": 1})           # [T, 1, h]
    x = ctx.node("concat", [ctx.node("repeat", [tok, like]), x], attr={"dim": 1})   # [T, P + 1, h]
    S = ctx.locenc_S
    pos = ctx.node("arange", [], attr={"n": S})
    o = ctx.emit(m.encoder, p + ".encoder", x, pos, False, None, ["B", "S"])
    o = ctx.node("slice", [o], attr={"dim": 1, "start": 0, "len": 1})
    return ctx.node("reshape", [o], attr={"shape": ["T", h]})


@emits("SinusoidalPosEmb")
def _e_sin(ctx, m, p, t):
    return ctx.node("timestep_embed", [t], attr={"dim": m.dim, "scale": 1000.0, "max_period": 10000.0})


@emits("TimestepEmbedding")
def _e_temb(ctx, m, p, x):
    x = ctx.emit(m.linear_1, p + ".linear_1", x)
    return ctx.emit(m.linear_2, p + ".linear_2", ctx.emit(m.act, p + ".act", x))


@emits("VoxCPMLocDiT")
def _e_dit(ctx, m, p, x, mu, t, cond, dt):
    """x / cond [B, P, 64] (time-major), mu [B, n_mu h], t / dt [B] -> v [B, P, 64]"""
    assert type(m).__module__.endswith("local_dit_v2"), type(m).__module__
    h = m.config.hidden_size
    x = ctx.emit(m.in_proj, p + ".in_proj", x)
    cond = ctx.emit(m.cond_proj, p + ".cond_proj", cond)
    te = ctx.emit(m.time_mlp, p + ".time_mlp", ctx.emit(m.time_embeddings, p + ".time_embeddings", t))
    de = ctx.emit(m.delta_time_mlp, p + ".delta_time_mlp", ctx.emit(m.time_embeddings, p + ".time_embeddings", dt))
    te = ctx.node("reshape", [ctx.node("add", [te, de])], attr={"shape": ["B", 1, h]})
    mu = ctx.node("reshape", [mu], attr={"shape": ["B", -1, h]})
    seq = ctx.node("concat", [mu, te, cond, x], attr={"dim": 1})
    n_mu, P = ctx.dit_n_mu, ctx.patch
    pos = ctx.node("arange", [], attr={"n": n_mu + 1 + 2 * P})
    o = ctx.emit(m.decoder, p + ".decoder", seq, pos, False, None, ["B", "S"])
    o = ctx.node("slice", [o], attr={"dim": 1, "start": n_mu + 1 + P, "len": None})
    return ctx.emit(m.out_proj, p + ".out_proj", o)


@emits("ScalarQuantizationLayer")
def _e_fsq(ctx, m, p, x):
    x = ctx.node("tanh", [ctx.emit(m.in_proj, p + ".in_proj", x)])
    x = ctx.node("round", [ctx.node("mul", [x], attr={"c": float(m.scale)})])
    x = ctx.node("div", [x], attr={"c": float(m.scale)})
    return ctx.emit(m.out_proj, p + ".out_proj", x)


# --------------------------------------------------------------------------------------------------- AudioVAE

def _conv_weight(ctx, m, p):
    """weight norm folded: <p>.weight = v g / ||v|| (dim 0), as the module's pre-hook computes it"""
    if hasattr(m, "weight_v"):
        ctx.derived[p + ".weight"] = torch._weight_norm(m.weight_v.detach().float(), m.weight_g.detach().float(), 0).cpu()
    else:
        ctx.use(p + ".weight")
    if m.bias is not None:
        ctx.use(p + ".bias")


@emits("CausalConv1d")
def _e_cconv(ctx, m, p, x):
    _conv_weight(ctx, m, p)
    c = m._CausalConv1d__padding * 2 - m._CausalConv1d__output_padding
    a = {"ctx": c, "stride": m.stride[0], "dilation": m.dilation[0], "groups": m.groups, "kernel": m.kernel_size[0],
         "bias": m.bias is not None}
    if ctx.conv_stream and c > 0:
        a["state"] = ctx.add_state(f"conv.{p}", "conv", channels=m.in_channels, len=c)
    return ctx.node("conv1d_causal", [x], p, a)


@emits("CausalTransposeConv1d")
def _e_ctconv(ctx, m, p, x):
    _conv_weight(ctx, m, p)
    trim = m._CausalTransposeConv1d__padding * 2 - m._CausalTransposeConv1d__output_padding
    c = (m.kernel_size[0] - 1) // m.stride[0]
    a = {"ctx": c, "trim": trim, "stride": m.stride[0], "groups": m.groups, "kernel": m.kernel_size[0],
         "bias": m.bias is not None}
    if ctx.conv_stream and c > 0:
        a["state"] = ctx.add_state(f"conv.{p}", "conv", channels=m.in_channels, len=c)
    return ctx.node("conv_transpose1d_causal", [x], p, a)


@emits("Snake1d")
def _e_snake(ctx, m, p, x):
    ctx.use(p + ".alpha")
    return ctx.node("snake", [x], p)


@emits("CausalResidualUnit")
def _e_cres(ctx, m, p, x):
    return ctx.node("add", [x, ctx.emit(m.block, p + ".block", x)], hint="x")


@emits("CausalEncoderBlock")
def _e_cenc_b(ctx, m, p, x):
    return ctx.emit(m.block, p + ".block", x)


@emits("CausalDecoderBlock")
def _e_cdec_b(ctx, m, p, x):
    return ctx.emit(m.block, p + ".block", x)


@emits("CausalEncoder")
def _e_cenc(ctx, m, p, x):
    return ctx.emit(m.fc_mu, p + ".fc_mu", ctx.emit(m.block, p + ".block", x))


@emits("CausalDecoder")
def _e_cdec(ctx, m, p, x, out_rate):
    """the decoder at a fixed output rate: each sample-rate condition layer's scale / bias rows for that rate's bucket
    are folded into derived tensors <cond>.sr_scale / <cond>.sr_bias [C, 1]"""
    if m.sr_bin_boundaries is None:
        return ctx.emit(m.model, p + ".model", x)
    b = int(torch.bucketize(torch.tensor([out_rate]), m.sr_bin_boundaries.cpu())[0])
    for i, (layer, cl) in enumerate(zip(m.model, m.sr_cond_model)):
        if cl is not None:
            q = f"{p}.sr_cond_model.{i}"
            assert cl.cond_type in ("scale_bias", "scale_bias_init") and isinstance(cl.out_layer, torch.nn.Identity)
            ctx.derived[q + ".sr_scale"] = cl.scale_embed.weight[b].detach().float().reshape(-1, 1).cpu()
            ctx.derived[q + ".sr_bias"] = cl.bias_embed.weight[b].detach().float().reshape(-1, 1).cpu()
            x = ctx.node("add", [ctx.node("mul", [x], attr={"w": q + ".sr_scale"})], attr={"w": q + ".sr_bias"})
        x = ctx.emit(layer, f"{p}.model.{i}", x)
    return x


# ------------------------------------------------------------------------------------------------ the family

def build_voxcpm2(m, sd):
    ctx = Ctx(sd)
    c = m.config
    ctx.patch, ctx.locenc_S, ctx.dit_n_mu, ctx.conv_stream = m.patch_size, m.patch_size + 1, 2, False
    d, P, F = c.lm_config.hidden_size, m.patch_size, m.feat_dim

    ids, = ctx.begin("text_embed", [("ids", ["T"], "i32")])
    ctx.end(e=ctx.emit(m.base_lm.embed_tokens, "base_lm.embed_tokens", ids))

    feat, = ctx.begin("locenc", [("feat", ["T", P, F], "f32")])
    ctx.end(embd=ctx.emit(m.enc_to_lm_proj, "enc_to_lm_proj", ctx.emit(m.feat_encoder, "feat_encoder", feat)))

    e, pos = ctx.begin("base_step", [("embd", ["T", d], "f32"), ("pos", ["T"], "i32")])
    ctx.end(hidden=ctx.emit(m.base_lm, "base_lm", e, pos, True, "kv.base", ["T"]))

    h, = ctx.begin("fsq", [("h", ["T", d], "f32")])
    ctx.end(h=ctx.emit(m.fsq_layer, "fsq_layer", h))

    lm, ft, pos = ctx.begin("res_step", [("lm", ["T", d], "f32"), ("feat", ["T", d], "f32"), ("pos", ["T"], "i32")])
    x = ctx.emit(m.fusion_concat_proj, "fusion_concat_proj", ctx.node("concat", [lm, ft], attr={"dim": -1}))
    ctx.end(hidden=ctx.emit(m.residual_lm, "residual_lm", x, pos, True, "kv.res", ["T"]))

    lm, rs = ctx.begin("dit_cond", [("lm", ["B", d], "f32"), ("res", ["B", d], "f32")])
    ctx.end(mu=ctx.node("concat", [ctx.emit(m.lm_to_dit_proj, "lm_to_dit_proj", lm),
                                   ctx.emit(m.res_to_dit_proj, "res_to_dit_proj", rs)], attr={"dim": -1}))

    hd = c.dit_config.hidden_dim
    x, mu, t, cond, dt = ctx.begin("dit", [("x", ["B", P, F], "f32"), ("mu", ["B", 2 * hd], "f32"), ("t", ["B"], "f32"),
                                           ("cond", ["B", P, F], "f32"), ("dt", ["B"], "f32")])
    ctx.end(v=ctx.emit(m.feat_decoder.estimator, "feat_decoder.estimator", x, mu, t, cond, dt))

    h, = ctx.begin("stop", [("h", ["T", d], "f32")])
    s = ctx.emit(m.stop_actn, "stop_actn", ctx.emit(m.stop_proj, "stop_proj", h))
    ctx.end(logits=ctx.emit(m.stop_head, "stop_head", s))

    vae = m.audio_vae
    z, = ctx.begin("vae_decode", [("z", [vae.latent_dim, "T"], "f32")])
    ctx.conv_stream = True
    w = ctx.emit(vae.decoder, "audio_vae.decoder", z, vae.out_sample_rate)
    ctx.conv_stream = False
    ctx.end(wav=ctx.node("reshape", [w], attr={"shape": [-1]}))

    wav, = ctx.begin("vae_encode", [("wav", [1, "L"], "f32")])
    ctx.end(z=ctx.emit(vae.encoder, "audio_vae.encoder", wav))

    cfm = m.feat_decoder
    driver = {"name": "lm_diffusion", "graphs": {g: g for g in ctx.graphs}, "attr": {
        "patch_size": P, "feat_dim": F, "latent_dim": vae.latent_dim,
        "solver": cfm.solver, "timesteps": 10, "cfg": 2.0, "sway": 1.0, "temperature": 1.0, "cfg_zero_star": True,
        "zero_init_frac": 0.04, "mean_mode": bool(cfm.mean_mode), "sigma_min": cfm.sigma_min,
        "min_len": 2, "max_len": 2000, "streaming_prefix_len": 4,
        "audio_start": m.audio_start_token, "audio_end": m.audio_end_token,
        "ref_audio_start": m.ref_audio_start_token, "ref_audio_end": m.ref_audio_end_token,
        "scale_emb": float(c.lm_config.scale_emb) if c.lm_config.use_mup else 1.0,
        "in_sample_rate": vae.sample_rate, "out_sample_rate": vae.out_sample_rate,
        "chunk_size": vae.chunk_size, "decode_chunk_size": vae.decode_chunk_size,
        "tokenizer_post": "mask_multichar_cjk",
        "mask_rules": {"base_in": "text_mask * e + feat_mask * embd", "lm_out": "feat_mask * fsq(h) + text_mask * h",
                       "res_in_feat": "feat_mask * embd"}}}
    ir = {"ir_version": 1, "dims": {"T": "dynamic", "B": "dynamic", "L": "dynamic"}, "state": ctx.state,
          "graphs": ctx.graphs, "driver": driver}
    return ir, ctx


def export(m, model_dir, out, meta=None):
    sd = m.state_dict()
    ir, ctx = build_voxcpm2(m, sd)
    D = {torch.bfloat16: "bf16", torch.float32: "f32"}[m.base_lm.norm.weight.dtype]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    write_ttf(out, ctx.tensors(), {}, ir, "voxcpm2", cfg, D, model_dir, meta,
              extra_kv={"ttf.layout.intk": getattr(ctx, "layout_intk", "nib2")})
    return ir


# ------------------------------------------------------------------------------------------------ the check

def _rel(a, b):
    return float((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm().clamp_min(1e-30))


@torch.no_grad()
def check(m, path, dev="cuda", float_mode=True):
    """every graph through the reference runner on the FILE vs the PyTorch modules (same inputs)"""
    torch.backends.cudnn.allow_tf32 = False          # (cuDNN convs default to TF32: 1e-3 on the VAE)
    torch.backends.cuda.matmul.allow_tf32 = False
    kv, W = read_ttf(path)
    ir = json.loads(kv["ttf.ir"])
    D = ttf_ir.DTYPES[kv["ttf.compute_dtype"]]
    mdt = m.base_lm.norm.weight.dtype
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, generator=g)
    P, F, d = m.patch_size, m.feat_dim, m.config.lm_config.hidden_size
    res = {}
    import contextlib
    with (float_path() if float_mode else contextlib.nullcontext()):
        R = ttf_ir.Runner(ir, W, D, dev)
        md = lambda t: t.to(dev, mdt)
        ids = torch.randint(0, 1000, (9,), generator=g)
        res["text_embed"] = _rel(R("text_embed", ids=ids)["e"], m.base_lm.embed_tokens(ids.to(dev)))
        feat = rn(6, P, F)
        ref = m.enc_to_lm_proj(m.feat_encoder(md(feat)[None]))[0]
        res["locenc"] = _rel(R("locenc", feat=feat)["embd"], ref)
        e = rn(10, d) * 0.05
        ref, _ = m.base_lm(md(e)[None], is_causal=True)
        full = R("base_step", embd=e, pos=torch.arange(10))["hidden"]
        res["base_step prefill"] = _rel(full, ref[0])
        R.reset()
        a = R("base_step", embd=e[:7], pos=torch.arange(7))["hidden"]
        inc = torch.cat([a] + [R("base_step", embd=e[i:i + 1], pos=torch.tensor([i]))["hidden"] for i in range(7, 10)])
        res["base_step 7 + 3 decode vs prefill"] = _rel(inc, full)
        h = rn(5, d)
        res["fsq"] = _rel(R("fsq", h=h)["h"], m.fsq_layer(md(h)))
        lm, ft = rn(8, d), rn(8, d) * 0.05
        ref, _ = m.residual_lm(m.fusion_concat_proj(torch.cat([md(lm), md(ft)], -1))[None], is_causal=True)
        R.reset()
        res["res_step prefill"] = _rel(R("res_step", lm=lm, feat=ft, pos=torch.arange(8))["hidden"], ref[0])
        lm, rs = rn(2, d), rn(2, d)
        mu = R("dit_cond", lm=lm, res=rs)["mu"]
        res["dit_cond"] = _rel(mu, torch.cat([m.lm_to_dit_proj(md(lm)), m.res_to_dit_proj(md(rs))], -1))
        x, cond = rn(2, P, F), rn(2, P, F)
        t, dt = torch.tensor([0.7, 0.7]), torch.zeros(2)
        ref = m.feat_decoder.estimator(md(x).transpose(1, 2), md(mu), md(t), md(cond).transpose(1, 2), md(dt)).transpose(1, 2)
        res["dit"] = _rel(R("dit", x=x, mu=mu.cpu(), t=t, cond=cond, dt=dt)["v"], ref)
        h = rn(4, d)
        res["stop"] = _rel(R("stop", h=h)["logits"], m.stop_head(m.stop_actn(m.stop_proj(md(h)))))
        vae = m.audio_vae
        z = rn(vae.latent_dim, 8) * 0.5
        ref = vae.decode(z[None].to(dev))[0, 0]
        R.reset()
        full = R("vae_decode", z=z)["wav"]
        res["vae_decode"] = _rel(full, ref)
        R.reset()
        parts = [R("vae_decode", z=z[:, i:i + 2])["wav"] for i in range(0, 8, 2)]
        res["vae_decode streamed 4 x 2 vs full"] = _rel(torch.cat(parts), full)
        wav = rn(1, 640 * 6) * 0.1
        res["vae_encode"] = _rel(R("vae_encode", wav=wav)["z"], vae.encode(wav.to(dev), vae.sample_rate)[0])
    return res


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    import vox_ttf as vt
    md_, ck, out = sys.argv[1:4]
    m, c = vt.load_student(md_, ck, "cuda")
    m.eval()
    ir = export(m, md_, out, {"ckpt": os.path.basename(os.path.dirname(ck)) + "/" + os.path.basename(ck)})
    n = {k: len(g["nodes"]) for k, g in ir["graphs"].items()}
    ops = {}
    for g in ir["graphs"].values():
        for nd in g["nodes"]:
            ops[nd["op"]] = ops.get(nd["op"], 0) + 1
    print("graphs (nodes):", n, "| state", len(ir["state"]), "\nops:", ops, flush=True)
    print("wrote", out, f"{os.path.getsize(out) / 2 ** 20:.1f} MiB", flush=True)
    if "--check" in sys.argv:
        for fm in ((True, False) if "--int8" in sys.argv else (True,)):
            r = check(m, out, float_mode=fm)
            print("float path" if fm else "int8 path", "rel L2 runner (file) vs PyTorch:", flush=True)
            for k, v in r.items():
                print(f"  {k:36s} {v:.3e}", flush=True)
