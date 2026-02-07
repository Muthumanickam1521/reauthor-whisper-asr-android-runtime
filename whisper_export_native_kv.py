from pyexpat import model
import torch
import whisper
import whisper.model

OPSET = 17

# disable SDPA for ONNX compatibility
whisper.model.MultiHeadAttention.use_sdpa = False

class OnnxDecoder(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

        self.kv_cache, self.hooks = model.install_kv_cache_hooks()

        self.kv_modules = (
            [block.attn.key for block in self.decoder.blocks] +
            [block.attn.value for block in self.decoder.blocks]
        )

        self.initial_token_length = 2  # sot + notimestamps

    def forward(self, tokens, audio, cache):

        # restore cache
        for m, c in zip(self.kv_modules, cache):
            self.kv_cache[m] = c

        # native Whisper behavior
        if tokens.shape[-1] > self.initial_token_length:
            tokens = tokens[:, -1:]

        logits = self.decoder(tokens, audio, kv_cache=self.kv_cache)

        new_cache = torch.stack([self.kv_cache[m] for m in self.kv_modules])

        return logits, new_cache


def export():

    model = whisper.load_model("tiny.en").cpu().eval()

    encoder = model.encoder
    decoder = OnnxDecoder(model)

    mel = torch.randn(1, 80, 3000)
    audio = encoder(mel)

    cache = torch.zeros(
        (len(decoder.kv_modules), 1, 0, model.dims.n_text_state)
    )

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    tokens = torch.tensor([[tokenizer.sot, tokenizer.no_timestamps]])

    print("Exporting encoder.onnx ...")

    torch.onnx.export(
        encoder,
        (mel,),
        "encoder.onnx",
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch", 2: "time"},
            "audio": {0: "batch", 1: "time"},
        },
        opset_version=OPSET,
    )

    print("Exporting decoder.onnx ...")

    torch.onnx.export(
        decoder,
        (tokens, audio, cache),
        "decoder.onnx",
        input_names=["tokens", "audio", "cache"],
        output_names=["logits", "new_cache"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            "audio": {0: "batch", 1: "time"},   # DO NOT mark dim 2 dynamic
            "cache": {0: "layer", 1: "batch", 2: "cached_seq"},
            "new_cache": {0: "layer", 1: "batch", 2: "cached_seq"},
        },
        opset_version=17,
    )

    print("Export complete.")


if __name__ == "__main__":
    export()
