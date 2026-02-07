from json import decoder
from pyexpat import model
import torch
import whisper
import whisper.model

OPSET = 17

# disable SDPA for ONNX compatibility
whisper.model.MultiHeadAttention.use_sdpa = False

class OnnxDecoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

        self.key_modules = [block.attn.key for block in decoder.blocks]
        self.value_modules = [block.attn.value for block in decoder.blocks]
        self.kv_modules = self.key_modules + self.value_modules

    def forward(self, tokens, audio, cache):

        kv_cache = dict(zip(self.kv_modules, cache))

        # native Whisper logic
        if tokens.shape[1] > 1:
            tokens = tokens[:, -1:]

        logits = self.decoder(tokens, audio, kv_cache=kv_cache)

        new_cache = torch.stack([kv_cache[m] for m in self.kv_modules])

        return logits, new_cache


def export():

    model = whisper.load_model("tiny.en").cpu().eval()

    encoder = model.encoder
    decoder = OnnxDecoder(model.decoder)

    mel = torch.randn(1, 80, 3000)
    audio = encoder(mel)

    cache = torch.zeros((len(decoder.kv_modules), 1, 1, 384))
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
        input_names=["tokens","audio","cache"],
        output_names=["logits","new_cache"],
        dynamic_axes={
            "tokens": {0:"batch", 1:"seq"},
            "audio": {0:"batch", 1:"time"},
            "cache": {1:"batch", 2:"cached_seq"},        # removed dim 0
            "new_cache": {1:"batch", 2:"cached_seq"},    # removed dim 0
        },

        opset_version=17,
    )

    print("Export complete.")


if __name__ == "__main__":
    export()
