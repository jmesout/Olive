# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import torch
from onnx import numpy_helper
from onnxruntime_extensions import PyOrtFunction, util
from onnxruntime_extensions.cvt import HFTokenizerConverter

# the flags for pre-processing
USE_ONNX_STFT = True

# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH



def ovr_merge_models(  # pylint: disable=too-many-branches
    m1: ModelProto,
    m2: ModelProto,
    io_map: List[Tuple[str, str]],
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    prefix1: Optional[str] = None,
    prefix2: Optional[str] = None,
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    producer_name: Optional[str] = "onnx.compose.merge_models",
    producer_version: Optional[str] = "1.0",
    domain: Optional[str] = "",
    model_version: Optional[int] = 1,
) -> ModelProto:
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of outputs/inputs.
    Those inputs/outputs not specified in the io_map argument will remain as
    inputs/outputs of the combined model.

    Both models should have the same IR version, and same operator sets imported.

    Arguments:
        m1 (ModelProto): First model
        m2 (ModelProto): Second model
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                          representing outputs of the first graph and inputs of the second
                                          to be connected
        inputs (list of string): Optional list of inputs to be included in the combined graph
                                 By default, all inputs not present in the ``io_map`` argument will be
                                 included in the combined model
        outputs (list of string): Optional list of outputs to be included in the combined graph
                                  By default, all outputs not present in the ``io_map`` argument will be
                                  included in the combined model
        prefix1 (string): Optional prefix to be added to all names in m1
        prefix2 (string): Optional prefix to be added to all names in m2
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used
        producer_name (string): Optional producer name for the combined model. Default: 'onnx.compose'
        producer_version (string): Optional producer version for the combined model. Default: "1.0"
        domain (string): Optional domain of the combined model. Default: ""
        model_version (int): Optional version of the graph encoded. Default: 1

    Returns:
        ModelProto
    """
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )
    ir_version = m1.ir_version

    opset_import_map: onnx.compose.MutableMapping[str, int] = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {m1.opset_import} and {m2.opset_import}"
                )
        else:
            opset_import_map[entry.domain] = entry.version

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = m1_copy
            m1 = onnx.compose.add_prefix(m1, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = m2_copy
            m2 = onnx.compose.add_prefix(m2, prefix=prefix2)
        io_map = [
            (
                prefix1 + io[0] if prefix1 else io[0],
                prefix2 + io[1] if prefix2 else io[1],
            )
            for io in io_map
        ]

    graph = onnx.compose.merge_graphs(
        m1.graph,
        m2.graph,
        io_map,
        inputs=inputs,
        outputs=outputs,
        name=name,
        doc_string=doc_string,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        opset_imports=opset_imports,
        ir_version=ir_version,
    )

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    # Merging functions
    function_overlap = list(
        {f.name for f in m1.functions} & {f.name for f in m2.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)

    # checker.check_model(model)
    return model

class CustomOpStftNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(g, self, n_fft, hop_length, window):
        t_n_fft = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        t_hop_length = g.op("Constant", value_t=torch.tensor(hop_length, dtype=torch.int64))
        t_frame_size = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        return g.op("ai.onnx.contrib::StftNorm", self, t_n_fft, t_hop_length, window, t_frame_size)

    @staticmethod
    def forward(ctx, audio, n_fft, hop_length, window):
        win_length = window.shape[0]
        stft = torch.stft(
            audio,
            n_fft,
            hop_length,
            win_length,
            window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return stft.abs() ** 2


class WhisperPrePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(N_FFT)
        self.mel_filters = torch.from_numpy(util.mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        stft_norm = CustomOpStftNorm.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)
        magnitudes = stft_norm[:, :, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        spec_min = log_spec.max() - 8.0
        log_spec = torch.maximum(log_spec, spec_min)
        spec_shape = log_spec.shape
        padding_spec = torch.ones(
            spec_shape[0], spec_shape[1], (N_SAMPLES // HOP_LENGTH - spec_shape[2]), dtype=torch.float
        )
        padding_spec *= spec_min
        log_spec = torch.cat((log_spec, padding_spec), dim=2)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def _to_onnx_stft(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert custom-op STFT-Norm to ONNX STFT"""
    node_idx = 0
    new_stft_nodes = []
    stft_norm_node = None
    for node in onnx_model.graph.node:
        if node.op_type == "StftNorm":
            stft_norm_node = node
            break
        node_idx += 1

    if stft_norm_node is None:
        raise RuntimeError("Cannot find STFTNorm node in the graph")

    make_node = onnx.helper.make_node
    replaced_nodes = [
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_14_output_0"],
            name="const_14",
            value=numpy_helper.from_array(np.array([0, N_FFT // 2, 0, N_FFT // 2], dtype="int64"), name="const_14"),
        ),
        make_node(
            "Pad", inputs=[stft_norm_node.input[0], "const_14_output_0"], outputs=["pad_1_output_0"], mode="reflect"
        ),
        make_node(
            "STFT",
            inputs=["pad_1_output_0", stft_norm_node.input[2], stft_norm_node.input[3], stft_norm_node.input[4]],
            outputs=["stft_output_0"],
            name="stft",
            domain="",
            onesided=1,
        ),
        make_node(
            "Transpose",
            inputs=["stft_output_0"],
            outputs=["transpose_1_output_0"],
            name="transpose_1",
            perm=[0, 2, 1, 3],
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_17_output_0"],
            name="const_17",
            value=numpy_helper.from_array(np.array([2], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_18_output_0"],
            name="const_18",
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_19_output_0"],
            name="const_19",
            value=numpy_helper.from_array(np.array([-1], dtype="int64"), name=""),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["const_20_output_0"],
            name="const_20",
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name=""),
        ),
        make_node(
            "Slice",
            inputs=[
                "transpose_1_output_0",
                "const_18_output_0",
                "const_19_output_0",
                "const_17_output_0",
                "const_20_output_0",
            ],
            outputs=["slice_1_output_0"],
            name="slice_1",
        ),
        make_node("Constant", inputs=[], outputs=["const0_output_0"], name="const0", value_int=0),
        make_node("Constant", inputs=[], outputs=["const1_output_0"], name="const1", value_int=1),
        make_node(
            "Gather",
            inputs=["slice_1_output_0", "const0_output_0"],
            outputs=["gather_4_output_0"],
            name="gather_4",
            axis=3,
        ),
        make_node(
            "Gather",
            inputs=["slice_1_output_0", "const1_output_0"],
            outputs=["gather_5_output_0"],
            name="gather_5",
            axis=3,
        ),
        make_node("Mul", inputs=["gather_4_output_0", "gather_4_output_0"], outputs=["mul_output_0"], name="mul0"),
        make_node("Mul", inputs=["gather_5_output_0", "gather_5_output_0"], outputs=["mul_1_output_0"], name="mul1"),
        make_node("Add", inputs=["mul_output_0", "mul_1_output_0"], outputs=[stft_norm_node.output[0]], name="add0"),
    ]
    new_stft_nodes.extend(onnx_model.graph.node[:node_idx])
    new_stft_nodes.extend(replaced_nodes)
    node_idx += 1
    new_stft_nodes.extend(onnx_model.graph.node[node_idx:])
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_stft_nodes)
    return onnx_model


def _load_test_data(filepath: str, use_audio_decoder: bool) -> npt.NDArray[np.uint8]:
    if use_audio_decoder:
        with open(filepath, "rb") as strm:
            audio_blob = np.asarray(list(strm.read()), dtype=np.uint8)
    else:
        try:
            import librosa
        except ImportError:
            raise ImportError("Please pip3 install librosa without ort-extensions audio codec support.")
        audio_blob, _ = librosa.load(filepath)
    audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size
    return audio_blob


def _preprocessing(audio_data: npt.NDArray[np.uint8], use_audio_decoder) -> onnx.ModelProto:
    if use_audio_decoder:
        decoder = PyOrtFunction.from_customop(
            "AudioDecoder", cpu_only=True, downsampling_rate=SAMPLE_RATE, stereo_to_mono=1
        )
        audio_pcm = torch.from_numpy(decoder(audio_data))
    else:
        audio_pcm = torch.from_numpy(audio_data)

    whisper_processing = WhisperPrePipeline()
    model_args = (audio_pcm,)

    with io.BytesIO() as strm:
        torch.onnx.export(
            whisper_processing,
            model_args,
            strm,
            input_names=["audio_pcm"],
            output_names=["log_mel"],
            do_constant_folding=True,
            export_params=True,
            opset_version=17,
            dynamic_axes={
                "audio_pcm": {1: "sample_len"},
            },
        )
        model = onnx.load_from_string(strm.getvalue())

    if USE_ONNX_STFT:
        model = _to_onnx_stft(model)

    if use_audio_decoder:
        model = ovr_merge_models(decoder.onnx_model, model, io_map=[("floatPCM", "audio_pcm")])

    return model


def _postprocessing(name: str) -> onnx.ModelProto:
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(name)
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder", cvt=HFTokenizerConverter(processor.tokenizer).bpe_decoder, skip_special_tokens=True, cpu_only=True
    )

    return fn_decoder.onnx_model


def _merge_models(
    pre_model: onnx.ModelProto, core_model: onnx.ModelProto, post_model: onnx.ModelProto
) -> onnx.ModelProto:
    pre_core_model = ovr_merge_models(pre_model, core_model, io_map=[("log_mel", "input_features")])
    all_models = ovr_merge_models(pre_core_model, post_model, io_map=[("sequences", "ids")])
    bpe_decoder_node = all_models.graph.node.pop(-1)
    bpe_decoder_node.input.pop(0)
    bpe_decoder_node.input.extend(["generated_ids"])
    all_models.graph.node.extend(
        [onnx.helper.make_node("Cast", ["sequences"], ["generated_ids"], to=onnx.TensorProto.INT64), bpe_decoder_node]
    )
    return all_models


def add_pre_post_processing_to_model(
    model: onnx.ModelProto,
    output_filepath: str,
    model_name: str,
    testdata_filepath: str,
    use_audio_decoder: bool = True,
) -> onnx.ModelProto:
    audio_blob = _load_test_data(testdata_filepath, use_audio_decoder)
    pre_model = _preprocessing(audio_blob, use_audio_decoder)
    post_model = _postprocessing(model_name)
    final_model = _merge_models(pre_model, model, post_model)
    onnx.checker.check_model(final_model)

    try:
        onnx.save_model(final_model, output_filepath)
    except ValueError:
        onnx.save_model(
            final_model,
            output_filepath,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(output_filepath).name}.data",
            convert_attribute=True,
        )

    return final_model
