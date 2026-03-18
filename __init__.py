# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

try:
    import torch

    # yapf: disable
    from src.bert_layers import (BertAlibiEmbeddings, BertAlibiEncoder, BertForMaskedLM,
                                 BertForSequenceClassification,
                                 BertResidualGLU, BertAlibiLayer,
                                 BertLMPredictionHead, BertModel,
                                 BertOnlyMLMHead, BertOnlyNSPHead, BertPooler,
                                 BertPredictionHeadTransform, BertSelfOutput,
                                 BertAlibiUnpadAttention, BertAlibiUnpadSelfAttention)
    # yapf: enable
    from src.bert_padding import (
        IndexFirstAxis,
        IndexPutFirstAxis,
        index_first_axis,
        index_put_first_axis,
        pad_input,
        unpad_input,
        unpad_input_only,
    )
    from src.hf_bert import create_hf_bert_classification, create_hf_bert_mlm
    from src.mosaic_bert import create_mosaic_bert_classification, create_mosaic_bert_mlm
except ImportError as e:
    raise ImportError(
        "Install project dependencies with `pixi install`."
    ) from e

__all__ = [
    "BertAlibiEmbeddings",
    "BertAlibiEncoder",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "BertResidualGLU",
    "BertAlibiLayer",
    "BertLMPredictionHead",
    "BertModel",
    "BertOnlyMLMHead",
    "BertOnlyNSPHead",
    "BertPooler",
    "BertPredictionHeadTransform",
    "BertSelfOutput",
    "BertAlibiUnpadAttention",
    "BertAlibiUnpadSelfAttention",
    "IndexFirstAxis",
    "IndexPutFirstAxis",
    "index_first_axis",
    "index_put_first_axis",
    "pad_input",
    "unpad_input",
    "unpad_input_only",
    "create_hf_bert_classification",
    "create_hf_bert_mlm",
    "create_mosaic_bert_classification",
    "create_mosaic_bert_mlm",
]
