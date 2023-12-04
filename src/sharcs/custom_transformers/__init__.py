# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

from .data import (is_sklearn_available,
                   InputExample, InputFeatures, DataProcessor,
                   glue_output_modes, glue_convert_examples_to_features,
                   glue_processors, glue_tasks_num_labels)

if is_sklearn_available():
    from .data import glue_compute_metrics

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
# from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
# from .tokenization_openai import OpenAIGPTTokenizer
# from .tokenization_transfo_xl import (TransfoXLTokenizer, TransfoXLCorpus)
# from .tokenization_gpt2 import GPT2Tokenizer
# from .tokenization_ctrl import CTRLTokenizer
# from .tokenization_xlnet import XLNetTokenizer, SPIECE_UNDERLINE
# from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
# from .tokenization_distilbert import DistilBertTokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_roberta import RobertaConfig, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
if is_torch_available():
    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)
    from .modeling_bert import (BertPreTrainedModel, BertModel,
                                BertForSequenceClassification,
                                BertForQuestionAnswering,
                                load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_distilbert import DistilBertForSequenceClassification
    from .modeling_albert import AlbertModel, AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer
    from .modeling_transkimer import SharcsBertForSequenceClassification
    from .modeling_roberta import (RobertaForMaskedLM, RobertaModel,
                                RobertaForSequenceClassification, RobertaForMultipleChoice,
                                RobertaForQuestionAnswering, 
                                ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .shallow_deep_net import ShallowDeepRoberta, ShallowDeepBert, ShallowDeepDistilBert
    from .branchy_net import BranchyRoberta, BranchyBert, BranchyDistilBert
    from .pabee import PABEERoberta, PABEEBert, PABEEDistilBert
    from .deebert import DeeRoberta, DeeBert, DeeDistilBert
    from .fastbert import FastRoberta, FastBert, FastDistilBert
    from .berxit import BerxitRoberta, BerxitBert, BerxitDistilBert

    # Optimization
    from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                               WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'distilbert': (BertConfig, DistilBertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'sharcstranskimer': (BertConfig, SharcsBertForSequenceClassification, BertTokenizer),

    'deeroberta': (RobertaConfig, DeeRoberta, RobertaTokenizer),
    'deebert': (BertConfig, DeeBert, BertTokenizer),
    'deedistilbert': (BertConfig, DeeDistilBert, BertTokenizer),
    'deedynabert': (BertConfig, DeeBert, BertTokenizer),

    'fastroberta': (RobertaConfig, FastRoberta, RobertaTokenizer),
    'fastbert': (BertConfig, FastBert, BertTokenizer),
    'fastdistilbert': (BertConfig, FastDistilBert, BertTokenizer),
    'fastdynabert': (BertConfig, FastBert, BertTokenizer),

    'berxitroberta': (RobertaConfig, BerxitRoberta, RobertaTokenizer),
    'berxitbert': (BertConfig, BerxitBert, BertTokenizer),
    'berxitdistilbert': (BertConfig, BerxitDistilBert, BertTokenizer),
    'berxitdynabert': (BertConfig, BerxitBert, BertTokenizer),

    'shallow_deep_roberta': (RobertaConfig, ShallowDeepRoberta, RobertaTokenizer),
    'shallow_deep_bert': (BertConfig, ShallowDeepBert, BertTokenizer),
    'shallow_deep_distilbert': (BertConfig, ShallowDeepDistilBert, BertTokenizer),
    'shallow_deep_dynabert': (BertConfig, ShallowDeepBert, BertTokenizer),

    'tinybert': (BertConfig, BertForSequenceClassification, BertTokenizer),

    'pabee_roberta': (RobertaConfig, PABEERoberta, RobertaTokenizer),
    'pabee_bert': (BertConfig, PABEEBert, BertTokenizer),
    'pabee_distilbert': (BertConfig, PABEEDistilBert, BertTokenizer),
    'pabee_dynabert': (BertConfig, PABEEBert, BertTokenizer),

    'branchy_roberta': (RobertaConfig, BranchyRoberta, RobertaTokenizer),
    'branchy_bert': (BertConfig, BranchyBert, BertTokenizer),
    'branchy_distilbert': (BertConfig, BranchyDistilBert, BertTokenizer),
    'branchy_dynabert': (BertConfig, BranchyBert, BertTokenizer),
}


if not is_tf_available() and not is_torch_available():
    logger.warning("Neither PyTorch nor TensorFlow >= 2.0 have been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")