from .build import build_model_from_cfg
# from models.Point_MAE import Point_MAE, PointTransformer
from models.MAE3Dsparse import MaskedAutoencoderSparseSmall, MaskedAutoencoderSparseBase
from models.MAE3Dsparse_finetune import SWITransformerSmall, SWITransformerBase