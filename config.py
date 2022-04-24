import numpy as np
from easydict import EasyDict as edict

config = edict()
config.ANCHOR_SIZE = (400, 600)

CFG_FIQA = edict()
CFG_FIQA.THRESHOLD = 40
CFG_FIQA.MODEL_PATH = 'model/SDD_FIQA_checkpoints_r50.pth'

CFG_REG = edict()
CFG_REG.THRESHOLD = 0.8
CFG_REG.BATCH_SIZE = 32
CFG_REG.NUM_NEIGHBORS = 5

CFG_SMILE = edict()
CFG_SMILE.MODEL_PATH = 'model/smile_score.h5'


