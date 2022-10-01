from torchpack.utils.config import configs

from core.register import import_all_modules_for_register, MODELS

configs.load("../../configs/default.yaml", recursive=True)

