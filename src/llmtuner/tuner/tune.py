from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 从 `typing` 模块导入类型标注工具
from llmtuner.extras.callbacks import LogCallback               # 从 `llmtuner.extras.callbacks` 模块导入 `LogCallback` 类
from llmtuner.extras.logging import get_logger                  # 从 `llmtuner.extras.logging` 模块导入 `get_logger` 函数
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer   # 从 `llmtuner.tuner.core` 模块导入 `get_train_args` 和 `load_model_and_tokenizer` 函数
from llmtuner.tuner.pt import run_pt                            # 从 `llmtuner.tuner.pt` 模块导入 `run_pt` 函数
from llmtuner.tuner.sft import run_sft                          # 从 `llmtuner.tuner.sft` 模块导入 `run_sft` 函数
from llmtuner.tuner.rm import run_rm                            # 从 `llmtuner.tuner.rm` 模块导入 `run_rm` 函数
from llmtuner.tuner.ppo import run_ppo                          # 从 `llmtuner.tuner.ppo` 模块导入 `run_ppo` 函数
from llmtuner.tuner.dpo import run_dpo                          # 从 `llmtuner.tuner.dpo` 模块导入 `run_dpo` 函数

if TYPE_CHECKING:                                               # 仅当进行类型检查时
    from transformers import TrainerCallback                    # 从 `transformers` 模块导入 `TrainerCallback` 类

logger = get_logger(__name__)                                   # 调用 `get_logger` 函数并传入当前模块的名称，返回一个日志对象

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):   # 定义一个函数 `run_exp`
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)  # 调用 `get_train_args`
    callbacks = [LogCallback()] if callbacks is None else callbacks   # 判断 `callbacks` 是否是 `None`


    if general_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, training_args, finetuning_args, _, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
