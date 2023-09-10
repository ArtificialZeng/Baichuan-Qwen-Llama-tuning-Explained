# 导入类型提示的相关模块
from typing import Any, Dict, Optional
# 导入dataclasses库的相关功能
from dataclasses import asdict, dataclass, field

# 为下一个类添加数据类的装饰器
@dataclass
class GeneratingArguments:
    # 类的文档描述，说明该类用于指定解码参数
    r"""
    Arguments pertaining to specify the decoding parameters.
    """

    # 定义do_sample属性，并提供其默认值和元数据
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    # 定义temperature属性，并提供其默认值和元数据
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    # 定义top_p属性，并提供其默认值和元数据
    top_p: Optional[float] = field(
        default=0.7,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    # 定义top_k属性，并提供其默认值和元数据
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    # 定义num_beams属性，并提供其默认值和元数据
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    # 定义max_length属性，并提供其默认值和元数据
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    # 定义max_new_tokens属性，并提供其默认值和元数据
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    # 定义repetition_penalty属性，并提供其默认值和元数据
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    # 定义length_penalty属性，并提供其默认值和元数据
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    # 定义一个将对象转换为字典的方法
    def to_dict(self) -> Dict[str, Any]:
        # 使用asdict函数将当前对象转换为字典
        args = asdict(self)
        # 检查args字典中是否有max_new_tokens键
        if args.get("max_new_tokens", None):
            # 从args字典中删除max_length键
            args.pop("max_length", None)
        # 返回args字典
        return args
