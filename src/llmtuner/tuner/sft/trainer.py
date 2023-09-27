import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core.trainer import PeftTrainer

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    # 说明: 这个类继承自Seq2SeqTrainer，主要用于计算生成指标如BLEU和ROUGE。
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 说明: 这个方法主要任务是从生成的令牌中移除提示部分。
        # 用户可以通过子类化和重写此方法来实现自定义行为。
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """

        # 检查是否使用生成进行预测。
        if self.args.predict_with_generate:
            
            # 断言：确保令牌化器的填充方向为左侧，且存在填充令牌ID。
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            
            # 获取输入的长度和标签的长度。
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)

            # 确保输入和标签的长度相同，如果不是，则将短的那个填充至和长的那个一样长。
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])

            # 检查其他可能的输入部分，如attention_mask和position_ids，如果它们存在，也进行相同的调整。
            if "attention_mask" in inputs:
                inputs["attention_mask"] = self._pad_tensors_to_target_len(
                    inputs["attention_mask"], inputs["labels"], pad_token_id=0
                )
            if "position_ids" in inputs:
                inputs["position_ids"] = self._pad_tensors_to_target_len(
                    inputs["position_ids"], inputs["labels"], pad_token_id=0
                )

        # 调用父类Seq2SeqTrainer的prediction_step方法来获得损失、生成的令牌和标签。
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )

        # 如果生成的令牌不为空并且是使用生成进行的预测，移除生成的令牌中的提示部分，并确保张量是连续的。
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        # 返回计算得到的损失、生成的令牌和标签。
        return loss, generated_tokens, labels


    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        # 说明: 将 src_tensor 填充至与 tgt_tensor 相同的长度。
        r"""
        Pads the tensor to the same length as the target tensor.
        """
    
        # 如果没有提供填充令牌ID，则使用类中的令牌化器的填充令牌ID。
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        
        # 创建一个与 tgt_tensor 形状相同的张量，但用填充令牌ID初始化。
        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        
        # 采用左填充的方式将 src_tensor 的值赋给新创建的填充张量。
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        # 确保返回的张量在内存中是连续的，并返回这个张量。
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
            self,
            predict_results: "PredictionOutput"
        ) -> None:
        # 说明: 保存模型的预测结果到指定的输出目录。
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        
        # 只在主进程中执行后续操作（在分布式训练中很重要，以避免多个进程重复保存）。
        if not self.is_world_process_zero():
            return

        # 定义预测输出文件的路径。
        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        # 记录一条日志消息。
        logger.info(f"Saving prediction results to {output_prediction_file}")

        # 将预测和标签中的 IGNORE_INDEX 替换为填充令牌ID。
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        # 使用令牌化器对预测和标签进行批量解码。
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # 打开预测输出文件以写入预测结果。
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            # 对于每一个解码后的预测和标签，将其作为JSON字符串添加到结果列表中。
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            # 将结果列表写入到文件中，每个结果占一行。
            writer.write("\n".join(res))


