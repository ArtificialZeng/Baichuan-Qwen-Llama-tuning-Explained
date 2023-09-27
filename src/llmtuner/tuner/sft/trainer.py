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
        r"""
        Pads the tensor to the same length as the target tensor.

        Should only be called when predict_with_generate=True.
        """
        if pad_token_id is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
                assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
                pad_token_id = self.tokenizer.pad_token_id
            else:
                if self.model.config.pad_token_id is not None:
                    pad_token_id = self.model.config.pad_token_id
                else:
                    raise ValueError("Pad_token_id must be set in the configuration of the model.")

        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
