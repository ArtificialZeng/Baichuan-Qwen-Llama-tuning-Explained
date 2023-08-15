# Baichuan-Qwen-Llama-tuning-Explained
Baichuan-Qwen-Llama-tuning-Explained


# Llama-Efficient-Tuning-Explained

Llama-Efficient-Tuning-相关代码，逐行详解版。


* [src/](./src)
  * [utils/](./src/utils)
    * [common.py](./src/utils/common.py)
      * init_adapter（）
      * load_pretrained()
      * prepare_args()
    * [peft_trainer.py  （定义LogCallback、PeftTrainer）](./src/utils/peft_trainer.py)
    * [data_collator.py（DataCollatorForLlama类）](./src/utils/data_collator.py)
    * [seq2seq.py  （ComputeMetrics、Seq2SeqTrainerForLlama)](./src/utils/seq2seq.py)
  * [train_sft.py（导入DataCollatorForLlama、Seq2SeqTrainerForLlama)](./src/train_sft.py)
* [examples/](./examples)
  * [ads_generation.md（分布式运行范例）](./examples/ads_generation.md)
* [README.md](./README.md)



# CSDN彩色博客版：
* [src/](./Llama-Efficient-Tuning-Explained/src)
  * [utils/](./Llama-Efficient-Tuning-Explained/src/utils)
    * [common.py](./Llama-Efficient-Tuning-Explained/src/utils/common.py)
    * [peft_trainer.py](./Llama-Efficient-Tuning-Explained/src/utils/peft_trainer.py)
  * [CSDN彩色源码解析train_sft.py](https://zengxiaojian.blog.csdn.net/article/details/131458667)
* [README.md](./Llama-Efficient-Tuning-Explained/README.md)

Llama Efficient Tuning源码解析train_sft.py   https://zengxiaojian.blog.csdn.net/article/details/131458667


## 引用 - 源项目
