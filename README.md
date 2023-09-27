# Baichuan-Qwen-Llama-tuning-Explained
Baichuan-Qwen-Llama-tuning-Explained  
说明：从Llama-Efficient-Tuning项目Fork而来，同时可以训练Baichuan、千问等，本项目是对其逐行代码的解析，更新中...

# Llama-Efficient-Tuning-Explained

Llama-Efficient-Tuning-相关代码，逐行详解版。


* [src/](./src)
  * [api_demo.py](/src/api_demo.py)
* [src/](./src)
  * [/src/llmtuner/tuner/tune.py](/src/llmtuner/tuner/tune.py)  主要微调模块
  * [/src/llmtuner/tuner/sft/trainer.py](/src/llmtuner/tuner/sft/trainer.py)
    * const logger
    * class CustomSeq2SeqTrainer
      * func prediction_step
      * func _pad_tensors_to_target_len
      * func save_predictions

  



# CSDN彩色博客版：
* [src/](./src)
  * [[CSDN彩色源码解析/src/api_demo.py]](https://blog.csdn.net/sinat_37574187/article/details/132303566?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132303566%22%2C%22source%22%3A%22sinat_37574187%22%7D)
* [README.md](./Llama-Efficient-Tuning-Explained/README.md)

Llama Efficient Tuning源码解析train_sft.py   https://zengxiaojian.blog.csdn.net/article/details/131458667


## 引用 - 源项目
