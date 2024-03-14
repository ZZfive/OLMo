from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer as BaseTokenizer

from .aliases import PathOrStr
from .config import ModelConfig, TokenizerConfig, TrainConfig, TruncationDirection
from .exceptions import OLMoConfigurationError

__all__ = ["Tokenizer"]


#  一个轻量级的 HuggingFace tokenizers.Tokenizer 的包装类
class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param eos_token_id: The token ID corresponding to the "end-of-sentence" token.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,  # 可以理解为最大长度
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,  # 默认右侧截断
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()  # 禁用base_tokenizer的自动截断功能
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)

    @property
    def vocab_size(self) -> int:  # 获取分词器中词典的大小
        return self.base_tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:  # 获取结束特殊字符
        return self.decode([self.eos_token_id], skip_special_tokens=False)

    @property
    def pad_token(self) -> str:  # 获取pad特殊字符
        return self.decode([self.pad_token_id], skip_special_tokens=False)

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> Tokenizer:  # 从TrainConfig构建Tokenizer
        tokenizer_identifier = config.tokenizer.identifier
        if Path(tokenizer_identifier).is_file():
            tokenizer = cls.from_file(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        else:
            tokenizer = cls.from_pretrained(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        if config.model.vocab_size != tokenizer.vocab_size:
            raise OLMoConfigurationError("vocab size mismatch between config and tokenizer")
        return tokenizer

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        # 获取eos_token_id，如果未传入，使用vocab中的最后一个词作为eos_token
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(filename)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: PathOrStr) -> Tokenizer:  # 从模模型保存路径构建Tokenizer
        """
        Load a tokenizer from a checkpoint.
        """
        from cached_path import cached_path

        # Load configs.
        config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))  # 获取配置文件路径
        tokenizer_config = TokenizerConfig.load(config_path, key="tokenizer")
        model_config = ModelConfig.load(config_path, key="model")

        # Initialize tokenizer and validate vocab size.
        if Path(tokenizer_config.identifier).is_file():
            tokenizer = cls.from_file(
                tokenizer_config.identifier,
                eos_token_id=model_config.eos_token_id,
                pad_token_id=model_config.pad_token_id,
            )
        else:
            tokenizer = cls.from_pretrained(
                tokenizer_config.identifier,
                eos_token_id=model_config.eos_token_id,
                pad_token_id=model_config.pad_token_id,
            )
        if model_config.vocab_size != tokenizer.vocab_size:
            raise OLMoConfigurationError("vocab size mismatch between config and tokenizer")
        return tokenizer

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:  # 给传入的字符ids列表末尾添加结束特殊字符id
        """
        Add special tokens in-place (if not already present) to the given token IDs. 将结束特殊标记ID就地添加到给定的标记IDs列表中（如果尚未存在）
        """
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1  # is_pair表示输入是否是成对，成对就要添加2个特殊字符，不成非只用添加1个字符

    # 根据设置的truncate_to和输出的input_ids长度进行截断
    def _truncate(
        self, input_ids: List[int], truncate_to: Optional[int], direction: TruncationDirection
    ) -> list[int]:
        if truncate_to is None or len(input_ids) <= truncate_to:
            return input_ids
        elif direction == TruncationDirection.left:
            return input_ids[len(input_ids) - truncate_to :]
        else:
            return input_ids[: -(len(input_ids) - truncate_to)]

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]: # 将单个文本句子编码为ids列表
        """
        Encode a string into token IDs.
        """
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]  # 因为使用的encode_batch，以[input]传入句子，故用[0]获取最终的编码列表ids

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        batch_encoding = self.base_tokenizer.encode_batch(inputs) # 先对完整的输入batch进行编码

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding.ids, truncate_to, self.truncate_direction)  # 对batch中的每个文本对应的编码ids进行截断
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)  # 在末尾添加结束特殊字符eos
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:  # 单个文本句子的ids列表解码还原为字符串
        """
        Decode a list of token IDs to a string. 将给定的IDs列表解码为字符串
        """
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
