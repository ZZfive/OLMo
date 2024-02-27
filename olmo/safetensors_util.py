import base64
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch

from olmo.aliases import PathOrStr

__all__ = [
    "state_dict_to_safetensors_file",
    "safetensors_file_to_state_dict",
]


# eq=True 表示为数据类自动生成 __eq__ 方法用于比较对象的相等性，参数 frozen=True 表示数据类是不可变的，即一旦创建对象后，对象的属性值不能再被修改
@dataclass(eq=True, frozen=True)
class STKey:
    keys: Tuple
    value_is_pickled: bool


def encode_key(key: STKey) -> str:  # 对STKey对象进行编码，返回的字符串可以作为唯一标识符来表示原始STKey对象
    b = pickle.dumps((key.keys, key.value_is_pickled))  # 先将所有keys序列化为字节流
    b = base64.urlsafe_b64encode(b)  # 将字节流进行 Base64 编码，得到一个 URL 安全的 Base64 编码字符串
    return str(b, "ASCII")  #  将编码后的字节流转换为 ASCII 编码的字符串返回


def decode_key(key: str) -> STKey:  # 将表示STKey对象的唯一字符串解码还原为STKey对象
    b = base64.urlsafe_b64decode(key)
    keys, value_is_pickled = pickle.loads(b)
    return STKey(keys, value_is_pickled)


def flatten_dict(d: Dict) -> Dict[STKey, torch.Tensor]:
    result = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[STKey((key,), False)] = value
        elif isinstance(value, dict):
            value = flatten_dict(value)  # 递归调用
            for inner_key, inner_value in value.items():
                result[STKey((key,) + inner_key.keys, inner_key.value_is_pickled)] = inner_value
        else:  # 如果值不是 torch.Tensor 类型也不是字典类型
            pickled = bytearray(pickle.dumps(value))   # 将其序列化为字节流
            pickled_tensor = torch.frombuffer(pickled, dtype=torch.uint8)  # 将字节流转换为 torch.Tensor
            result[STKey((key,), True)] = pickled_tensor
    return result


def unflatten_dict(d: Dict[STKey, torch.Tensor]) -> Dict:  # 将拉平的dict转换为嵌套有层级的dict
    result: Dict = {}

    for key, value in d.items():
        if key.value_is_pickled:  # 值是经过序列化的字节流，需要进行反序列化，使用 pickle.loads 方法将字节流反序列化为原始对象
            value = pickle.loads(value.numpy().data)

        target_dict = result
        for k in key.keys[:-1]:
            new_target_dict = target_dict.get(k)
            if new_target_dict is None:
                new_target_dict = {}
                target_dict[k] = new_target_dict
            target_dict = new_target_dict  # 如果存在嵌套的key，将target_dict指向new_target_dict相当于往下够了一级
        target_dict[key.keys[-1]] = value  # keys中的最后元素就是最底层的key，其值为value

    return result


def state_dict_to_safetensors_file(state_dict: Dict, filename: PathOrStr):  # 将state_dict保存为safetensors格式文件
    state_dict = flatten_dict(state_dict)
    state_dict = {encode_key(k): v for k, v in state_dict.items()}
    safetensors.torch.save_file(state_dict, filename)


def safetensors_file_to_state_dict(filename: PathOrStr, map_location: Optional[str] = None) -> Dict:  # 将safetensors格式文件解析为state_dict保存
    if map_location is None:
        map_location = "cpu"
    state_dict = safetensors.torch.load_file(filename, device=map_location)
    state_dict = {decode_key(k): v for k, v in state_dict.items()}
    return unflatten_dict(state_dict)
