import numpy as np
from bitarray import bitarray
from typing import Dict, Any
from heapq import heapify, heappop, heappush


def count_letter(arr: np.ndarray) -> Dict:
    symbols, counts = np.unique(arr, return_counts=True)
    return dict(zip(symbols, counts))


def create_huffman_table(arr: np.ndarray) -> Dict:
    value_counts = count_letter(arr)
    huffman_tree = [[value_counts[symbol], [symbol, '']] for symbol in value_counts]
    heapify(huffman_tree)

    while len(huffman_tree) > 1:
        first_min = heappop(huffman_tree)
        second_min = heappop(huffman_tree)

        for pair in first_min[1:]:
            pair[1] = '1' + pair[1]
        for pair in second_min[1:]:
            pair[1] = '0' + pair[1]

        new_elem = [first_min[0] + second_min[0]] + first_min[1:] + second_min[1:]
        heappush(huffman_tree, new_elem)

    huffman_dict = {pair[0]: bitarray(pair[1]) for pair in huffman_tree[0][1:]}
    return huffman_dict


def encode_huffman(arr: np.ndarray) -> Any:
    huffman_dict = create_huffman_table(arr)
    encoded_text = bitarray()
    encoded_text.encode(huffman_dict, "".join(arr))
    return encoded_text, huffman_dict


def decode_huffman(text: bitarray, huffman_dict: Dict) -> Any:
    decoded_text = text
    decoded_text = decoded_text.decode(huffman_dict)
    decoded_text = ''.join(decoded_text)
    return decoded_text


def test_huffman():
    text = ("abracadabraz")
    assert decode_huffman(*encode_huffman(np.array(list(text)))) == "abracadabraz", "Should be abracadabraz"


if __name__ == "__main__":
    test_huffman()
    print("Everything passed")

