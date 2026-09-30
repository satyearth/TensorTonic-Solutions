def encode(text: str, merges: list[list[int]]) -> list[int]:

    ids = list(text.encode("utf-8"))

    for left, right, new in merges:
        result = []
        i = 0

        while i < len(ids):

            if (
                i + 1 < len(ids)
                and ids[i] == left
                and ids[i + 1] == right
            ):
                result.append(new)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        ids = result

    return ids

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:

    data = bytearray()

    for token_id in ids:
        data.extend(vocab[token_id])

    return bytes(data).decode("utf-8")