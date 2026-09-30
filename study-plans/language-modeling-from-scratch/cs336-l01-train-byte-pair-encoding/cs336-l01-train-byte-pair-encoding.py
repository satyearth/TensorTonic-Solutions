def train_bpe(corpus: list[str], vocab_size: int) -> dict:

    token_bytes = {i: bytes([i]) for i in range(256)}

    sequences = [
        list(s.encode("utf-8"))
        for s in corpus
    ]

    vocab = []
    merges = []

    next_id = 256

    while next_id < vocab_size:
        counts = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1

        if not counts:
            break

        best_pair = max(
            counts,
            key=lambda pair: (
                counts[pair],
                token_bytes[pair[0]],
                token_bytes[pair[1]],
            ),
        )

        left_id, right_id = best_pair
        new_id = next_id

        token_bytes[new_id] = (
            token_bytes[left_id] + token_bytes[right_id]
        )

        vocab.append([
            new_id,
            list(token_bytes[new_id]),
        ])

        merges.append([
            left_id,
            right_id,
            new_id,
        ])

        for seq_index, seq in enumerate(sequences):
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i + 1 < len(seq)
                    and seq[i] == left_id
                    and seq[i + 1] == right_id
                ):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[seq_index] = new_seq

        next_id += 1

    return {
        "vocab": vocab,
        "merges": merges,
    }