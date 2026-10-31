from enn._rust import semantic_texts


def relationship_texts(names, output_names, descriptions, output_descriptions):
    output_names = tuple(output_names)
    if (
        not output_names
        or any(not isinstance(n, str) or not n.strip() for n in output_names)
        or len(set(output_names)) != len(output_names)
    ):
        raise ValueError("output names must be nonempty and unique")
    inputs, input_rows = semantic_texts(names, descriptions)
    outputs, output_rows = semantic_texts(output_names, output_descriptions)
    return semantic_texts(
        [inputs[j] for j in input_rows] + [outputs[j] for j in output_rows]
    )
