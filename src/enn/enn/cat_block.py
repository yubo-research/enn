from enn._rust import cat_log_width


def log_code_width(card, dim_cap=48):
    return cat_log_width(int(card), int(dim_cap))
