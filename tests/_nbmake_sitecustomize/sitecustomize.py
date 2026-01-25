try:
    import jupyter_client.localinterfaces as _li

    def _disable_psutil_net_if_addrs() -> None:
        raise ImportError(
            "disabled psutil-based IP discovery for nbmake in this environment"
        )

    _li._load_ips_psutil = _disable_psutil_net_if_addrs
except Exception:
    pass
