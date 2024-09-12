try:
    import pyximport

    pyximport.install(language_level="3")

    from .auroc import calc_auroc, csr_to_csc
except ModuleNotFoundError:
    pass
