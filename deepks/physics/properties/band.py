"""Band property helpers."""


def get_bandgap(band, occ):
    return band[..., occ] - band[..., occ - 1]


def band_from_solution(band_solution):
    band_pred, _ = band_solution
    return band_pred
