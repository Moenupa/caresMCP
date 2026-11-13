import filetype


def is_dicom(obj: bytes) -> bool:
    return filetype.guess_mime(obj) == "application/dicom"
