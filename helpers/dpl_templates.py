"""Source-domain-only text side information, transcribed from saved FASTPT configs."""
TEMPLATES = {
    "PACS": {"art_painting": "an artistic painting of a {}.", "cartoon": "a cartoon drawing of a {}.",
             "photo": "a realistic photo of a {}.", "sketch": "a pencil drawing of a {}."},
    "VLCS": {"Caltech101": "a clean object-centric image of a {} from Caltech101",
             "LabelMe": "an image from the LabelMe dataset showing a {}",
             "SUN09": "a scene photo from the SUN09 dataset that contains a {}",
             "VOC2007": "a PASCAL VOC-style photo of a {}."},
    "OfficeHome": {"Art": "an artistic painting of a {}.", "Clipart": "clipart of a {}.",
                   "Product": "a product image of a {} with clean background.", "RealWorld": "a realistic photo of a {}."},
    "DomainNet": {"clipart": "clipart of a {}.", "infograph": "an infographic showing a {}",
                  "painting": "a painting of a {}", "quickdraw": "a quickdraw of a {}",
                  "real": "a realistic photo of a {}", "sketch": "a pencil drawing of a {}."},
}
CAPTIONS = {
    "PACS": {"art_painting": "an artistic painting", "cartoon": "a cartoon drawing",
             "photo": "a realistic photo", "sketch": "a pencil drawing"},
    "VLCS": {"Caltech101": "a clean object-centric image from Caltech101",
             "LabelMe": "an image from the LabelMe dataset", "SUN09": "a scene photo from the SUN09 dataset",
             "VOC2007": "a PASCAL VOC-style photo"},
    "OfficeHome": {"Art": "an artistic painting", "Clipart": "clipart",
                   "Product": "a product image with clean background", "RealWorld": "a realistic photo"},
    "DomainNet": {"clipart": "clipart", "infograph": "an infographic", "painting": "a painting",
                  "quickdraw": "a quickdraw", "real": "a realistic photo", "sketch": "a pencil drawing"},
}


def source_texts(manifest):
    dataset = manifest["dataset"]
    names = manifest["source_domains"]
    if manifest["target_domain"] in names:
        raise ValueError("Held-out target cannot enter source text conditions")
    return ({name: [TEMPLATES[dataset][name].format(c.replace("_", " ")) for c in manifest["classes"]]
             for name in names}, {name: CAPTIONS[dataset][name] for name in names})
