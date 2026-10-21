from pathlib import Path

import catalogue

pkg_name = Path(__file__).parent.name

IMAGE_PROC = catalogue.create(pkg_name, "image")
MEDAI = catalogue.create(pkg_name, "medai")
PROMPT = catalogue.create(pkg_name, "prompt")
PATHOLOGY = catalogue.create(pkg_name, "pathology")
XRAY = catalogue.create(pkg_name, "xray")

# categorical dict for tools/prompts
# strict 1-to-n mapping without overlap
REGISTRIES = {
    "image": IMAGE_PROC,
    "medai": MEDAI,
    "prompt": PROMPT,
    "pathology": PATHOLOGY,
    "xray": XRAY,
}

# many-to-many mapping for tag for flexible association
TAG_MONAI = catalogue.create(pkg_name, "tag_monai")
TAG_MRI = catalogue.create(pkg_name, "tag_mri")
TAG_XRAY = catalogue.create(pkg_name, "tag_xray")
TAG_CT = catalogue.create(pkg_name, "tag_ct")

TASK_CLS = catalogue.create(pkg_name, "task_classify")
TASK_SEG = catalogue.create(pkg_name, "task_segment")
TASK_CLIP = catalogue.create(pkg_name, "task_clip")
TASK_EMBED = catalogue.create(pkg_name, "task_embed")

TAGS = {name: value for name, value in locals().items() if name.startswith("TAG_")}
TASKS = {name: value for name, value in locals().items() if name.startswith("TASK_")}
