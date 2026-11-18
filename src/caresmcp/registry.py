from pathlib import Path

import catalogue

pkg_name = Path(__file__).parent.name

IMAGE_PROC = catalogue.create(pkg_name, "image")
MEDAI = catalogue.create(pkg_name, "medai")
PROMPT = catalogue.create(pkg_name, "prompt")

CT = catalogue.create(pkg_name, "ct")
DSA = catalogue.create(pkg_name, "dsa")
MAMMOGRAPHY = catalogue.create(pkg_name, "mammography")
MRI = catalogue.create(pkg_name, "mri")
NUCLEARMED = catalogue.create(pkg_name, "nuclearmed")
PATHOLOGY = catalogue.create(pkg_name, "pathology")
ULTRASOUND = catalogue.create(pkg_name, "ultrasound")
XRAY = catalogue.create(pkg_name, "xray")

# categorical dict for tools/prompts
# strict 1-to-n mapping without overlap
REGISTRIES = {
    each_category.namespace[-1]: each_category
    for each_category in [
        IMAGE_PROC,
        MEDAI,
        CT,
        DSA,
        MAMMOGRAPHY,
        MRI,
        NUCLEARMED,
        PATHOLOGY,
        ULTRASOUND,
        XRAY,
    ]
}

# many-to-many mapping for tag for flexible association
TAG_MONAI = catalogue.create(pkg_name, "tag_monai")

TASK_CLS = catalogue.create(pkg_name, "task_classify")
TASK_REG = catalogue.create(pkg_name, "task_regression")
TASK_SEG = catalogue.create(pkg_name, "task_segment")
TASK_RET = catalogue.create(pkg_name, "task_retrieve")
TASK_CAP = catalogue.create(pkg_name, "task_caption")
TASK_EMB = catalogue.create(pkg_name, "task_embed")
TASK_LOC = catalogue.create(pkg_name, "task_localize")
TASK_DEN = catalogue.create(pkg_name, "task_denoise")
TASK_EVAL = catalogue.create(pkg_name, "task_evaluate")

TAGS = {name: value for name, value in locals().items() if name.startswith("TAG_")}
TASKS = {name: value for name, value in locals().items() if name.startswith("TASK_")}
