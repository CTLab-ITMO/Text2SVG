import random
import xml.etree.ElementTree as ET

def _rand_hex() -> str:
    """Return a random #RRGGBB colour string."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def augment_svg(
    svg_text: str,
    *,
    change_root_prob: float = 0.10,
    seed: int | None = None
) -> str:
    """
    Augment an SVG by either:
      • randomising the root <svg> fill (with probability `change_root_prob`)
        and removing all other fills, or
      • randomising each <path>'s fill.

    Parameters
    ----------
    svg_text : str
        Original SVG markup.
    change_root_prob : float, default 0.10
        Chance of switching to the 'global-fill' mode.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    str
        Augmented SVG markup.
    """
    if seed is not None:
        random.seed(seed)

    root = ET.fromstring(svg_text)
    use_root_fill = random.random() < change_root_prob

    if use_root_fill:
        # 1️⃣  Pick a new colour for the root and apply it
        root.set("fill", _rand_hex())

        # 2️⃣  Remove individual fills so they inherit the root colour
        for el in root.iter():
            if "fill" in el.attrib and el is not root:
                del el.attrib["fill"]
    else:
        # Random-colour each <path> (namespace-agnostic)
        for el in root.iter():
            if el.tag.split('}')[-1] == "path":
                el.set("fill", _rand_hex())

    return ET.tostring(root, encoding="unicode")