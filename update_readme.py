#!/usr/bin/env python
"""Update README.rst from nvector._info."""

from pathlib import Path

from nvector import _info

REFERENCES = """

References
==========

.. [Gad10] K. Gade, `A Nonsingular Horizontal Position Representation,
           J. Navigation, 63(3):395-417, 2010.
           <http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_

.. [Kar13] C.F.F. Karney, `Algorithms for geodesics.
           J. Geodesy, 87(1):43-55, 2013.
           <https://rdcu.be/cccgm>`_

.. [GB25] K. Gade and P.A. Brodtkorb,
           `Nvector Documentation for Python, 2025.
           <https://nvector.readthedocs.io/en/latest>`_

"""


def update_readme() -> None:
    """Generate README.rst from nvector._info.__doc__."""

    text = _info.__doc__
    if not text:
        raise ValueError("nvector._info.__doc__ is empty")

    # Replace title
    old_header = (
        "Introduction to nvector\n"
        "=======================\n\n"
        #".. only:: html"
    )

    new_header = (
        "=======\n"
        "nvector\n"
        "=======\n"
    )

    text = text.replace(old_header, new_header, 1)


    # Replace sphinx-only references
    text = text.replace(
        "the :doc:`functional examples </tutorials/getting_started_functional>` section",
        "the functional examples section",
    )

    # Replace citations
    text = text.replace(":cite:`GadeAndBrodtkorb2025Nvector`", "[GB25]_")
    text = text.replace(":cite:`Gade2010Nonsingular`", "[Gad10]_")
    text = text.replace(":cite:`Karney2013Algorithms`", "[Kar13]_")

    # Find last ".. only:: html"
    marker = ".. only:: html"
    pos = text.rfind(marker)

    if pos >= 0:
        before = text[:pos]
        after = text[pos + len(marker):]

        # remove leading blank lines after marker
        after = after.lstrip("\n")

        # dedent by 4 spaces
        lines = []
        for line in after.splitlines():
            if line.startswith("    "):
                lines.append(line[4:])
            else:
                lines.append(line)

        text = before.rstrip() + "\n\n" + "\n".join(lines) + "\n"


    text = text.replace(".. only:: html", "")

    if "\nReferences\n==========\n" in text:
        text = text.split("\nReferences\n==========\n", 1)[0].rstrip()
    text = text.rstrip() + REFERENCES

    readme_file = Path("README.rst")
    readme_file.write_text(text, encoding="utf-8")

    print(f"Updated {readme_file}")


if __name__ == "__main__":
    update_readme()
