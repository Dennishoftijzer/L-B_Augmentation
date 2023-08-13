"""Helper functions used to create literate documentation from python files."""
import importlib
import inspect
import os
from typing import Optional, Sequence, List, cast

from typing.io import TextIO

from constants import ABS_PATH_OF_DOCS_DIR, ABS_PATH_OF_TOP_LEVEL_DIR


def get_literate_output_path(file: TextIO) -> Optional[str]:
    for l in file:
        l = l.strip()
        if l != "":
            if l.lower().startswith(("# literate", "#literate")):
                parts = l.split(":")
                if len(parts) == 1:
                    assert (
                        file.name[-3:].lower() == ".py"
                    ), "Can only run literate on python (*.py) files."
                    return file.name[:-3] + ".md"
                elif len(parts) == 2:
                    rel_outpath = parts[1].strip()
                    outpath = os.path.abspath(
                        os.path.join(ABS_PATH_OF_DOCS_DIR, rel_outpath)
                    )
                    assert outpath.startswith(
                        ABS_PATH_OF_DOCS_DIR
                    ), f"Path {outpath} is not allowed, must be within {ABS_PATH_OF_DOCS_DIR}."
                    return outpath
                else:
                    raise NotImplementedError(
                        f"Line '{l}' is not of the correct format."
                    )
            else:
                return None
    return None


def source_to_markdown(dot_path: str, summarize: bool = False):
    importlib.invalidate_caches()
    module_path, obj_name = ".".join(dot_path.split(".")[:-1]), dot_path.split(".")[-1]
    module = importlib.import_module(module_path)
    obj = getattr(module, obj_name)
    source = inspect.getsource(obj)

    if not summarize:
        return source
    elif inspect.isclass(obj):
        lines = source.split("\n")
        newlines = [lines[0]]
        whitespace_len = float("inf")
        k = 1
        started = False
        while k < len(lines):
            l = lines[k]
            lstripped = l.lstrip()
            if started:
                newlines.append(l)
                started = "):" not in l and "->" not in l
                if not started:
                    newlines.append(l[: cast(int, whitespace_len)] + "    ...\n")

            if (
                l.lstrip().startswith("def ")
                and len(l) - len(lstripped) <= whitespace_len
            ):
                whitespace_len = len(l) - len(lstripped)
                newlines.append(l)
                started = "):" not in l and "->" not in l
                if not started:
                    newlines.append(l[:whitespace_len] + "    ...\n")
            k += 1
        return "\n".join(newlines).strip()
    elif inspect.isfunction(obj):
        return source.split("\n")[0] + "\n    ..."
    else:
        return


def _strip_empty_lines(lines: Sequence[str]) -> List[str]:
    lines = list(lines)
    if len(lines) == 0:
        return lines

    for i in range(len(lines)):
        if lines[i].strip() != "":
            lines = lines[i:]
            break

    for i in reversed(list(range(len(lines)))):
        if lines[i].strip() != "":
            lines = lines[: i + 1]
            break
    return lines


def literate_python_to_markdown(path: str) -> bool:
    assert path[-3:].lower() == ".py", "Can only run literate on python (*.py) files."

    with open(path, "r") as file:
        output_path = get_literate_output_path(file)

        if output_path is None:
            return False

        output_lines = [
            f"<!-- DO NOT EDIT THIS FILE. --> ",
            f"<!-- THIS FILE WAS AUTOGENERATED FROM"
            f" 'ALLENACT_BASE_DIR/{os.path.relpath(path, ABS_PATH_OF_TOP_LEVEL_DIR)}', EDIT IT INSTEAD. -->\n",
        ]
        md_lines: List[str] = []
        code_lines = md_lines

        lines = file.readlines()
        mode = None

        for line in lines:
            line = line.rstrip()
            stripped_line = line.strip()
            if (mode is None or mode == "change") and line.strip() == "":
                continue

            if mode == "markdown":
                if stripped_line in ['"""', "'''"]:
                    output_lines.extend(_strip_empty_lines(md_lines) + [""])
                    md_lines.clear()
                    mode = None
                elif stripped_line.endswith(('"""', "'''")):
                    output_lines.extend(
                        _strip_empty_lines(md_lines) + [stripped_line[:-3]]
                    )
                    md_lines.clear()
                    mode = None
                    # TODO: Does not account for the case where a string is ended with a comment.
                else:
                    md_lines.append(line.strip())
            elif stripped_line.startswith(("# %%", "#%%")):
                last_mode = mode
                mode = "change"
                if last_mode == "code":
                    output_lines.extend(
                        ["```python"] + _strip_empty_lines(code_lines) + ["```"]
                    )
                    code_lines.clear()

                if " import " in stripped_line:
                    path = stripped_line.split(" import ")[-1].strip()
                    output_lines.append(
                        "```python\n" + source_to_markdown(path) + "\n```"
                    )
                elif " import_summary " in stripped_line:
                    path = stripped_line.split(" import_summary ")[-1].strip()
                    output_lines.append(
                        "```python\n"
                        + source_to_markdown(path, summarize=True)
                        + "\n```"
                    )
                elif " hide" in stripped_line:
                    mode = "hide"
            elif mode == "hide":
                continue
            elif mode == "change":
                if stripped_line.startswith(('"""', "'''")):
                    mode = "markdown"
                    if len(stripped_line) != 3:
                        if stripped_line.endswith(('"""', "'''")):
                            output_lines.append(stripped_line[3:-3])
                            mode = "change"
                        else:
                            output_lines.append(stripped_line[3:])
                else:
                    mode = "code"
                    code_lines.append(line)
            elif mode == "code":
                code_lines.append(line)
            else:
                raise NotImplementedError(
                    f"mode {mode} is not implemented. Last 5 lines: "
                    + "\n".join(output_lines[-5:])
                )

        if mode == "code" and len(code_lines) != 0:
            output_lines.extend(
                ["```python"] + _strip_empty_lines(code_lines) + ["```"]
            )

    with open(output_path, "w") as f:
        f.writelines([l + "\n" for l in output_lines])

    return True


if __name__ == "__main__":
    # print(
    #     source_to_markdown(
    #         "allenact_plugins.minigrid_plugin.minigrid_offpolicy.ExpertTrajectoryIterator",
    #         True
    #     )
    # )

    literate_python_to_markdown(
        os.path.join(
            ABS_PATH_OF_TOP_LEVEL_DIR,
            "projects/tutorials/training_a_pointnav_model.py",
        )
    )
