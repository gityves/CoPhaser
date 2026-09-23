"""Generate the standalone Python script shown in and run by the GUI.

The script is `cophaser.cli.train`'s own source, rewritten via its AST so that
`config[...]` lookups become literal values, `_resolve_trainer_kwargs` is inlined,
and the `def train(...)` wrapper is dropped, leaving a plain top-level script.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from cophaser import cli as _cli


class _TrainerKwargsInliner(ast.NodeTransformer):
    """Inline `trainer_kwargs = _resolve_trainer_kwargs(...)`: a literal dict for
    manual/fixed modes, a call to `auto_hyperparameters` for autotune (needs the model)."""

    def __init__(self, config: dict):
        self.config = config

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "trainer_kwargs"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_resolve_trainer_kwargs"
        ):
            cycle = self.config["cycle"]
            mode = self.config.get("hyperparams_mode") or (
                "autotune" if cycle == "cell_cycle" else "fixed"
            )
            if mode == "autotune":
                overrides = _to_builtin(self.config.get("trainer_kwargs", {}) or {})
                # same as cli._resolve_trainer_kwargs: a user-set prior re-derives the weights
                prior_arg = (
                    f", cycling_status_prior={overrides.get('cycling_status_prior')!r}"
                    if cycle == "cell_cycle"
                    else ""
                )
                src = (
                    f"result = auto_hyperparameters(model, cycle={cycle!r}, adata=adata, "
                    f"layer=layer{prior_arg})\n"
                    "trainer_kwargs = result['trainer']\n"
                    f"trainer_kwargs.update({overrides!r})\n"
                )
            else:
                resolved = _cli._resolve_trainer_kwargs(self.config, None, cycle)
                src = f"trainer_kwargs = {resolved!r}\n"
            return ast.parse(src).body
        return node


class _ConfigInliner(ast.NodeTransformer):
    """Replace `config[key]` / `config.get(key[, default])` with its literal value."""

    def __init__(self, config: dict):
        self.config = config

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "config":
            key = self._const_str(node.slice)
            if key is not None and key in self.config:
                return self._value_node(self.config[key])
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "config"
            and node.args
        ):
            key = self._const_str(node.args[0])
            if key is not None:
                default = ast.literal_eval(node.args[1]) if len(node.args) > 1 else None
                return self._value_node(self.config.get(key, default))
        return node

    @staticmethod
    def _const_str(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    @staticmethod
    def _value_node(value):
        return ast.parse(repr(_to_builtin(value)), mode="eval").body


def _to_builtin(value):
    """numpy scalars/arrays -> Python types, so their repr is valid in the script."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_to_builtin(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_builtin(v) for v in value)
    return value


def build_run_script(config: dict) -> str:
    header = (
        "import os\n"
        "import random\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "\n"
        "import scanpy as sc\n"
        "\n"
        "from cophaser import CoPhaser, Trainer, Loss, auto_hyperparameters, utils, cycle_configs\n"
        "\n"
        "\n"
    )
    train_fn = ast.parse(inspect.getsource(_cli.train)).body[0]
    train_fn = _TrainerKwargsInliner(config).visit(train_fn)
    train_fn = _ConfigInliner(config).visit(train_fn)

    # drop the `def train(config)` wrapper, keep its body
    flat = ast.Module(body=train_fn.body, type_ignores=[])
    ast.fix_missing_locations(flat)
    body = ast.unparse(flat)

    return header + body + "\n"
