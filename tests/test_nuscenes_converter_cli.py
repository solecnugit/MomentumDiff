import argparse
import ast
import pathlib
import sys
import unittest
from unittest.mock import patch


CONVERTER = pathlib.Path(__file__).resolve().parents[1] / "tools/data_converter/nuscenes_converter.py"


def run_cli(*argv):
    tree = ast.parse(CONVERTER.read_text(encoding="utf-8"), filename=str(CONVERTER))
    cli_start = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets)
    )
    cli = ast.Module(body=tree.body[cli_start:], type_ignores=[])
    calls = []
    namespace = {
        "__name__": "__main__",
        "argparse": argparse,
        "nuscenes_data_prep": lambda **kwargs: calls.append(kwargs),
    }
    with patch.object(sys, "argv", [str(CONVERTER), "nuscenes", "--out-dir", "data/infos", *argv]):
        exec(compile(cli, str(CONVERTER), "exec"), namespace)
    return calls


class ConverterCliTest(unittest.TestCase):
    def test_full_version_generates_trainval_and_test(self):
        calls = run_cli("--version", "v1.0", "--root-path", "/data/nuscenes")
        self.assertEqual([call["version"] for call in calls], ["v1.0-trainval", "v1.0-test"])
        self.assertTrue(all(call["root_path"] == "/data/nuscenes" for call in calls))

    def test_mini_generates_only_mini(self):
        calls = run_cli("--version", "v1.0-mini")
        self.assertEqual([call["version"] for call in calls], ["v1.0-mini"])

    def test_skip_test_uses_only_trainval_tables(self):
        calls = run_cli("--version", "v1.0", "--skip-test")
        self.assertEqual([call["version"] for call in calls], ["v1.0-trainval"])


if __name__ == "__main__":
    unittest.main()
