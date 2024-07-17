# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import shutil
import venv
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.common.utils import run_subprocess

# pylint: disable=redefined-outer-name


class DependencySetupEnvBuilder(venv.EnvBuilder):
    def post_setup(self, context) -> None:
        super().post_setup(context)
        # Install Olive only
        olive_root = str(Path(__file__).parents[3].resolve())
        run_subprocess([context.env_exe, "-Im", "pip", "install", olive_root], check=True)


@pytest.fixture()
def config_json(tmp_path):
    # create a user_script.py file in the tmp_path and refer to it
    # this way the test can be run from any directory
    tmp_path = Path(tmp_path)

    user_script_py = tmp_path / "user_script.py"
    with user_script_py.open("w") as f:
        f.write("")

    with (Path(__file__).parent / "mock_data" / "dependency_setup.json").open() as f:
        config_json = json.load(f)

    for i in range(len(config_json["data_configs"])):
        if "user_script" in config_json["data_configs"][i]:
            config_json["data_configs"][i]["user_script"] = user_script_py.as_posix()

    ep = ["DmlExecutionProvider" if platform.system() == OS.WINDOWS else "CUDAExecutionProvider"]
    for i in range(len(config_json["systems"]["local_system"]["config"]["accelerators"])):
        config_json["systems"]["local_system"]["config"]["accelerators"][i]["execution_providers"] = ep

    config_json_file = tmp_path / "config.json"
    with config_json_file.open("w") as f:
        json.dump(config_json, f)

    return str(config_json_file)


def test_dependency_setup(tmp_path, config_json):
    builder = DependencySetupEnvBuilder(with_pip=True)
    builder.create(str(tmp_path))

    if platform.system() == OS.WINDOWS:
        python_path = tmp_path / "Scripts" / "python"
        ort_extra = "onnxruntime-directml"
    else:
        python_path = tmp_path / "bin" / "python"
        ort_extra = "onnxruntime-gpu"

    user_script_config_file = config_json
    cmd = [
        python_path,
        "-Im",
        "olive.workflows.run",
        "--config",
        str(user_script_config_file),
        "--setup",
    ]

    return_code, _, stderr = run_subprocess(cmd, check=True)
    if return_code != 0:
        pytest.fail(stderr)

    _, outputs, _ = run_subprocess([python_path, "-Im", "pip", "list"], check=True)
    assert ort_extra in outputs
    shutil.rmtree(tmp_path, ignore_errors=True)
