# coding=utf-8
# Copyright 2025 The Google Research Authors.
# Modifications Copyright 2026 Songlin Cai.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


__version__ = "0.1.0.dev0"

with open("requirements.txt", encoding="utf-8") as requirements_file:
  REQUIRED_PKGS = [
      line.strip()
      for line in requirements_file
      if line.strip() and not line.startswith("#")
  ]


setup(
    name="UCP-Clarify",
    include_package_data=True,
    package_data={"ucp_clarify": ["config/*.json", "utils/*.yaml"]},
    packages=find_packages("src", exclude={"tests"}),
    package_dir={"": "src"},
    install_requires=REQUIRED_PKGS,
    python_requires=">=3.10",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    version=__version__,
    license="Apache-2.0",
    description="UCP-Clarify reference implementation.",
    keywords="clarifying questions, dpo, transformers, huggingface, rlhf",
)
