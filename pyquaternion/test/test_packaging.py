import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _run_python(args, cwd):
    proc = subprocess.Popen(
        [sys.executable] + list(args),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            "Command {} failed with exit code {}:\n{}".format(
                args, proc.returncode, stderr.decode("utf-8", errors="replace")
            )
        )
    return stdout.decode("utf-8")


class TestSourceDistribution(unittest.TestCase):
    def test_sdist_includes_version_txt(self):
        tmpdir = tempfile.mkdtemp()
        try:
            src_dir = os.path.join(tmpdir, "src")
            dist_dir = os.path.join(tmpdir, "dist")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(src_dir)
            os.makedirs(dist_dir)
            os.makedirs(extract_dir)

            for filename in ("setup.py", "setup.cfg", "VERSION.txt", "README.md", "LICENSE.txt", "MANIFEST.in"):
                src_path = os.path.join(REPO_ROOT, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(src_dir, filename))
            shutil.copytree(
                os.path.join(REPO_ROOT, "pyquaternion"),
                os.path.join(src_dir, "pyquaternion"),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

            _run_python(["setup.py", "sdist", "--dist-dir", dist_dir], cwd=src_dir)

            archives = [name for name in os.listdir(dist_dir) if name.endswith(".tar.gz")]
            self.assertEqual(len(archives), 1)
            archive_path = os.path.join(dist_dir, archives[0])

            with tarfile.open(archive_path, "r:gz") as tf:
                members = tf.getnames()
                self.assertTrue(
                    any(member.endswith("/VERSION.txt") for member in members),
                    "VERSION.txt is missing from sdist archive: {}".format(members),
                )
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(extract_dir, filter="data")
                else:
                    tf.extractall(extract_dir)

            extracted_roots = os.listdir(extract_dir)
            self.assertEqual(len(extracted_roots), 1)
            pkg_dir = os.path.join(extract_dir, extracted_roots[0])

            version_out = _run_python(["setup.py", "--version"], cwd=pkg_dir).strip()
            with open(os.path.join(REPO_ROOT, "VERSION.txt")) as f:
                expected_version = f.read().strip()
            self.assertEqual(version_out, expected_version)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
