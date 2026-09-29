import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.release.sanitize_checkpoint import sanitize_archive


class SanitizeCheckpointTest(unittest.TestCase):
    def test_redacts_metadata_without_changing_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            destination = Path(directory) / "public.pth"
            metadata = {
                "env_info": "CUDA_HOME: /home/researcher/miniconda3",
                "last_ckpt": "/home/researcher/work/iter.pth",
            }
            tensor = b"\x00\x01\x02\xff" * 2048
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("archive/data.pkl", pickle.dumps(metadata, protocol=2))
                archive.writestr("archive/data/0", tensor)

            sanitize_archive(source, destination)

            with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination) as clean:
                self.assertEqual(clean.testzip(), None)
                self.assertEqual(clean.read("archive/data/0"), tensor)
                self.assertEqual(clean.getinfo("archive/data/0").CRC, original.getinfo("archive/data/0").CRC)
                self.assertEqual(pickle.loads(original.read("archive/data.pkl")), metadata)
                sanitized = pickle.loads(clean.read("archive/data.pkl"))
                self.assertNotIn("researcher", str(sanitized))
                self.assertIn("/home/", sanitized["env_info"])


if __name__ == "__main__":
    unittest.main()
