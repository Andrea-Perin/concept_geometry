from pathlib import Path
from datetime import datetime
from subprocess import Popen, PIPE
import json
import shutil
import __main__


def get_current_hash():
    process = Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=PIPE)
    return process.communicate()[0].strip().decode('utf-8')


class ExpLogger:
    """Context manager to save experiments and save things."""

    def __init__(self, dir=Path('./experiments'), zip=True):
        self.zip = zip
        # simply collect metadata
        self.git_hash = get_current_hash()
        print(f"Current hash: {self.git_hash}")
        self.fname = Path(__main__.__file__).stem
        print(f"Current fname: {self.fname}")
        self.dt = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        self.tmpdir = dir / f"{self.fname}_{self.dt}" / "results"
        print(f"Saving in directory {self.tmpdir}")
        # make the tmpdir (with parents) if not exist
        # and save the metadata already
        self.tmpdir.mkdir(parents=True, exist_ok=False)
        meta = dict(hash=self.git_hash, dt=self.dt)
        with (self.tmpdir / "meta.json").open("w", encoding="UTF-8") as target:
            json.dump(meta, target, indent=4)

    def __enter__(self):
        return self.tmpdir

    def __exit__(self, exc_type, exc_value, traceback):
        # here we catch all exceptions as deadly
        # so we cleanup the whole directory in such case
        if exc_type is not None:
            shutil.rmtree(self.tmpdir)
        else:
            # zip the contents of the folder
            if self.zip:
                shutil.make_archive(self.tmpdir, "zip", self.tmpdir)
        return True
