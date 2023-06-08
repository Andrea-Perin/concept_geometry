from pathlib import Path
from datetime import datetime
from subprocess import Popen, PIPE
import json
import shutil
import sys
import __main__


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def get_current_hash():
    process = Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=PIPE)
    return process.communicate()[0].strip().decode('utf-8')


class Experiment(type(Path())):
    def save_dict(self, obj, name):
        with (self / name).open("w", encoding="UTF-8") as target:
            json.dump(obj, target, indent=4)


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
        self.tmpdir = Experiment(dir / f"{self.fname}_{self.dt}" / "results")
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
            # ask the user what to do
            if query_yes_no("Delete experiment folder?"):
                shutil.rmtree(self.tmpdir.parent)
            return None
        else:
            # zip the contents of the folder
            if self.zip:
                shutil.make_archive(self.tmpdir, "zip", self.tmpdir)
            return True
