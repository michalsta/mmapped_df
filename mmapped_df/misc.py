import subprocess


def shell(command: str, **kwargs):
    run = subprocess.run(command, shell=True, **kwargs)
    run.check_returncode()
    return run
